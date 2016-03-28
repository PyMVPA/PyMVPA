# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Searchlight-based hyperalignment"""

import os
import numpy as np

from tempfile import mktemp
from numpy.linalg import LinAlgError
from scipy.sparse import coo_matrix, csc_matrix

import mvpa2
from mvpa2.base.state import ClassWithCollections
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import *
from mvpa2.algorithms.hyperalignment import Hyperalignment
from mvpa2.measures.base import Measure
from mvpa2.datasets import Dataset, vstack
from mvpa2.mappers.staticprojection import StaticProjectionMapper
from mvpa2.misc.neighborhood import IndexQueryEngine, Sphere
from mvpa2.base.progress import ProgressBar
from mvpa2.base.hdf5 import h5save, h5load
from mvpa2.base import externals, warning
from mvpa2.support import copy
from mvpa2.featsel.helpers import FractionTailSelector, FixedNElementTailSelector

from mvpa2.support.due import due, Doi

if __debug__:
    from mvpa2.base import debug
    if 'SHPAL' in debug.active:
        def _shpaldebug(msg):
            debug('SHPAL', "%s" % msg)
    else:
        def _shpaldebug(*args):
            return None
else:
    def _shpaldebug(*args):
        return None

@due.dcite(
    Doi('10.1016/j.neuron.2011.08.026'),
    description="Per-feature measure of maximal correlation to features in other datasets",
    tags=["implementation"])
def compute_feature_scores(datasets, exclude_from_model=None):
    """
    Takes a list of datasets and computes a magical feature
    score for each feature in each dataset
    :ref:`Haxby et al., Neuron (2011) <HGC+11>`

    Parameters
    ----------
    datasets : list or tuple of datasets

    exclude_from_model: list of dataset indices that won't participate
                    in voxel selection of others

    Returns
    -------
    list : a list of feature scores; higher the score, better the feature

    NOTE: This function assumes that the datasets are zscored
    """
    if exclude_from_model is None:
        exclude_from_model = []
    feature_scores = [np.zeros(sd.nfeatures) for sd in datasets]
    for i, sd in enumerate(datasets):
        for j, sd2 in enumerate(datasets[i + 1:]):
            corr_temp = np.dot(sd.samples.T, sd2.samples)
            if j + i + 1 not in exclude_from_model:
                feature_scores[i] += np.max(corr_temp, axis=1)
            if i not in exclude_from_model:
                feature_scores[j + i + 1] += np.max(corr_temp, axis=0)
    return feature_scores

# XXX Is it really a measure or a Mapper or just a Node???
class HyperalignmentMeasure(Measure):
    """Feature selection and hyperalignment in a single node

    HyperalignmentMeasure combines feature selection and hyperalignment
    into a single node. This facilitates its usage in any searchlight
    or ROI.
    """
    is_trained = True

    def __init__(self, hyperalignment=Hyperalignment(ref_ds=0),
                 featsel=1.0, full_matrix=True, use_same_features=False,
                 exclude_from_model=None, dtype='float32', **kwargs):
        Measure.__init__(self, **kwargs)
        self.hyperalignment = hyperalignment
        self.featsel = featsel
        self.use_same_features = use_same_features
        self.exclude_from_model = exclude_from_model
        if self.exclude_from_model is None:
            self.exclude_from_model = []
        self.full_matrix = full_matrix
        self.dtype = dtype

    def _call(self, ds):
        ref_ds = self.hyperalignment.params.ref_ds
        nsamples, nfeatures = ds[ref_ds].shape
        if 'roi_seed' in ds[ref_ds].fa:
            seed_index = np.where(ds[ref_ds].fa.roi_seed)
        else:
            seed_index = None
            self.full_matrix = True
        # Voxel selection within Searchlight
        # Usual metric of between-subject between-voxel correspondence
        if self.featsel != 1.0:
            # computing feature scores from the data
            feature_scores = compute_feature_scores(ds, self.exclude_from_model)
            if self.featsel < 1.0:
                fselector = FractionTailSelector(self.featsel, tail='upper', mode='select', sort=False)
            else:
                fselector = FixedNElementTailSelector(np.floor(self.featsel), tail='upper', mode='select', sort=False)
            # XXX Artificially make the seed_index feature score high to keep it(?)
            if self.use_same_features:
                if len(self.exclude_from_model):
                    feature_scores = [feature_scores[ifs] for ifs in range(len(ds))
                                      if ifs not in self.exclude_from_model]
                feature_scores = np.mean(np.asarray(feature_scores), axis=0)
                if seed_index is not None:
                    feature_scores[seed_index] = max(feature_scores)
                features_selected = fselector(feature_scores)
                ds = [sd[:, features_selected] for sd in ds]
            else:
                features_selected = []
                for fs in feature_scores:
                    if seed_index is not None:
                        fs[seed_index] = max(fs)
                    features_selected.append(fselector(fs))
                ds = [sd[:, fsel] for fsel, sd in zip(features_selected, ds)]
        # Try hyperalignment
        try:
            # it is crucial to retrain hyperalignment, otherwise it would simply
            # project into the common space of a previous iteration
            if len(self.exclude_from_model) == 0:
                self.hyperalignment.train(ds)
            else:
                self.hyperalignment.train([ds[i] for i in range(len(ds))
                                           if i not in self.exclude_from_model])
            mappers = self.hyperalignment(ds)
            if mappers[0].proj.dtype is self.dtype:
                mappers = [m.proj for m in mappers]
            else:
                mappers = [m.proj.astype(self.dtype) for m in mappers]
            if self.featsel != 1.0:
                # Reshape the projection matrix from selected to all features
                mappers_full = [np.zeros((nfeatures, nfeatures)) for im in range(len(mappers))]
                if self.use_same_features:
                    for mf, m in zip(mappers_full, mappers):
                        mf[np.ix_(features_selected, features_selected)] = m
                else:
                    for mf, m, fsel in zip(mappers_full, mappers, features_selected):
                        mf[np.ix_(fsel, features_selected[ref_ds])] = m
                mappers = mappers_full
        except LinAlgError:
            print "SVD didn't converge. Try with a new reference, may be."
            mappers = [np.eye(nfeatures, dtype='int')] * len(ds)
        # Extract only the row/column corresponding to the center voxel if full_matrix is False
        if not self.full_matrix:
            mappers = [np.squeeze(m[:, seed_index]) for m in mappers]
        # Package results
        results = np.asanyarray([{'proj': mapper} for mapper in mappers])
        # Add residual errors to the seed voxel to be used later to weed out bad SLs(?)
        if 'residual_errors' in self.hyperalignment.ca.enabled:
            [result.update({'residual_error': self.hyperalignment.ca['residual_errors'][ires]})
             for ires, result in enumerate(results)]
        return Dataset(samples=results)


class SearchlightHyperalignment(ClassWithCollections):
    """
    Given a list of datasets, provide a list of mappers
    into common space using searchlight based hyperalignment.
    :ref:`Guntupalli et al., Cerebral Cortex (2016)`

    1) Input datasets should all be of the same size in terms of
    nsamples and nfeatures, and be coarsely aligned (using anatomy).
    2) All features in all datasets should be zscored.
    3) Datasets should have feature attribute `voxel_indices`
    containing spatial coordinates of all features
    """

    # TODO: add {training_,}residual_errors .ca ?

    ## Parameters common with Hyperalignment but overriden

    ref_ds = Parameter(0, constraints=EnsureInt() & EnsureRange(min=0),
        doc="""Index of a dataset to use as a reference. First dataset is used
            as default. If you supply exclude_from_model list, you should supply
            the ref_ds index as index after you remove those excluded datasets.
            Note that unlike regular Hyperalignment, there is no automagic
            choosing of the "best" ref_ds by default.""")

    ## Parameters specific to SearchlightHyperalignment

    # TODO: atm hardcodes to use Sphere.  Theoretically can easily allow any neighborhood
    radius = Parameter(
        3,
        constraints=EnsureInt() & EnsureRange(min=1),
        doc=""" radius of searchlight in number of voxels""")

    nproc = Parameter(
        1,
        constraints=EnsureInt() & EnsureRange(min=1) | EnsureNone(),
        doc="""Number of cores to use.""")

    nblocks = Parameter(
        100,
        constraints=EnsureInt() & EnsureRange(min=1) | EnsureNone(),
        doc="""Number of blocks to divide to process. Higher number results in
            smaller memory consumption.""")

    sparse_radius = Parameter(
        None,
        constraints=(EnsureRange(min=1) & EnsureInt() | EnsureNone()),
        doc="""Radius supplied to scatter_neighborhoods in units of voxels.
            This is effectively the distance between the centers where
            hyperalignment is performed in searchlights.
            If None, hyperalignment is performed at every voxel (default).""")

    hyperalignment = Parameter(
        Hyperalignment(ref_ds=0),
        doc="""Hyperalignment instance to be used in each searchlight sphere.
            Default is just the Hyperalignment instance with default parameters.
            """)

    combine_neighbormappers = Parameter(
        True,
        constraints=EnsureBool(),
        doc="""This param determines whether to combine mappers for each voxel
            from its neighborhood searchlights or just use the mapper for which
            it is the center voxel.  Use this option with caution, as enabling
            it might square the runtime memory requirement. If you run into
            memory issues, reduce the nproc in sl. """)

    compute_recon = Parameter(
        True,
        constraints=EnsureBool(),
        doc="""This param determines whether to compute reverse mappers for each
            subject from common-space to subject space. These will be stored in
            the StaticProjectionMapper() and used when reverse() is called.
            Enabling it will double the size of the mappers returned.""")

    featsel = Parameter(
        1.0,
        constraints=EnsureFloat() & EnsureRange(min=0.0, max=1.0) |
            EnsureInt() & EnsureRange(min=2),
        doc="""Determines if feature selection will be performed in each searchlight.
            1.0: Use all features. < 1.0 is understood as selecting that
            proportion of features in each searchlight using feature scores;
            > 1.0 is understood as selecting at most that many features in each
            searchlight.""")

    # TODO: Should we get rid of this feature?
    use_same_features = Parameter(
        False,
        constraints=EnsureBool(),
        doc="""Select the same (best) features when doing feature selection for
            all datasets.""")

    exclude_from_model = Parameter(
        [],
        constraints=EnsureListOf(int),
        doc="""List of dataset indices that will not participate in building
            common model.  These will still get mappers back but they don't
            influence the model or voxel selection.""")

    mask_node_ids = Parameter(
        None,
        constraints=EnsureListOf(int) | EnsureNone(),
        doc="""You can specify a mask to compute searchlight hyperalignment only
            within this mask.  These would be a list of voxel indices.""")

    dtype = Parameter(
        'float32',
        constraints='str',
        doc="""dtype of elements transformation matrices to save on memory for
            big datasets""")

    results_backend = Parameter(
        'hdf5',
        constraints=EnsureChoice('hdf5', 'native'),
        doc="""'hdf5' or 'native'. See Searchlight documentation.""")

    tmp_prefix = Parameter(
        'tmpsl',
        constraints='str',
        doc="""Prefix for temporary files. See Searchlight documentation.""")

    def __init__(self, **kwargs):
        _shpaldebug("Initializing.")
        ClassWithCollections.__init__(self, **kwargs)
        self.ndatasets = 0
        self.nfeatures = 0
        self.projections = None
        self.projections_recon = None
        if self.params.nproc is not None and self.params.nproc > 1 \
                and not externals.exists('pprocess'):
            raise RuntimeError("The 'pprocess' module is required for "
                               "multiprocess searchlights. Please either "
                               "install python-pprocess, or reduce `nproc` "
                               "to 1 (got nproc=%i) or set to default None"
                               % self.params.nproc)

    def _proc_block(self, block, datasets, measure, qe, seed=None, iblock='main'):
        if seed is not None:
            mvpa2.seed(seed)
        if __debug__:
            debug('SLC', 'Starting computing block for %i elements' % len(block))
        bar = ProgressBar()
        projections = [csc_matrix((self.nfeatures, self.nfeatures),
                                  dtype=self.params.dtype)
                       for isub in range(self.ndatasets)]
        for i, node_id in enumerate(block):
            # retrieve the feature ids of all features in the ROI from the query
            # engine

            # Find the neighborhood for that selected nearest node
            roi_feature_ids = qe[node_id]
            # if qe returns zero-sized ROI for any subject, pass...
            if len(roi_feature_ids) == 0:
                continue
            # selecting neighborhood for all subject for hyperalignment
            ds_temp = [sd[:, roi_feature_ids] for sd in datasets]
            roi_seed = np.array(roi_feature_ids) == node_id
            ds_temp[self.params.ref_ds].fa['roi_seed'] = roi_seed
            if __debug__:
                msg = 'ROI (%i/%i), %i features' % (i + 1, len(block), len(roi_seed))
                debug('SLC', bar(float(i + 1) / len(block), msg), cr=True)
            hmappers = measure(ds_temp)
            hmappers = hmappers.samples
            for isub in range(len(hmappers)):
                if not self.params.combine_neighbormappers:
                    I = roi_feature_ids
                    #J = [roi_feature_ids[node_id]] * len(roi_feature_ids)
                    J = [node_id] * len(roi_feature_ids)
                    V = hmappers[isub][0]['proj'].tolist()
                else:
                    I = []
                    J = []
                    V = []
                    for f2 in xrange(len(roi_feature_ids)):
                        I += roi_feature_ids
                        J += [roi_feature_ids[f2]] * len(roi_feature_ids)
                        V += hmappers[isub][0]['proj'][:, f2].tolist()
                proj = coo_matrix(
                    (V, (I, J)),
                    shape=(max(self.nfeatures, max(I) + 1), max(self.nfeatures, max(J) + 1)),
                    dtype=self.params.dtype)
                proj = proj.tocsc()
                # Cleaning up the current subject's projections to free up memory
                hmappers[isub, ] = [[] for _ in xrange(hmappers.shape[1])]
                projections[isub] = projections[isub] + proj

        if self.params.results_backend == 'native':
            return projections
        elif self.params.results_backend == 'hdf5':
            # store results in a temporary file and return a filename
            results_file = mktemp(prefix=self.params.tmp_prefix,
                                  suffix='-%s.hdf5' % iblock)
            if __debug__:
                debug('SLC', "Storing results into %s" % results_file)
            h5save(results_file, projections)
            if __debug__:
                debug('SLC_', "Results stored")
            return results_file
        else:
            raise RuntimeError("Must not reach this point")

    def __handle_results(self, results):
        if self.params.results_backend == 'hdf5':
            # 'results' must be just a filename
            assert(isinstance(results, str))
            if __debug__:
                debug('SLC', "Loading results from %s" % results)
            results_data = h5load(results)
            os.unlink(results)
            if __debug__:
                debug('SLC_', "Loaded results of len=%d from"
                      % len(results_data))
            for isub, res in enumerate(results_data):
                self.projections[isub] = self.projections[isub] + res
                if self.params.compute_recon:
                    self.projections_recon[isub] = self.projections_recon[isub] + res.T
            return

    def __handle_all_results(self, results):
        """Helper generator to decorate passing the results out to
        results_fx
        """
        for r in results:
            yield self.__handle_results(r)

    @due.dcite(
        Doi('10.1093/cercor/bhw068'),
        description="Full cortex hyperalignment of data to a common space",
        tags=["implementation"])
    def __call__(self, datasets):
        """Estimate mappers for each dataset using searchlight-based
        hyperalignment.

        Parameters
        ----------
          datasets : list or tuple of datasets

        Returns
        -------
        A list of trained StaticProjectionMappers of the same length as datasets
        """
        self.ndatasets = len(datasets)
        params = self.params

        _shpaldebug("SearchlightHyperalignment %s for %i datasets"
                    % (self, self.ndatasets))
        if params.ref_ds in params.exclude_from_model:
            raise ValueError("Requested reference dataset %i is also "
                             "in the exclude list." % params.ref_ds)
        if params.ref_ds != params.hyperalignment.params.ref_ds:
            warning('Supplied ref_ds & hyperalignment instance ref_ds:%d differ.'
                    % params.hyperalignment.params.ref_ds)
            warning('Using default hyperalignment instance with ref_ds: %d' % params.ref_ds)
            params.hyperalignment = Hyperalignment(ref_ds=params.ref_ds)
        if params.ref_ds >= self.ndatasets:
            raise ValueError("Requested reference dataset %i is out of "
                             "bounds. We have only %i datasets provided"
                             % (params.ref_ds, self.ndatasets))
        if len(params.exclude_from_model) > 0:
            warning("These datasets will not participate in building common "
                    "model: %s" % params.exclude_from_model)

        # Setting up SearchlightHyperalignment
        # we need to know which original features where comprising the
        # individual SL ROIs
        _shpaldebug('Initializing HyperalignmentMeasure.')
        hmeasure = HyperalignmentMeasure(
            featsel=params.featsel,
            hyperalignment=params.hyperalignment,
            full_matrix=params.combine_neighbormappers,
            use_same_features=params.use_same_features,
            exclude_from_model=params.exclude_from_model,
            dtype=params.dtype)

        # Performing SL processing manually
        _shpaldebug("Setting up for searchlights")
        if params.nproc is None and externals.exists('pprocess'):
            import pprocess
            try:
                params.nproc = pprocess.get_number_of_cores() or 1
            except AttributeError:
                warning("pprocess version %s has no API to figure out maximal "
                        "number of cores. Using 1"
                        % externals.versions['pprocess'])
                params.nproc = 1

        # XXX I think this class should already accept a single dataset only.
        # It should have a ``space`` setting that names a sample attribute that
        # can be used to identify individual/original datasets.
        # Taking a single dataset as argument would be cleaner, because the
        # algorithm relies on the assumption that there is a coarse feature
        # alignment, i.e. the SL ROIs cover roughly the same area
        _shpaldebug('Setting up query engine.')
        qe = IndexQueryEngine(voxel_indices=Sphere(params.radius))
        qe.train(datasets[params.ref_ds])
        self.nfeatures = datasets[params.ref_ds].nfeatures
        _shpaldebug("Performing Hyperalignment in searchlights")
        # Setting up centers for running SL Hyperalignment
        if params.sparse_radius is None:
            roi_ids = params.mask_node_ids if params.mask_node_ids is not None else qe.ids
        else:
            _shpaldebug("Setting up sparse neighborhood")
            from mvpa2.misc.neighborhood import scatter_neighborhoods
            if params.mask_node_ids is None:
                scoords, sidx = scatter_neighborhoods(
                    Sphere(params.sparse_radius),
                    datasets[params.ref_ds].fa.voxel_indices,
                    deterministic=True)
                roi_ids = sidx
            else:
                scoords, sidx = scatter_neighborhoods(
                    Sphere(params.sparse_radius),
                    datasets[params.ref_ds].fa.voxel_indices[params.mask_node_ids],
                    deterministic=True)
                roi_ids = [params.mask_node_ids[sid] for sid in sidx]

        # Initialize projections
        _shpaldebug('Initializing projection matrices')
        self.projections = [
            csc_matrix((self.nfeatures, self.nfeatures), dtype=params.dtype)
            for isub in range(self.ndatasets)]
        if params.compute_recon:
            self.projections_recon = [
                csc_matrix((self.nfeatures, self.nfeatures), dtype=params.dtype)
                for isub in range(self.ndatasets)]

        # compute
        if params.nproc is not None and params.nproc > 1:
            # split all target ROIs centers into `nproc` equally sized blocks
            nproc_needed = min(len(roi_ids), params.nproc)
            params.nblocks = nproc_needed \
                if params.nblocks is None else params.nblocks
            params.nblocks = min(len(roi_ids), params.nblocks)
            node_blocks = np.array_split(roi_ids, params.nblocks)
            # the next block sets up the infrastructure for parallel computing
            # this can easily be changed into a ParallelPython loop, if we
            # decide to have a PP job server in PyMVPA
            import pprocess
            p_results = pprocess.Map(limit=nproc_needed)
            if __debug__:
                debug('SLC', "Starting off %s child processes for nblocks=%i"
                      % (nproc_needed, params.nblocks))
            compute = p_results.manage(
                        pprocess.MakeParallel(self._proc_block))
            seed = mvpa2.get_random_seed()
            for iblock, block in enumerate(node_blocks):
                # should we maybe deepcopy the measure to have a unique and
                # independent one per process?
                compute(block, datasets, copy.copy(hmeasure), qe,
                        seed=seed, iblock=iblock)
        else:
            # otherwise collect the results in an 1-item list
            _shpaldebug('Using 1 process to compute mappers.')
            p_results = [self._proc_block(roi_ids, datasets, hmeasure, qe)]
        results_ds = self.__handle_all_results(p_results)
        # Dummy iterator for, you know, iteration
        list(results_ds)

        _shpaldebug('Wrapping projection matrices into StaticProjectionMappers')
        if params.compute_recon:
            self.projections = [
                StaticProjectionMapper(proj=proj, recon=proj_recon)
                for proj, proj_recon in zip(self.projections, self.projections_recon)]
        else:
            self.projections = [
                StaticProjectionMapper(proj=proj)
                for proj in self.projections]
        return self.projections
