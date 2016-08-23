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

import mvpa2
from mvpa2.base.state import ClassWithCollections
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import *
from mvpa2.algorithms.hyperalignment import Hyperalignment
from mvpa2.mappers.staticprojection import StaticProjectionMapper
from mvpa2.misc.neighborhood import IndexQueryEngine, Sphere
from mvpa2.base.progress import ProgressBar
from mvpa2.base import externals, warning
from mvpa2.support import copy
from mvpa2.featsel.helpers import FixedNElementTailSelector
from mvpa2.base.types import is_datasetlike
from mvpa2.misc.surfing.queryengine import SurfaceVerticesQueryEngine

if externals.exists('h5py'):
    from mvpa2.base.hdf5 import h5save, h5load

if externals.exists('scipy'):
    from scipy.sparse import coo_matrix, csc_matrix

from mvpa2.support.due import due, Doi


# A little debug helper to avoid constant if __debug__ conditioning,
# but it also means that debugging could not be activated at run time
# after the import of this module
def _shpaldebug(*args):
    pass
if __debug__:
    from mvpa2.base import debug
    if 'SHPAL' in debug.active:
        def _shpaldebug(msg):
            debug('SHPAL', "%s" % msg)


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


class FeatureSelectionHyperalignment(ClassWithCollections):
    """A helper which brings feature selection and hyperalignment in a single call

    It was created to facilitate its usage in any searchlight or ROI, where
    hyperalignment is trained and called on the same list of datasets.  But
    unlike Hyperalignment, it doesn't need to be separately trained.
    Called with a list of datasets, it returns a list of Mappers, one per each
    dataset, similarly to how Hyperalignment does.

    """


    def __init__(self, hyperalignment=Hyperalignment(ref_ds=0),
                 featsel=1.0, full_matrix=True, use_same_features=False,
                 exclude_from_model=None, dtype='float32', **kwargs):
        """
        For description of parameters see :class:`SearchlightHyperalignment`
        """
        super(FeatureSelectionHyperalignment, self).__init__(**kwargs)
        self.hyperalignment = hyperalignment
        self.featsel = featsel
        self.use_same_features = use_same_features
        self.exclude_from_model = exclude_from_model
        if self.exclude_from_model is None:
            self.exclude_from_model = []
        self.full_matrix = full_matrix
        self.dtype = dtype

    def __call__(self, datasets):
        ref_ds = self.hyperalignment.params.ref_ds
        nsamples, nfeatures = datasets[ref_ds].shape
        if 'roi_seed' in datasets[ref_ds].fa and np.any(datasets[ref_ds].fa['roi_seed']):
            seed_index = np.where(datasets[ref_ds].fa.roi_seed)
        else:
            if not self.full_matrix:
                raise ValueError(
                    "Setting full_matrix=False requires roi_seed `fa` in the "
                    "reference dataset indicating center feature and some "
                    "feature(s) being marked as `roi_seed`.")
            seed_index = None
        # Voxel selection within Searchlight
        # Usual metric of between-subject between-voxel correspondence
        # Making sure ref_ds has most features, if not force feature selection on others
        nfeatures_all = [sd.nfeatures for sd in datasets]
        bigger_ds_idxs = [i for i, sd in enumerate(datasets) if sd.nfeatures > nfeatures]
        if self.featsel != 1.0:
            # computing feature scores from the data
            feature_scores = compute_feature_scores(datasets, self.exclude_from_model)
            nfeatures_sel = nfeatures  # default
            if self.featsel < 1.0 and int(self.featsel * nfeatures) > 0:
                nfeatures_sel = int(self.featsel * nfeatures)
            if self.featsel > 1.0:
                nfeatures_sel = min(nfeatures, self.featsel)
            fselector = FixedNElementTailSelector(nfeatures_sel, tail='upper', mode='select', sort=False)
            # XXX Artificially make the seed_index feature score high to keep it(?)
            if self.use_same_features:
                if len(self.exclude_from_model):
                    feature_scores = [feature_scores[ifs] for ifs in range(len(datasets))
                                      if ifs not in self.exclude_from_model]
                feature_scores = np.mean(np.asarray(feature_scores), axis=0)
                if seed_index is not None:
                    feature_scores[seed_index] = max(feature_scores)
                features_selected = fselector(feature_scores)
                datasets = [sd[:, features_selected] for sd in datasets]
            else:
                features_selected = []
                for fs in feature_scores:
                    if seed_index is not None:
                        fs[seed_index] = max(fs)
                    features_selected.append(fselector(fs))
                datasets = [sd[:, fsel] for fsel, sd in zip(features_selected, datasets)]
        elif bigger_ds_idxs:
            # compute feature scores and select for bigger datasets
            feature_scores = compute_feature_scores(datasets, self.exclude_from_model)
            feature_scores = [feature_scores[isub] for isub in bigger_ds_idxs]
            fselector = FixedNElementTailSelector(nfeatures, tail='upper', mode='select', sort=False)
            features_selected = [fselector(fs) for fs in feature_scores]
            for selected_features, isub in zip(features_selected, bigger_ds_idxs):
                datasets[isub] = datasets[isub][:, selected_features]
        # Try hyperalignment
        try:
            # it is crucial to retrain hyperalignment, otherwise it would simply
            # project into the common space of a previous iteration
            if len(self.exclude_from_model) == 0:
                self.hyperalignment.train(datasets)
            else:
                self.hyperalignment.train([datasets[i] for i in range(len(datasets))
                                           if i not in self.exclude_from_model])
            mappers = self.hyperalignment(datasets)
            if mappers[0].proj.dtype is self.dtype:
                mappers = [m.proj for m in mappers]
            else:
                mappers = [m.proj.astype(self.dtype) for m in mappers]
            if self.featsel != 1.0:
                # Reshape the projection matrix from selected to all features
                mappers_full = [np.zeros((nfeatures_all[im], nfeatures_all[ref_ds]))
                                for im in range(len(mappers))]
                if self.use_same_features:
                    for mf, m in zip(mappers_full, mappers):
                        mf[np.ix_(features_selected, features_selected)] = m
                else:
                    for mf, m, fsel in zip(mappers_full, mappers, features_selected):
                        mf[np.ix_(fsel, features_selected[ref_ds])] = m
                mappers = mappers_full
            elif bigger_ds_idxs:
                for selected_features, isub in zip(features_selected, bigger_ds_idxs):
                    mapper_full = np.zeros((nfeatures_all[isub], nfeatures_all[ref_ds]))
                    mapper_full[np.ix_(selected_features, range(nfeatures))] = mappers[isub]
                    mappers[isub] = mapper_full
        except LinAlgError:
            print "SVD didn't converge. Try with a new reference, may be."
            mappers = [np.eye(nfeatures, dtype='int')] * len(datasets)
        # Extract only the row/column corresponding to the center voxel if full_matrix is False
        if not self.full_matrix:
            mappers = [np.squeeze(m[:, seed_index]) for m in mappers]
        return mappers


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

    queryengine = Parameter(
        None,
        doc="""A single (or a list of query engines, one per each dataset) to be
        used.  If not provided, volumetric searchlight, with spherical
        neighborhood as instructed by radius parameter will be used.""")

    radius = Parameter(
        3,
        constraints=EnsureInt() & EnsureRange(min=1),
        doc="""Radius of a searchlight sphere in number of voxels to be used if
         no `queryengine` argument was provided.""")

    nproc = Parameter(
        1,
        constraints=EnsureInt() & EnsureRange(min=1) | EnsureNone(),
        doc="""Number of cores to use.""")

    nblocks = Parameter(
        None,
        constraints=EnsureInt() & EnsureRange(min=1) | EnsureNone(),
        doc="""Number of blocks to divide to process. Higher number results in
            smaller memory consumption.""")

    sparse_radius = Parameter(
        None,
        constraints=(EnsureRange(min=1) & EnsureInt() | EnsureNone()),
        doc="""Radius supplied to scatter_neighborhoods in units of voxels.
            This is effectively the distance between the centers where
            hyperalignment is performed in searchlights.  ATM applicable only
            if no custom queryengine was provided.
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
            it is the center voxel. This will not be applicable for certain
            queryengines whose ids and neighborhoods are from different spaces,
            such as for SurfaceVerticesQueryEngine""")

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
            proportion of features in each searchlight of ref_ds using feature scores;
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
        # This option makes the roi_seed in each SL to be selected during feature selection
        self.force_roi_seed = True
        if self.params.nproc is not None and self.params.nproc > 1 \
                and not externals.exists('pprocess'):
            raise RuntimeError("The 'pprocess' module is required for "
                               "multiprocess searchlights. Please either "
                               "install python-pprocess, or reduce `nproc` "
                               "to 1 (got nproc=%i) or set to default None"
                               % self.params.nproc)
        if not externals.exists('scipy'):
            raise RuntimeError("The 'scipy' module is required for "
                               "searchlight hyperalignment.")
        if self.params.results_backend == 'native':
            raise NotImplementedError("'native' mode to handle results is still a "
                                      "work in progress.")
            #warning("results_backend is set to 'native'. This has been known"
            #        "to result in longer run time when working with big datasets.")
        if self.params.results_backend == 'hdf5' and \
                not externals.exists('h5py'):
            raise RuntimeError("The 'hdf5' module is required for "
                               "when results_backend is set to 'hdf5'")

    def _proc_block(self, block, datasets, featselhyper, queryengines, seed=None, iblock='main'):
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
            roi_feature_ids_all = [qe[node_id] for qe in queryengines]
            # handling queryengines that return AttrDatasets
            for isub in range(len(roi_feature_ids_all)):
                if is_datasetlike(roi_feature_ids_all[isub]):
                    # making sure queryengine returned proper shaped output
                    assert(roi_feature_ids_all[isub].nsamples == 1)
                    roi_feature_ids_all[isub] = roi_feature_ids_all[isub].samples[0, :].tolist()
            if len(roi_feature_ids_all) == 1:
                # just one was provided to be "broadcasted"
                roi_feature_ids_all *= len(datasets)
            # if qe returns zero-sized ROI for any subject, pass...
            if any(len(x)==0 for x in roi_feature_ids_all):
                continue
            # selecting neighborhood for all subject for hyperalignment
            ds_temp = [sd[:, ids] for sd, ids in zip(datasets, roi_feature_ids_all)]
            if self.force_roi_seed:
                roi_seed = np.array(roi_feature_ids_all[self.params.ref_ds]) == node_id
                ds_temp[self.params.ref_ds].fa['roi_seed'] = roi_seed
            if __debug__:
                msg = 'ROI (%i/%i), %i features' % (i + 1, len(block),
                                                    ds_temp[self.params.ref_ds].nfeatures)
                debug('SLC', bar(float(i + 1) / len(block), msg), cr=True)
            hmappers = featselhyper(ds_temp)
            assert(len(hmappers) == len(datasets))
            roi_feature_ids_ref_ds = roi_feature_ids_all[self.params.ref_ds]
            for isub, roi_feature_ids in enumerate(roi_feature_ids_all):
                if not self.params.combine_neighbormappers:
                    I = roi_feature_ids
                    #J = [roi_feature_ids[node_id]] * len(roi_feature_ids)
                    J = [node_id] * len(roi_feature_ids)
                    V = hmappers[isub].tolist()
                    if np.isscalar(V):
                        V = [V]
                else:
                    I, J, V = [], [], []
                    for f2, roi_feature_id_ref_ds in enumerate(roi_feature_ids_ref_ds):
                        I += roi_feature_ids
                        J += [roi_feature_id_ref_ds] * len(roi_feature_ids)
                        V += hmappers[isub][:, f2].tolist()
                proj = coo_matrix(
                    (V, (I, J)),
                    shape=(max(self.nfeatures, max(I) + 1), max(self.nfeatures, max(J) + 1)),
                    dtype=self.params.dtype)
                proj = proj.tocsc()
                # Cleaning up the current subject's projections to free up memory
                hmappers[isub] = [[] for _ in hmappers]
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

        # Perform some checks first before modifying internal state
        params = self.params
        ndatasets = len(datasets)

        if len(datasets) <= 1:
            raise ValueError("SearchlightHyperalignment needs > 1 dataset to "
                             "operate on. Got: %d" % self.ndatasets)

        if params.ref_ds in params.exclude_from_model:
            raise ValueError("Requested reference dataset %i is also "
                             "in the exclude list." % params.ref_ds)

        if params.ref_ds >= ndatasets:
            raise ValueError("Requested reference dataset %i is out of "
                             "bounds. We have only %i datasets provided"
                             % (params.ref_ds, self.ndatasets))

        # The rest of the checks are just warnings
        self.ndatasets = ndatasets

        _shpaldebug("SearchlightHyperalignment %s for %i datasets"
                    % (self, self.ndatasets))

        if params.ref_ds != params.hyperalignment.params.ref_ds:
            warning('Supplied ref_ds & hyperalignment instance ref_ds:%d differ.'
                    % params.hyperalignment.params.ref_ds)
            warning('Using default hyperalignment instance with ref_ds: %d' % params.ref_ds)
            params.hyperalignment = Hyperalignment(ref_ds=params.ref_ds)
        if len(params.exclude_from_model) > 0:
            warning("These datasets will not participate in building common "
                    "model: %s" % params.exclude_from_model)

        if __debug__:
            # verify that datasets were zscored prior the alignment since it is
            # assumed/required preprocessing step
            for ids, ds in enumerate(datasets):
                for f, fname, tval in ((np.mean, 'means', 0),
                                       (np.std, 'stds', 1)):
                    vals = f(ds, axis=0)
                    vals_comp = np.abs(vals - tval) > 1e-5
                    if np.any(vals_comp):
                        warning('%d %s are too different (max diff=%g) from %d in '
                                'dataset %d to come from a zscored dataset. '
                                'Please zscore datasets first for correct operation '
                                '(unless if was intentional)'
                                % (np.sum(vals_comp), fname,
                                   np.max(np.abs(vals)), tval, ids))

        # Setting up SearchlightHyperalignment
        # we need to know which original features where comprising the
        # individual SL ROIs
        _shpaldebug('Initializing FeatureSelectionHyperalignment.')
        hmeasure = FeatureSelectionHyperalignment(
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
        queryengines = self._get_trained_queryengines(
            datasets, params.queryengine, params.radius, params.ref_ds)
        # For surface nodes to voxels queryengines, roi_seed hardly makes sense
        if isinstance(queryengines[params.ref_ds], SurfaceVerticesQueryEngine):
            self.force_roi_seed = False
            if not self.params.combine_neighbormappers:
                raise NotImplementedError("Mapping from voxels to surface nodes is not "
                        "implmented yet. Try setting combine_neighbormappers to True.")
        self.nfeatures = datasets[params.ref_ds].nfeatures
        _shpaldebug("Performing Hyperalignment in searchlights")
        # Setting up centers for running SL Hyperalignment
        if params.sparse_radius is None:
            roi_ids = self._get_verified_ids(queryengines) \
                if params.mask_node_ids is None \
                else params.mask_node_ids
        else:
            if params.queryengine is not None:
                raise NotImplementedError(
                    "using sparse_radius whenever custom queryengine is "
                    "provided is not yet supported.")
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
                compute(block, datasets, copy.copy(hmeasure), queryengines,
                        seed=seed, iblock=iblock)
        else:
            # otherwise collect the results in an 1-item list
            _shpaldebug('Using 1 process to compute mappers.')
            if params.nblocks is None:
                params.nblocks = 1
            params.nblocks = min(len(roi_ids), params.nblocks)
            node_blocks = np.array_split(roi_ids, params.nblocks)
            p_results = [self._proc_block(block, datasets, hmeasure, queryengines)
                         for block in node_blocks]
        results_ds = self.__handle_all_results(p_results)
        # Dummy iterator for, you know, iteration
        list(results_ds)

        _shpaldebug('Wrapping projection matrices into StaticProjectionMappers')
        self.projections = [
            StaticProjectionMapper(proj=proj, recon=proj.T) if params.compute_recon
            else StaticProjectionMapper(proj=proj)
            for proj in self.projections]
        return self.projections

    def _get_verified_ids(self, queryengines):
        """Helper to return ids of queryengines, verifying that they are the same"""
        qe0 = queryengines[0]
        roi_ids = qe0.ids
        for qe in queryengines:
            if qe is not qe0:
                # if a different query engine (so wasn't just replicated)
                if np.any(qe.ids != qe0.ids):
                    raise RuntimeError(
                        "Query engine %s provided different ids than %s. Not supported"
                        % (qe0, qe))
        return roi_ids

    def _get_trained_queryengines(self, datasets, queryengine, radius, ref_ds):
        """Helper to return trained query engine(s), either list of one or one per each dataset

        if queryengine is None then IndexQueryEngine based on radius is created
        """
        ndatasets = len(datasets)
        if queryengine:
            if isinstance(queryengine, (list, tuple)):
                queryengines = queryengine
                if len(queryengines) != ndatasets:
                    raise ValueError(
                        "%d query engines were specified although %d datasets "
                        "provided" % (len(queryengines), ndatasets))
                _shpaldebug("Training provided query engines")
                for qe, ds in zip(queryengines, datasets):
                    qe.train(ds)
            else:
                queryengine.train(datasets[ref_ds])
                queryengines = [queryengine]
        else:
            _shpaldebug('No custom query engines were provided. Setting up the '
                        'volumetric query engine on voxel_indices.')
            queryengine = IndexQueryEngine(voxel_indices=Sphere(radius))
            queryengine.train(datasets[ref_ds])
            queryengines = [queryengine]
        return queryengines
