# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8 -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Connectivity-based hyperalignment on surface with PCs

(Because everything is better on surface and PCA is the silver bullet!)
See :ref:`Guntupalli et al., Plos Comp. Bio (2018)` for details.

Created on Feb 03, 2016

@author: Swaroop Guntupalli
"""

__docformat__ = 'restructuredtext'

import numpy as np
import os

from mvpa2.base import externals
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import *
from mvpa2.base.dataset import hstack
from mvpa2.base.types import is_datasetlike

from mvpa2.datasets import Dataset

from mvpa2.mappers.zscore import zscore
from mvpa2.mappers.fxy import FxyMapper
from mvpa2.mappers.svd import SVDMapper

from mvpa2.measures.searchlight import Searchlight
from mvpa2.measures.base import Measure

from mvpa2.algorithms.searchlight_hyperalignment import SearchlightHyperalignment
# experimental sparse matrix usage for faster computing with possible extra mem load
# from searchlight_hyperalignment import SearchlightHyperalignment

from mvpa2.algorithms.hyperalignment import Hyperalignment

from mvpa2.support.due import due, Doi

if externals.exists('h5py'):
    from mvpa2.base.hdf5 import h5save, h5load

if __debug__:
    from mvpa2.base import debug
    if 'CHPAL' in debug.active:
        def _chpaldebug(msg):
            debug('CHPAL', "%s" % msg)
    else:
        def _chpaldebug(*args):
            return None
else:
    def _chpaldebug(*args):
        return None

class MeanFeatureMeasure(Measure):
    """Mean group feature measure

    Because the vanilla one doesn't want to work for me (Swaroop).
    TODO: figure out why "didn't work" exactly, and adjust description
    and possibly name above
    """

    is_trained = True

    def __init__(self, **kwargs):
        Measure.__init__(self, **kwargs)

    def _call(self, dataset):
        return Dataset(samples=np.mean(dataset.samples, axis=1))


class ConnectivityHyperalignment(SearchlightHyperalignment):
    """
    Given a list of datasets, provide a list of mappers
    into common space using connectivity based hyperalignment.
    This time on Surface!!!

    - Compute the mean time-series for each connectivity target.
    - Use these mean time-series to align each target region and get `npc`
      PC time-series per region that are aligned across individuals (optional).
    - Compute a connectivity profile for each feature (e.g., vertex) depicting its
      connectivities to the targets. If `npc` is None, the mean time-series of
      each target is used; otherwise, the `npc` PC time-series are used.
    - Use SL HA to align the whole cortex based on connectivity profiles.

    See :ref:`Guntupalli et al., Plos Comp. Bio (2018)` for details.
    """
    mask_ids = Parameter(None, constraints=EnsureListOf(int) | EnsureNone(),
            doc="""You can specify a mask to compute searchlight hyperalignment
            only within this mask..""")

    seed_indices = Parameter(None, constraints=EnsureListOf(int) | EnsureNone(),
            doc="""A list of node indices that correspond to seed centers for
            seed queryengines. If None, all centers of seed_queryengines
            are used.""")

    seed_queryengines = Parameter(None,
            doc="""A list of queryengines to determine seed searchlights for
            connectomes. If a single queryengine is given in the list, then it
            is assumed that it applies to all datasets.""")

    seed_radius = Parameter(None, constraints=EnsureInt() & EnsureRange(min=1) |
                            EnsureNone(),
            doc=""" Radius in voxels for seed size in volume.""")

    conn_metric = Parameter(lambda x, y: np.dot(x.samples.T, y.samples)/x.nsamples, #
            doc="""How to compute the connectivity metric between features.
            Default is the dot product of samples (which on zscored data becomes
            correlation if you normalize by nsamples.""")

    npcs = Parameter(3, constraints=EnsureInt() & EnsureRange(min=1) | EnsureNone(),
            doc="""Maximum number of PCs to be considered in each surface searchlight.
            If None, use seed mean instead of PCs.
            """)

    connectomes = Parameter(None, constraints=EnsureStr() | EnsureNone(),
            doc="""Precomputed connectomes supplied as hdf5 filename (for now).
            It is expected to be a dictionary with key 'hmappers' (for now).""")

    common_model = Parameter(None, constraints=EnsureStr() | EnsureNone(),
            doc="""Precomputed common model supplied as hdf5 filename (for now).
            It is expected to be a dict with feature-targets connectome and
            common models in each target ROI with appropriate pcs (for now).
            Expects 'local_models' and 'connectome_model' keys.""")

    save_model = Parameter(None, constraints=EnsureStr() | EnsureNone(),
            doc="""Precomputed common model supplied as hdf5 filename (for now).
            It is expected to be a tuple with feature-targets connectome and
            common models in each target ROI with appropriate pcs (for now).""")

    def __init__(self, **kwargs):
        SearchlightHyperalignment.__init__(self, **kwargs)

    def _get_seed_means(self, measure, queryengine, dataset, seed_indices):
        # Computing seed data as mean timeseries in each SL
        seed_data = Searchlight(measure, queryengine=queryengine,
                                nproc=self.params.nproc, roi_ids=seed_indices)
        seed_data = seed_data(dataset)
        zscore(seed_data, chunks_attr=None)
        return seed_data
    
    def _get_sl_connectomes(self, seed_means, qe_all, datasets, inode, connectivity_mapper):
        # For each SL, computing connectivity of features to seed means
        sl_connectomes = []
        # Looping over each subject
        for seed_mean, qe_, sd in zip(seed_means, qe_all, datasets):
            connectivity_mapper.train(seed_mean)
            sl_ids = qe_[inode]
            if is_datasetlike(sl_ids):
                assert (sl_ids.nsamples == 1)
                sl_ids = sl_ids.samples[0, :].tolist()
            sl_connectomes.append(connectivity_mapper.forward(sd[:, sl_ids]))
        return sl_connectomes
    
    def _get_hypesvs(self, sl_connectomes, local_common_model=None):
        '''
        Hyperalign connectomes and return mapppers
        and trained SVDMapper of common space.

        Parameters
        ----------
        sl_connectomes: a list of connectomes to hyperalign
        local_common_model: a reference common model to be used.

        Returns
        -------
        a tuple (sl_hmappers, svm, local_common_model)
        sl_hmappers: a list of mappers corresponding to input list in that order.
        svm: a svm mapper based on the input data. if given a common model, this is None.
        local_common_model: If local_common_model is provided as input, this will be None.
            Otherwise, local_common_model will be computed here and returned.
        '''
        # TODO Should we z-score sl_connectomes?
        return_model = False if self.params.save_model is None else True
        if local_common_model is not None:
            ha = Hyperalignment(level2_niter=0)
            if not is_datasetlike(local_common_model):
                local_common_model = Dataset(samples=local_common_model)
            ha.train([local_common_model])
            sl_hmappers = ha(sl_connectomes)
            return sl_hmappers, None, None
        ha = Hyperalignment()
        sl_hmappers = ha(sl_connectomes)
        sl_connectomes = [slhm.forward(slc) for slhm, slc in zip(sl_hmappers, sl_connectomes)]
        _ = [zscore(slc, chunks_attr=None) for slc in sl_connectomes]
        sl_connectomes = np.dstack(sl_connectomes).mean(axis=-1)
        svm = SVDMapper(force_train=True)
        svm.train(sl_connectomes)
        if return_model:
            local_common_model = svm.forward(sl_connectomes)
        else:
            local_common_model = None
        return sl_hmappers, svm, local_common_model

    def _get_connectomes(self, datasets):
        params = self.params
        # If no precomputed connectomes are supplied, compute them.
        if params.connectomes is not None and os.path.exists(params.connectomes):
            _chpaldebug("Loading pre-computed connectomes from %s" % params.connectomes)
            connectomes = h5load(params.connectomes)
            return connectomes
        connectivity_mapper = FxyMapper(params.conn_metric)
        # Initializing datasets with original anatomically aligned datasets
        mfm = MeanFeatureMeasure()
        # TODO Handle seed_radius if seed queryengines are not provided
        seed_radius = params.seed_radius
        _chpaldebug("Performing surface connectivity hyperalignment with seeds")
        _chpaldebug("Computing connectomes.")
        ndatasets = len(datasets)
        if params.seed_queryengines is None:
            raise NotImplementedError("For now, we need seed queryengines.")
        qe_all = super(ConnectivityHyperalignment, self)._get_trained_queryengines(
            datasets, params.seed_queryengines, seed_radius, params.ref_ds)
        # If seed_indices are not supplied, use all as centers
        if not params.seed_indices:
            roi_ids = super(ConnectivityHyperalignment, self)._get_verified_ids(qe_all)
        else:
            roi_ids = params.seed_indices
        if len(qe_all) == 1:
            qe_all *= ndatasets
        # Computing Seed means to be used for aligning seed features
        seed_means = [self._get_seed_means(MeanFeatureMeasure(), qe, ds, params.seed_indices)
                      for qe, ds in zip(qe_all, datasets)]
        if params.npcs is None:
            conn_targets = []
            for seed_mean in seed_means:
                zscore(seed_mean, chunks_attr=None)
                conn_targets.append(seed_mean)
        else:
            # compute all PC-seed connectivity in each subject
            # 1. make common model SVs in each seed SL based on connectivity to seed_means
            # 2. Use these SVs for computing connectomes
            _chpaldebug("Aligning SVs in each searchlight across subjects")
            # Looping over all seeds in which SVD is done
            pc_data = [[] for isub in range(ndatasets)]
            sl_common_models = dict()
            if params.common_model is not None and os.path.exists(params.common_model):
                _chpaldebug("Loading common model from %s" % params.common_model)
                common_model = h5load(params.common_model)
                sl_common_models = common_model['local_models']
            for inode in roi_ids:
                # For each SL, computing connectivity of features to seed means
                # This line below doesn't need common model
                sl_connectomes = self._get_sl_connectomes(seed_means, qe_all, datasets,
                                                          inode, connectivity_mapper)
                # Hyperalign connectomes in SL
                # XXX TODO Common model input to below function should be updated.
                local_common_model = sl_common_models[inode][:, :params.npcs] \
                                        if params.common_model else None
                sl_hmappers, svm, sl_common_model = self._get_hypesvs(sl_connectomes,
                                                local_common_model=local_common_model)
                if sl_common_model is not None:
                    sl_common_models[inode] = sl_common_model
                # make common model SV timeseries data in each subject
                for sd, slhm, qe, pcd in zip(datasets, sl_hmappers, qe_all, pc_data):
                    sd_svs = slhm.forward(sd[:, qe[inode]])
                    zscore(sd_svs, chunks_attr=None)
                    if svm is not None:
                        sd_svs = svm.forward(sd_svs)
                        sd_svs = sd_svs[:, :params.npcs]
                        zscore(sd_svs, chunks_attr=None)
                    pcd.append(sd_svs)
            if params.save_model is not None:
                # TODO: should use debug
                print('Saving local models to %s' % params.save_model)
                h5save(params.save_model, sl_common_models)
            pc_data = [hstack(pcd) for pcd in pc_data]
            conn_targets = pc_data
            #print pc_data[-1]
        # compute connectomes using connectivity targets (PCs or seed means)
        connectomes = []
        if params.common_model is not None and os.path.exists(params.common_model):
            # TODO: should use debug
            print('Loading from saved common model: %s' % params.common_model)
            connectome_model = common_model['connectome_model']
            connectomes.append(connectome_model)
        for t_, ds in zip(conn_targets, datasets):
            connectivity_mapper.train(t_)
            connectome = connectivity_mapper.forward(ds)
            t_ = None
            connectome.fa = ds.fa
            if connectome.samples.dtype == 'float64':
                connectome.samples = connectome.samples.astype('float32')
            zscore(connectome, chunks_attr=None)
            connectomes.append(connectome)
        if params.connectomes is not None and not os.path.exists(params.connectomes):
            _chpaldebug("Saving connectomes to ", params.connectomes)
            h5save(params.connectomes, connectomes)
        return connectomes

    @due.dcite(
        Doi('10.1371/journal.pcbi.1006120'),
        description="Connectivity-based hyperalignment",
        tags=["implementation"])
    def __call__(self, datasets):
        """Estimate mappers for each dataset

        Parameters
        ----------
          datasets : list or tuple of datasets

        Returns
        -------
        A list of trained Mappers of the same length as datasets
        """
        connectomes = self._get_connectomes(datasets)
        # TODO Add assertion about nsamples matching across connectomes
        _chpaldebug("Performing hyperalignment of %d connectomes with %d samples" %
                    (len(connectomes), connectomes[0].nsamples))
        _chpaldebug("Running searchlight hyperalignment")
        conhypmappers = super(ConnectivityHyperalignment, self).__call__(connectomes)
        _chpaldebug("Finished Connectivity hyperalignment. Returning mappers.")
        return conhypmappers
