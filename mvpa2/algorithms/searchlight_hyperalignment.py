# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Searchlight-based hyperalignment"""
import numpy as np
from mvpa2.base.state import ClassWithCollections
from mvpa2.base.param import Parameter
from mvpa2.algorithms.hyperalignment import Hyperalignment
from mvpa2.measures.base import Measure
from mvpa2.datasets import Dataset, vstack
from mvpa2.mappers.staticprojection import StaticProjectionMapper
from scipy.linalg import LinAlgError
from mvpa2.measures.searchlight import sphere_searchlight
from mvpa2.mappers.zscore import zscore
from scipy.sparse import coo_matrix, dok_matrix, csc_matrix
from mvpa2.featsel.helpers import FractionTailSelector, FixedNElementTailSelector

if __debug__:
    from mvpa2.base import debug


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
        for j, sd2 in enumerate(datasets[i+1:]):
            corr_temp = np.dot(sd.samples.T, sd2.samples)
            if j+i+1 not in exclude_from_model:
                feature_scores[i] += np.max(corr_temp, axis=1)
            if i not in exclude_from_model:
                feature_scores[j + i + 1] += np.max(corr_temp, axis=0)
    return feature_scores


class HyperalignmentMeasure(Measure):
    """
    HyperalignmentMeasure combines feature selection and hyperalignment
    into a single node. This facilitates it's usage in any searchlight
    or ROI.
    """
    is_trained = True

    def __init__(self, ndatasets, hyperalignment, scale=0.0, ref_ds=0,
                 featsel=1.0, full_matrix=True, use_same_features=False,
                 exclude_from_model=None, dtype='float64', **kwargs):
        Measure.__init__(self, **kwargs)
        self.ndatasets = ndatasets
        self.hyperalignment = hyperalignment
        self.scale = scale
        self.ref_ds = ref_ds
        self.featsel = featsel
        self.use_same_features = use_same_features
        self.exclude_from_model = exclude_from_model
        if self.exclude_from_model is None:
            self.exclude_from_model = []
        self.full_matrix = full_matrix
        self.dtype = dtype

    def _call(self, dataset):
        ds = []
        nsamples = dataset.nsamples/self.ndatasets
        nfeatures = dataset.nfeatures
        seed_index = np.where(dataset.fa.roi_seed)
        if self.scale > 0.0:
            dist = np.sum(np.abs(dataset.fa.voxel_indices-dataset.fa.voxel_indices[seed_index]), axis=1)
            dist = np.exp(-(self.scale*dist/np.float(max(dist)) )**2)
            dataset.samples = dataset.samples*dist
        # Creating a list of datasets for hyperalignment
        for i in range(self.ndatasets):
            # XXX this should rather be a Splitter taking the dataset apart
            # based on a sample attribute
            ds.append(dataset[0+i*nsamples:nsamples*(i+1),])
        # Voxel selection within Searchlight
        # Usual metric of between-subject between-voxel correspondence
        if self.featsel != 1.0:
            # computing feature scores from the data
            feature_scores = compute_feature_scores(ds, self.exclude_from_model)
            if self.featsel < 1.0:
                fselector = FractionTailSelector(self.featsel, tail='upper', mode='select',sort=False)
            else:
                fselector = FixedNElementTailSelector(np.floor(self.featsel), tail='upper', mode='select',sort=False)
            # XXX Artificially make the seed_index feature score high to keep it(?)
            if self.use_same_features:
                if len(self.exclude_from_model):
                    feature_scores = [feature_scores[ifs] for ifs in range(len(self.ndatasets)) 
                                      if ifs not in self.exclude_from_model]
                feature_scores = np.mean(np.asarray(feature_scores),axis=0)
                feature_scores[seed_index] = max(feature_scores)
                features_selected = fselector(feature_scores)
                ds = [sd[:, features_selected] for sd in ds]
            else:
                features_selected = []
                for fs in feature_scores:
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
                mappers_full = [np.zeros((nfeatures,nfeatures)) for im in range(len(mappers))]
                if self.use_same_features:
                    for mf, m in zip(mappers_full, mappers):
                        mf[np.ix_(features_selected, features_selected)] = m
                else:
                    for mf,m,fsel in zip(mappers_full, mappers, features_selected):
                        mf[np.ix_(fsel, features_selected[self.ref_ds])] = m
                mappers = mappers_full
        except LinAlgError:
            print "SVD didn't converge. Try with a new reference, may be."
            mappers = [np.eye(nfeatures, dtype='int')]*self.ndatasets
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

    """
    ref_ds = Parameter(0, allowedtype='int', min=0,
             doc="""Index of a dataset to use as a reference. First dataset
             is used as default. 
             If you supply exclude_from_model list, you should
             supply the ref_ds index as index after you remove those excluded datasets """)

    sl = Parameter(sphere_searchlight(None, radius=3, add_center_fa=True, nblocks=19), allowedtype='Searchlight',
            doc="""Searchlight instance with datameasure as None & add_center_fa=True.
            """)

    sparse_radius = Parameter(None, allowedtype='int',
            doc="""Radius supplied to scatter_neighborhoods. This is effectively the distance
            between the centers where hyperalignment is performed in searchlights. If None, hyperalignment
            is performed at every voxel.""")

    hyperalignment = Parameter(Hyperalignment(ref_ds=0),
            doc="""Hyperalignment instance to be used in each searchlight sphere.
            Default is just the Hyperalignment instance with default parameters.""")

    combine_neighbormappers = Parameter(True, allowedtype='bool',
            doc="""This param determines whether to combine mappers for each voxel
            from its neighborhood searchlights or just use the mapper for which it is the
            center voxel.
            Use this option with caution, as enabling it might square the runtime memory
            requirement. If you run into memory issues, reduce the nproc in sl. """)

    zscore_all = Parameter(False, allowedtype='bool',
            doc="""Z-score all datasets prior hyperalignment.  Turn it off
            if zscoring is not desired or was already performed. """)

    compute_recon = Parameter(True, allowedtype='bool',
            doc="""This param determines whether to compute reverse mappers for each 
            subject from common-space to subject space. These will be stored in the
            StaticProjectionMapper() and used when reverse() is called.
            Enabling it will double the size of the mappers returned.""")
    
    featsel = Parameter(1.0, allowedtype='float',
            doc="""Determines if feature selection will be performed in each searchlight.
            1.0: Use all features. <1.0 is understood as selecting that proportion of
            of feature in each searchlight using feature scores (Refer to the code);
            >1.0 is understood as selecting at most that many features in each searchlight.""")

    use_same_features = Parameter(False, allowedtype='bool',
            doc="""Select same features when doing feature selection for all datasets.
            If this is true, feature scores across datasets will be averaged to select best features""")

    exclude_from_model = Parameter([], allowedtype='list',
            doc="""List of dataset indices that will not participate in building common model.
            These will still get mappers back but they don't influence the model or voxel selection.""")

    sparse_mode = Parameter('coo', allowedtype='str',
                            doc="""Via what type of sparse matrix to construct
                            the full projection matrix. Possible values are
                            'coo' (COOrdinate format; this is faster but
                            potentially more memory-hungry) or 'dok' (Dictionary
                            Of Keys; .lower but incremental and less
                            memory-hungry.""")

    dtype = Parameter('float32', allowedtype='str',
            doc="""dtype of elements transformation matrices to save on memory for big datasets""")
            
    def __init__(self, **kwargs):
        ClassWithCollections.__init__(self, **kwargs)
        self.params.sl.results_fx = self.results_fx
        self.params.sl.results_postproc_fx = self.results_postproc_fx
        self.ndatasets = 0
        self.nfeatures = 0

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
        ndatasets = self.ndatasets
        params = self.params
        if __debug__:
            debug('SHPAL', "SearchlightHyperalignment %s for %i datasets"
                % (self, ndatasets))
        if params.ref_ds in params.exclude_from_model:
            raise ValueError, "Requested reference dataset %i is also " \
                  "in the exclude list."% (params.ref_ds)            
        if params.ref_ds != params.hyperalignment.params.ref_ds:
            print('SHPAL: Supplied ref_ds & hyperalignment instance ref_ds:%d differ.'%(params.hyperalignment.params.ref_ds))
            print('SHPAL: using default hyperalignment instance with ref_ds: %d'%(params.ref_ds))
            params.hyperalignment = Hyperalignment(ref_ds=params.ref_ds)
        if params.ref_ds < 0 and params.ref_ds >= ndatasets:
            raise ValueError, "Requested reference dataset %i is out of " \
                  "bounds. We have only %i datasets provided" \
                  % (params.ref_ds, ndatasets)
        if len(params.exclude_from_model) > 0:
            print('SHPAL: These datasets will not participate in building common model:'),
            print params.exclude_from_model
        # Setting up SearchlightHyperalignment
        slhyper = params.sl
        # we need to know which original features where comprising the
        # individual SL ROIs
        slhyper.ca.enable(['roi_feature_ids'])
        hmeasure = HyperalignmentMeasure(ndatasets=self.ndatasets, featsel=params.featsel, ref_ds=params.ref_ds,
                      hyperalignment=params.hyperalignment, full_matrix=params.combine_neighbormappers,
                      use_same_features=params.use_same_features, exclude_from_model=params.exclude_from_model, dtype=params.dtype)
        slhyper.datameasure = hmeasure
        if params.sparse_radius is not None:
            from mvpa2.misc.neighborhood import Sphere, scatter_neighborhoods
            if slhyper.roi_ids is None:
                scoords, sidx = scatter_neighborhoods(Sphere(params.sparse_radius), 
                                                      datasets[params.ref_ds].fa.voxel_indices)
                slhyper._BaseSearchlight__roi_ids = sidx
            else:
                scoords, sidx = scatter_neighborhoods(Sphere(params.sparse_radius), 
                                                      datasets[params.ref_ds].fa.voxel_indices[slhyper.roi_ids])
                slhyper._BaseSearchlight__roi_ids = [slhyper.roi_ids[sid] for sid in sidx]
        # Zscoring time-series data if desired.
        if params.zscore_all:
            if __debug__:
                debug('SHPAL', "Zscoring datasets")
            for ds in datasets:
                zscore(ds, chunks_attr=None)

        # XXX I think this class should already accept a single dataset only.
        # It should have a ``space`` setting that names a sample attribute that
        # can be used to identify individual/original datasets.
        # Taking a single dataset as argument would be cleaner, because the
        # algorithm relies on the assumption that there is a coarse feature
        # alignment, i.e. the SL ROIs cover roughly the same area
        datasets = vstack(datasets)
        self.nfeatures = datasets.nfeatures
        #nfeatures = self.nfeatures                
        if __debug__:
            debug('SHPAL', "Performing Hyperalignment")
        hmappers = slhyper(datasets).samples.tolist()
        datasets = None
        projections = [hm[0] for hm in hmappers]
        '''
        hmappers = hmappers.samples
        if __debug__:
            debug('SHPAL', "Computing projection mappers for")
        projections = []
        # iterate over mappers corresponding to (the number of) input datasets
        for isub in range(len(hmappers)):
            if __debug__:
                debug('SHPAL', "dataset %d"%(isub+1))
            # create a giant sparse projection matrix, covering all features in
            # the input dataset. To do this we need to iterate over all SL
            # projections and place them into the matrix elements corresponding
            # to the respective SL ROI
            if params.sparse_mode == 'coo':
                proj = self._proj_via_coo(hmappers[isub,], slhyper.ca.roi_feature_ids,
                                                nfeatures)
            else:
                proj = self._proj_via_dok(hmappers[isub,], slhyper.ca.roi_feature_ids,
                                                nfeatures)
            # Cleaning up the current subject's projections to free up memory
            hmappers[isub,] = [[]]*hmappers.shape[1]
            # Scaling the features weights by number of times it contributes
            # This step takes away the advantage of a neighboring voxels which
            # they gain by appearing in multiple searchlights
            nsls = proj[-1]
            if params.compute_recon:
                proj_recon=proj[1]
            proj = proj[0]
            if params.scale_down_neighbors & params.combine_neighbormappers:
                nsls = 1/nsls
                proj = proj*spdiags([nsls],[0], proj.shape[0], proj.shape[0])
            #projections.append(StaticProjectionMapper(proj=proj/nsls))
            # Can be done using QR decomposition as well
            # XXX We should switch to using sparsesvd wrapper to accomplish this.
            # Also, why? really! why?
            if params.orthogonalize_mappers:
                Uh, S, Vh = sparsesvd(proj, proj.shape[0])
                proj = np.dot(Uh.T, Vh)
            #projections.append(StaticProjectionMapper(proj=csc_matrix(proj)))
            if params.compute_recon:
                projections.append(StaticProjectionMapper(proj=proj, recon=proj_recon))
            else:                
                projections.append(StaticProjectionMapper(proj=proj))
        '''
        return projections

    def results_postproc_fx(self, results=None):
        # XXX Divide & Rule
        # chunk features into subsets and do them in series
        # initialize mappers
        ndatasets = self.ndatasets
        nfeatures = self.nfeatures
        params = self.params
        projections = [csc_matrix((nfeatures, nfeatures), dtype=params.dtype) for i in range(ndatasets)]
        if params.compute_recon:
            projections_recon = [csc_matrix((nfeatures, nfeatures), dtype=params.dtype) for i in range(ndatasets)]
        # Handle each results block
        for res in results:
            roi_feature_ids_list = [rs_.a.roi_feature_ids for rs_ in res]
            #res = hstack(rs)
            for isub in range(ndatasets):
                if params.sparse_mode == 'coo':
                    proj = self._proj_via_coo(res.samples[isub,],
                                                    [res.a.roi_feature_ids], nfeatures)
                else:
                    proj = self._proj_via_dok(res.samples[isub,],
                                                    roi_feature_ids_list, nfeatures)
                # Cleaning up the current subject's projections to free up memory
                res[isub,].samples = [[]]*res.shape[1]
                if params.compute_recon:
                    proj_recon=proj[1]
                    projections_recon[isub] = projections_recon[isub] + proj_recon
                proj = proj[0]
                projections[isub] = projections[isub] + proj
        return projections

    def results_fx(self, sl=None, dataset=None, roi_ids=None, results=None):
        # XXX Divide & Rule
        # chunk features into subsets and do them in series
        # initialize mappers
        ndatasets = self.ndatasets
        nfeatures = self.nfeatures
        params = self.params
        projections = [csc_matrix((nfeatures, nfeatures), dtype=params.dtype) for i in range(ndatasets)]             
        for rs in results:
            for isub, irs in enumerate(rs):
                projections[isub] = projections[isub] + irs
        # Wrap StaticProjectionMapper around the matrices
        if params.compute_recon:
            projections = [StaticProjectionMapper(proj=proj, recon=proj.T) for proj in projections]
        else:                
            projections = [StaticProjectionMapper(proj=proj) for proj in projections]
        return projections
    
    def _proj_via_coo(self, sl_results, roi_feature_id_list, nfeatures):
        """Construct the combined projection matrix via a COO sparse matrix"""
        params = self.params
        #nseeds = len(sl_results)
        I = [] # matrix element row
        J = [] # matrix element column
        V = [] # matrix element value
        # track from how many ROIs each voxel receives "contributions"
        # regarding its projection
        nsls = np.zeros(nfeatures, dtype='int')
        # iterate over SL ROIs for the current mapper
        for ivox,f in enumerate(sl_results):
            fproj = f['proj']
            roi_feature_ids = roi_feature_id_list[ivox]
            if not params.combine_neighbormappers:
                I += roi_feature_ids
                # in this case the projection is always a vector only
                assert(len(fproj.shape) == 1)
                J += [ivox] * len(roi_feature_ids)
                V += fproj.tolist()
                # only the center voxel is affected by the projection
                # computed from this ROI
                nsls[ivox] += 1
            else:
                # in this case we have a full projection matrix
                assert(len(fproj.shape) == 2)
                for f2 in xrange(len(roi_feature_ids)):
                    I += roi_feature_ids
                    J += [roi_feature_ids[f2]] * len(roi_feature_ids)
                    V += fproj[:,f2].tolist()
                # all voxels in the ROI are affected by the projection
                nsls[roi_feature_ids] += 1
        proj = coo_matrix((V,(I,J)), shape=(max(nfeatures, max(I)+1),
                                            max(nfeatures,max(J)+1)), dtype=params.dtype)
        if params.compute_recon:
            proj_recon = coo_matrix((V,(J,I)), shape=(max(nfeatures, max(J)+1),
                                            max(nfeatures,max(I)+1)), dtype=params.dtype)
            return proj.tocsc(), proj_recon.tocsc(), nsls
        else:
            return proj.tocsc(), nsls


    def _proj_via_dok(self, sl_results, roi_feature_id_list, nfeatures):
        """Construct the combined projection matrix via a DOK sparse matrix"""
        params = self.params
        #nseeds = len(sl_results)
        # create the sparse matrix to match the dtype of the results
        proj = dok_matrix((nfeatures, nfeatures),
                          dtype=sl_results[0]['proj'].dtype)
        if params.compute_recon:
            proj_recon = dok_matrix((nfeatures, nfeatures),
                            dtype=sl_results[0]['proj'].dtype)
        # track from how many ROIs each voxel receives "contributions"
        # regarding its projection
        nsls = np.zeros(nfeatures, dtype='int')
        # iterate over SL ROIs for the current mapper
        for ivox,f in enumerate(sl_results):
            fproj = f['proj']
            roi_feature_ids = roi_feature_id_list[ivox]
            if not params.combine_neighbormappers:
                # in this case the projection is always a vector only
                assert(len(fproj.shape) == 1)
                for i, fp in enumerate(fproj):
                    # += to be consistent with what COO does for duplicates
                    proj[roi_feature_ids[i], ivox] += fp
                    if params.compute_recon:
                        proj_recon[ivox, roi_feature_ids[i]] += fp
                # only the center voxel is affected by the projection
                # computed from this ROI
                nsls[ivox] += 1
            else:
                # in this case we have a full projection matrix
                assert(len(fproj.shape) == 2)
                for f2 in xrange(len(roi_feature_ids)):
                    for i, fp in enumerate(fproj[:, f2]):
                        # Needs to be += as there could be multiple assignments
                        proj[roi_feature_ids[i], roi_feature_ids[f2]] += fp
                        if params.compute_recon:
                            proj_recon[roi_feature_ids[f2], roi_feature_ids[i]] += fp
                # all voxels in the ROI are affected by the projection
                nsls[roi_feature_ids] += 1
        if params.compute_recon:
            return proj.tocsc(), proj_recon.tocsc(), nsls
        else:
            return proj.tocsc(), nsls
