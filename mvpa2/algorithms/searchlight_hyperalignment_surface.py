# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Searchlight-based hyperalignment"""
import os, tempfile
import numpy as np
from mvpa2.base.state import ClassWithCollections
from mvpa2.base.param import Parameter
from mvpa2.algorithms.hyperalignment import Hyperalignment
from mvpa2.measures.base import Measure
from mvpa2.datasets import Dataset
from mvpa2.mappers.staticprojection import StaticProjectionMapper
from scipy.linalg import LinAlgError
from mvpa2.mappers.zscore import zscore
from scipy.sparse import coo_matrix, csc_matrix
#from mvpa2.featsel.helpers import FractionTailSelector
from mvpa2.featsel.helpers import FixedNElementTailSelector
import mvpa2.misc.surfing.volgeom as volgeom
from mvpa2.support.nibabel import surf
import mvpa2.misc.surfing.volume_mask_dict as volume_mask_dict
import mvpa2.misc.surfing.surf_voxel_selection as surf_voxel_selection
from mvpa2.misc.surfing.queryengine import SurfaceVerticesQueryEngine
from mvpa2.algorithms.searchlight_hyperalignment import compute_feature_scores
from mvpa2.base.progress import ProgressBar
from mvpa2.support import copy
from mvpa2.base.hdf5 import h5save, h5load
import mvpa2
if __debug__:
    from mvpa2.base import debug

class HyperalignmentMeasure(Measure):
    is_trained=True
    def __init__(self, ndatasets, hyperalignment, ref_ds=0,
            featsel=1.0 ,full_matrix=True, use_same_features=False, 
            exclude_from_model=[], dtype='float64', **kwargs):
        Measure.__init__(self, **kwargs)
        self.ndatasets = ndatasets
        self.hyperalignment = hyperalignment
        self.ref_ds = ref_ds
        self.featsel = featsel
        self.use_same_features = use_same_features
        self.exclude_from_model = exclude_from_model        
        self.full_matrix = full_matrix
        self.dtype = dtype

    def _call(self, dataset):
        ds = dataset
        #nsamples = dataset[self.ref_ds].nsamples
        nfeatures = dataset[self.ref_ds].nfeatures
        seed_index = np.where(dataset[self.ref_ds].fa.roi_seed)
        nfeatures_all = [sd.nfeatures for sd in ds]
        nfeatures_bin = [sd.nfeatures>nfeatures for sd in ds]
        # Voxel selection within Searchlight
        # Usual metric of between-subject between-voxel correspondence
        if self.featsel != 1.0:
            # computing feature scores from the data
            feature_scores = compute_feature_scores(ds, self.exclude_from_model)
            if self.featsel <1.0:
                fs_nfeatures = np.floor(self.featsel*nfeatures) if np.floor(self.featsel*nfeatures) else nfeatures
                #fselector = FractionTailSelector(self.featsel, tail='upper', mode='select',sort=False)
            else:
                if self.featsel > nfeatures: 
                    fs_nfeatures = nfeatures
                else:
                    fs_nfeatures = self.featsel
            fselector = FixedNElementTailSelector(np.floor(fs_nfeatures), tail='upper', mode='select',sort=False)
            # XXX Artificially make the seed_index feature score high to keep it(?)
            # use_same_features is NA
            if self.use_same_features:
                print "Can't use same features! Think again."
            else:
                feature_scores[self.ref_ds][seed_index] = max(feature_scores[self.ref_ds])
                features_selected = [fselector(fs) for fs in feature_scores]
                ds = [ds[isub][:, features_selected[isub]] for isub in range(self.ndatasets)]
        elif np.sum(nfeatures_bin)>0:
            feature_scores = compute_feature_scores(ds, self.exclude_from_model)
            feature_scores = [feature_scores[isub] for isub in np.where(nfeatures_bin)[0]]            
            fselector = FixedNElementTailSelector(nfeatures, tail='upper', mode='select',sort=False)
            features_selected = [fselector(fs) for fs in feature_scores]
            for isub_large,isub in enumerate(np.where(nfeatures_bin)[0]):
                ds[isub] = ds[isub][:, features_selected[isub_large]]
        # Try hyperalignment
        try:
            # it is crucial to retrain hyperalignment, otherwise it would simply
            # project into the common space of a previous iteration
            if len(self.exclude_from_model)==0:
                self.hyperalignment.train(ds)
            else:
                self.hyperalignment.train([ds[i] for i in range(len(ds)) 
                                            if i not in self.exclude_from_model])
            mappers = self.hyperalignment(ds)
            if mappers[0].proj.dtype is self.dtype:
                mappers = [ m.proj for m in mappers]
            else:
                mappers = [ m.proj.astype(self.dtype) for m in mappers]
            if self.featsel != 1.0:
                # Reshape the projection matrix from selected to all features
                mappers_full = [np.zeros((nfeatures_all[im],nfeatures_all[self.ref_ds])) for im in range(len(mappers))]
                if self.use_same_features:
                    print "Not applicable"
                else:
                    for mf,m,fsel in zip(mappers_full, mappers, features_selected):
                        mf[np.ix_(fsel, features_selected[self.ref_ds])]=m
                mappers = mappers_full
            elif np.sum(nfeatures_all)>0:
                for isub_large,isub in enumerate(np.where(nfeatures_bin)[0]):
                    mapper_full = np.zeros((nfeatures_all[isub],nfeatures_all[self.ref_ds]))
                    mapper_full[np.ix_(features_selected[isub_large], range(nfeatures))]=mappers[isub]
                    mappers[isub] = mapper_full
        except LinAlgError:
            print "SVD didn't converge. Try with a new reference, may be."
            mappers = [np.eye(nfeatures_all[isub],nfeatures_all[self.ref_ds],dtype='int') for isub in range(self.ndatasets)]
        # Extract only the row/column corresponding to the center voxel if full_matrix is False
        if not self.full_matrix:
            mappers = [ np.squeeze(m[:,seed_index]) for m in mappers]
        # Package results
        results = np.asanyarray([{'proj': mapper} for mapper in mappers])
        # Add residual errors to the seed voxel to be used later to weed out bad SLs(?)
        if 'residual_errors' in self.hyperalignment.ca.enabled:
            [ result.update({'residual_error': self.hyperalignment.ca['residual_errors'][ires]})
                    for ires, result in enumerate(results)]
        return Dataset(samples=results)


def compute_query_engine(voxsel_fname, dataset=None,
                         pial_surf_fn=None, white_surf_fn=None,
                         intermediate_surf_fn=None, radius=10.,
                         start_fr=0., stop_fr=1., 
                         start_mm=0., stop_mm=0.,
                         nproc=None,
                         node_voxel_mapping='minimal',
                         outside_node_margin=True, check=False):

    if __debug__:
        debug('SHPAL', "Getting voxel selection: %s" % voxsel_fname)

    if voxsel_fname is not None and os.path.isfile(voxsel_fname):
        # it's a filename, load it
        #voxsel = volume_mask_dict.from_any(voxsel_fname)
        voxsel = h5load(voxsel_fname)
    elif isinstance(voxsel_fname, volume_mask_dict.VolumeMaskDictionary):
        pass
    else:
        voxsel = surf_voxel_selection.run_voxel_selection(radius=radius,
                        volume=dataset[0,:],
                        white_surf=white_surf_fn, pial_surf=pial_surf_fn,
                        source_surf=intermediate_surf_fn,
                        distance_metric='dijkstra',
                        start_fr=start_fr, stop_fr=stop_fr, 
                        start_mm=start_mm, stop_mm=stop_mm,
                        nproc=nproc,
                        outside_node_margin=outside_node_margin,
                        node_voxel_mapping=node_voxel_mapping)
        if not voxsel_fname is None:
            h5save(voxsel_fname, voxsel, compression='gzip')
            #sparse_attributes.to_file(voxsel_fname, voxsel)

    # do a sanity check to ensure voxsel is set properly
    meta = voxsel.meta

    # helper function to check that surfaces have the same number of nodes
    def vertex_match(fld, meta=meta):
        return lambda s: surf.from_any(s).nvertices == meta[fld]

    # helper function to check that the fld in meta has the same value
    def eq_(fld, meta=meta):
        if meta[fld] is not None:
            return lambda s: meta[fld] == s
        else:
            return True

    def same_geometry(meta=meta):
        return lambda x: meta['volgeom'].same_geometry(volgeom.from_any(x))

    # comparisons to do with voxsel result - as a sanity check.
    # the values are a tuple (v, matches), where v is a value for which
    # it should hold that if v is not None, then matches(v) should
    # evaluate to True.
    comparisons = dict(geometry=(dataset, same_geometry),
                     pial=(pial_surf_fn, vertex_match('volsurf_nvertices')),
                     white=(white_surf_fn, vertex_match('volsurf_nvertices')),
                     interm=(intermediate_surf_fn, vertex_match('source_nvertices')),
                     start_fr=(start_fr, eq_('start_fr')),
                     stop_fr=(stop_fr, eq_('stop_fr')),
                     radius=(radius, eq_('radius')),
                     n_margin=(outside_node_margin, eq_('outside_node_margin')),
                     nv_mapping=(node_voxel_mapping, lambda x: x in meta['class'].lower()))

    for label, (v, matches) in comparisons.iteritems():
        if not v is None and not matches(v):
            raise ValueError('illegal value for %s: %s' % (label, v))

    # My pathetic attempt to work around default *_mm values
    # and keep using pre-generated voxsels
    if check:
        comparisons = dict(start_mm=(start_mm, eq_('start_mm')),
                     stop_mm=(stop_mm, eq_('stop_mm')))
        for label, (v, matches) in comparisons.iteritems():
            if not v is None and not matches(v):
                raise ValueError('illegal value for %s: %s' % (label, v))
        
    # all good - return a query engine
    qe = SurfaceVerticesQueryEngine(voxsel, add_fa=['center_distances'])

    return qe


class SurfaceSearchlightHyperalignment(ClassWithCollections):
    """
    Given a list of datasets, provide a list of mappers
    into common space using searchlight based hyperalignment.

    """
    ref_ds = Parameter(0, allowedtype='int', min=0,
            doc="""Index of a dataset to use as a reference.  First dataset
             is used as default.""")

    radius = Parameter(10., allowedtype='float32',
            doc=""" radius of surface disc in mm if float, number of voxels if int""")

    nproc = Parameter(None, allowedtype='int', min=1,
            doc="""Number of cores to use.
             """)

    nblocks = Parameter(100, allowedtype='int', min=1,
            doc="""Number of blocks to divide to process. More = less memory overload.
             """)

    hyperalignment = Parameter(Hyperalignment(ref_ds=0),
            doc="""Hyperalignment instance to be used in each searchlight sphere.
            Default is just the Hyperalignment instance with default parameters.""")

    combine_neighbormappers = Parameter(True, allowedtype='bool',
            doc="""This param determines whether to combine mappers for each voxel
            from its neighborhood searchlights or just use the mapper for which it is the
            center voxel.
            Use this option with caution, as enabling it might square the runtime memory
            requirement. If you run into memory issues, reduce the nproc in sl. """)

    voxsel_fnames = Parameter(None, allowedtype='list',
            doc="""A list of filenames with full path to store/load voxel selection. 
                   If this is left as None, voxel selection will be computed but not saved. 
                   If the filenames given already exist in that path, they are loaded instead
                   of recomputing the voxel selection. Surface files are not used in that case.
                   Length of this list should be equal to the length of the datasets list.""")

    pial_surf_fns = Parameter([], allowedtype='list',
            doc="""A list of filenames with full path to pial surface files. This is required input.    
                   Length of this list should be equal to the length of the datasets list.""")

    white_surf_fns = Parameter([], allowedtype='list',
            doc="""A list of filenames with full path to white matter surface files. This is required input.
                   Length of this list should be equal to the length of the datasets list.""")

    intermediate_surf_fns = Parameter([], allowedtype='list',
            doc="""A list of filenames with full path to intermediate surface files. This is required input.
                   Length of this list should be equal to the length of the datasets list.""")

    start_fr = Parameter(0.0, allowedtype='float',
            doc="""Fraction of gray matter to extend the normal inward.
            Please refer to surface voxel selection documentation for more info.""")

    stop_fr = Parameter(1.0, allowedtype='float',
            doc="""Fraction of gray matter to extend the normal outward.
            Please refer to surface voxel selection documentation for more info.""")

    start_mm = Parameter(0.0, allowedtype='float',
            doc="""Absolute start position of gray matter in mm after applying _fr values.
            Please refer to surface voxel selection documentation for more info.""")

    stop_mm = Parameter(0.0, allowedtype='float',
            doc="""Absolute length of gray matter in mm to extend the normal outward
            after applying _fr values.
            Please refer to surface voxel selection documentation for more info.""")

    node_voxel_mapping = Parameter('minimal', allowedtype='str',
        doc="""'minimal' or 'maximal'. If 'minimal' then each voxel is 
        associated with at most one node. If 'maximal' it is associated 
        with as many nodes that contain the voxel.""")
        
    zscore_all = Parameter(False, allowedtype='bool',
            doc="""Z-score all datasets prior hyperalignment. Turn it off
            if zscoring is not desired or was already performed. """)

    compute_recon = Parameter(True, allowedtype='bool',
            doc="""This param determines whether to compute reverse mappers for each 
            subject from common-space to subject space. These will be stored in the
            StaticProjectionMapper() and used when reverse() is called.
            Enabling it will almost double the size of the mappers returned.""")
    
    featsel = Parameter(1.0, allowedtype='float',
            doc="""Determines if feature selection will be performed in each searchlight.
            1.0: Use all features. <1.0 is understood as selecting that proportion of
            of feature in each searchlight using feature scores (Refer to the code);
            >1.0 is understood as selecting at most that many features in each searchlight.""")

    mask_node_ids = Parameter( None, allowedtype='list',
            doc="""You can specify a mask to compute searchlight hyperalignment only within this mask.
            These would be a list of surface nodes. Providing mask_node_ids forces the algorithm to compute connectivity hyperalignment
            only for the last entry of the resolution_radii.""")
    
    exclude_from_model = Parameter([],allowedtype='list',
            doc="""List of dataset indices that will not participate in building common model.
            These will still get mappers back but they don't influence the model or voxel selection.""")

    dtype = Parameter('float32', allowedtype='str',
            doc="""dtype of elements transformation matrices to save on memory for big datasets""")
            
    results_backend = Parameter('hdf5', allowedtype='str',
            doc=""" 'hdf5' or 'native' See Searchlight documentation.""")
            
    tmp_prefix = Parameter('/local/tmp/tmpsl', allowedtype='str',
            doc=""" 'hdf5' or 'native' See Searchlight documentation.""")
    
    def __init__(self, **kwargs):
        ClassWithCollections.__init__(self, **kwargs)
        self.projections = None
        self.projections_recon = None
        self.nfeatures_all = None
   
    def _proc_block(self, block, datasets, measure, qe, seed=None, iblock='main'):
        """Little helper to capture the parts of the computation that can be
        parallelized
        """
        if seed is not None:
            mvpa2.seed(seed)
        if __debug__:
            debug('SLC',
                  "Starting computing block for %i elements" % len(block))
        # put rois around all features in the dataset and compute the
        # measure within them
        bar = ProgressBar()
        ndatasets=len(qe)
        projections = [csc_matrix((self.nfeatures_all[isub], self.nfeatures_all[self.params.ref_ds]), 
                        dtype=self.params.dtype) for isub in range(ndatasets)]
        for i, node_id in enumerate(block):
            # retrieve the feature ids of all features in the ROI from the query
            # engine

            # Find the neighborhood for that selected nearest node
            try:
                roi_feature_ids = [qe[isub][node_id] for isub in range(ndatasets)]
            except KeyError:
                print "sub:%d node_id:%d"%(isub,node_id)
            # just making them lists instead of dataset-like
            roi_feature_ids = [rfi.samples.tolist()[0] for rfi in roi_feature_ids]
            # if qe returns zero-sized ROI for any subject, pass...
            if 0 in [len(rfi) for rfi in roi_feature_ids]:
                continue
            # selecting neighborhood for each subject for hyperalignment
            ds_temp = [datasets[isub][:,roi_feature_ids[isub]] for isub in range(ndatasets)]            
            roi_seed = np.zeros(ds_temp[self.params.ref_ds].nfeatures)
            # since the query engine output is sorted in the order of distance from center node
            # first voxel acts like center of searchlight disc, so making it roi_seed
            roi_seed[0] = 1
            ds_temp[self.params.ref_ds].fa['roi_seed'] =  roi_seed           
            if __debug__:
                msg = 'ROI (%i/%i), %i features' %(i + 1, len(block), len(roi_seed))
                debug('SLC', bar(float(i + 1) / len(block), msg), cr=True)                
            hmappers = measure(ds_temp)
            hmappers = hmappers.samples
            for isub in range(len(hmappers)):
                if not self.params.combine_neighbormappers:
                    I = roi_feature_ids[isub]
                    J = [roi_feature_ids[self.params.ref_ds][0]] * len(roi_feature_ids[isub])
                    V = hmappers[isub][0]['proj'].tolist()
                else:
                    I = []
                    J = []
                    V = []
                    for f2 in xrange(len(roi_feature_ids[self.params.ref_ds])):
                        I += roi_feature_ids[isub]
                        J += [roi_feature_ids[self.params.ref_ds][f2]] * len(roi_feature_ids[isub])
                        V += hmappers[isub][0]['proj'][:,f2].tolist()
                proj = coo_matrix((V,(I,J)), shape=(max(self.nfeatures_all[isub], max(I)+1),
                                            max(self.nfeatures_all[self.params.ref_ds],max(J)+1)), dtype=self.params.dtype)
                proj = proj.tocsc()
                # Cleaning up the current subject's projections to free up memory
                hmappers[isub,] = [[]]*hmappers.shape[1]
                projections[isub] = projections[isub] + proj

        if self.params.results_backend == 'native':
            return projections
        elif self.params.results_backend == 'hdf5':
            # store results in a temporary file and return a filename
            results_file = tempfile.mktemp(prefix=self.params.tmp_prefix,
                                           suffix='-%s.hdf5' % iblock)
            if __debug__:
                debug('SLC', "Storing results into %s" % results_file)
            h5save(results_file, projections)
            if __debug__:
                debug('SLC_', "Results stored")
            return results_file
        else:
            raise RuntimeError("Must not reach this point")

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
        ndatasets = len(datasets)
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
        if params.voxsel_fnames is None or not np.alltrue([os.path.isfile(vxfn) for vxfn in params.voxsel_fnames]):
            if params.pial_surf_fns is None or params.white_surf_fns is None or params.intermediate_surf_fns is None:
                raise ValueError, "pial, white, & intermediate surface file names can't be None."\
                    "They are required input arguments if pre-computed voxsel_fnames are given or present."
            if len(params.pial_surf_fns) != ndatasets:
                raise ValueError, "Length of pial_surf_fns %i doesn't match " \
                    "the length of the datasets list %i."% (len(params.pial_surf_fns), ndatasets)
            if len(params.white_surf_fns) != ndatasets:
                raise ValueError, "Length of white_surf_fns %i doesn't match " \
                    "the length of the datasets list %i."% (len(params.white_surf_fns), ndatasets)
            if len(params.intermediate_surf_fns) != ndatasets:
                raise ValueError, "Length of intermediate_surf_fns %i doesn't match " \
                    "the length of the datasets list %i."% (len(params.intermediate_surf_fns), ndatasets)
        
        hmeasure = HyperalignmentMeasure(ndatasets=ndatasets, featsel=params.featsel, ref_ds=params.ref_ds,
                      hyperalignment=params.hyperalignment, full_matrix=params.combine_neighbormappers,
                      use_same_features=False, exclude_from_model=params.exclude_from_model, dtype=params.dtype)
        # Zscoring time-series data if desired.
        if params.zscore_all:
            if __debug__:
                debug('SHPAL', "Zscoring datasets")
            for ds in datasets:
                zscore(ds, chunks_attr=None)
        
        nfeatures = datasets[params.ref_ds].nfeatures
        self.nfeatures_all = [sd.nfeatures for sd in datasets]
        if params.voxsel_fnames is not None:
            if len(params.voxsel_fnames) != ndatasets:
                raise ValueError, "Length of voxsel_fnames %i doesn't match " \
                    "the length of the datasets list %i."% (len(params.voxsel_fnames), ndatasets)
            qe = [compute_query_engine(voxsel_fname = params.voxsel_fnames[isub], dataset=datasets[isub], 
                    pial_surf_fn=params.pial_surf_fns[isub], white_surf_fn=params.white_surf_fns[isub], 
                    intermediate_surf_fn=params.intermediate_surf_fns[isub], radius=params.radius, 
                    start_fr=params.start_fr, stop_fr=params.stop_fr, 
                    start_mm=params.start_mm, stop_mm=params.stop_mm, 
                    nproc=params.nproc, 
                    node_voxel_mapping=params.node_voxel_mapping) for isub in range(ndatasets)]
        else:
            qe = [compute_query_engine(voxsel_fname = None, dataset=datasets[isub], pial_surf_fn=params.pial_surf_fns[isub],
                    white_surf_fn=params.white_surf_fns[isub], intermediate_surf_fn=params.intermediate_surf_fns[isub],
                    radius=params.radius, start_fr=params.start_fr, stop_fr=params.stop_fr, 
                    start_mm=params.start_mm, stop_mm=params.stop_mm, 
                    nproc=params.nproc, 
                    node_voxel_mapping=params.node_voxel_mapping) for isub in range(ndatasets)]
        # Train each qe with its dataset
        _=[qe[isub].train(datasets[isub]) for isub in range(ndatasets)]
        # Adding feature ids to keep track of features in a SL
        if __debug__:
            debug('SHPAL', "Adding Feature IDs")
        for sd in datasets:
            sd.fa['feature_ids'] = range(sd.nfeatures)
        if __debug__:
            debug('SHPAL', "Performing Hyperalignment")
        #vox2node_mapping =voxel2nearest_node(qe[params.ref_ds].voxsel)
        # Initialize projections
        self.projections = [csc_matrix((self.nfeatures_all[isub], nfeatures), dtype=params.dtype) for isub in range(ndatasets)]
        if params.compute_recon:
            #projections_recon = [csc_matrix((nfeatures, nfeatures_all[isub]), dtype=params.dtype) for isub in range(ndatasets)]
            self.projections_recon = [csc_matrix((nfeatures, self.nfeatures_all[isub]), dtype=params.dtype) for isub in range(ndatasets)]
        # XXX SORT the VOXELS (KEYS)??
        '''
        for ivox in xrange(nfeatures):   #range(nfeatures):
            debug('SHPAL',"Doing %d of %d [%.2f%%]"%(ivox,nfeatures, 100*ivox/float(nfeatures)), cr=True)
            # Find the nearest surface node for each voxel
            node_id = qe[params.ref_ds].feature_id2nearest_vertex_id(ivox, True)
        '''
        node_ids = qe[params.ref_ds].ids
        if params.mask_node_ids is not None and len(params.mask_node_ids) :
            node_ids = [ node_ids[i_] for i_  in params.mask_node_ids]
        # compute
        if params.nproc is not None and params.nproc > 1:
            # split all target ROIs centers into `nproc` equally sized blocks
            nproc_needed = min(len(node_ids), params.nproc)
            params.nblocks = nproc_needed \
                      if params.nblocks is None else params.nblocks
            params.nblocks = min(len(node_ids), params.nblocks)
            node_blocks = np.array_split(node_ids, params.nblocks)
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
            p_results = [
                    self._proc_block(node_ids, datasets, hmeasure, qe)]
        results_ds = self.__handle_all_results(p_results)
        # Dummy iterator for, you know, iteration
        for res in results_ds:
            pass
        datasets = None
        if params.compute_recon:
            self.projections = [StaticProjectionMapper(proj=proj, recon=proj_recon) 
                            for proj,proj_recon in zip(self.projections,self.projections_recon)]
        else:
            self.projections = [StaticProjectionMapper(proj=proj) for proj in self.projections]
        return self.projections

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
