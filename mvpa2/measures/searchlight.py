# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Implementation of the Searchlight algorithm"""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa2.base import debug

import numpy as np

from mvpa2.base import externals, warning
from mvpa2.base.types import is_datasetlike
from mvpa2.base.dochelpers import borrowkwargs, _repr_attrs

from mvpa2.datasets import hstack
from mvpa2.support import copy
from mvpa2.featsel.base import StaticFeatureSelection
from mvpa2.measures.base import Measure
from mvpa2.base.state import ConditionalAttribute
from mvpa2.misc.neighborhood import IndexQueryEngine, Sphere
from mvpa2.mappers.base import ChainMapper

class BaseSearchlight(Measure):
    """Base class for searchlights.

    The idea for a searchlight algorithm stems from a paper by
    :ref:`Kriegeskorte et al. (2006) <KGB06>`.
    """

    roi_sizes = ConditionalAttribute(enabled=False,
        doc="Number of features in each ROI.")

    is_trained = True
    """Indicate that this measure is always trained."""


    def __init__(self, queryengine, roi_ids=None, nproc=None, **kwargs):
        """
        Parameters
        ----------
        queryengine : QueryEngine
          Engine to use to discover the "neighborhood" of each feature.
          See :class:`~mvpa2.misc.neighborhood.QueryEngine`.
        roi_ids : None or list(int) or str
          List of feature ids (not coordinates) the shall serve as ROI seeds
          (e.g. sphere centers). Alternatively, this can be the name of a
          feature attribute of the input dataset, whose non-zero values
          determine the feature ids. By default all features will be used.
        nproc : None or int
          How many processes to use for computation.  Requires `pprocess`
          external module.  If None -- all available cores will be used.
        **kwargs
          In addition this class supports all keyword arguments of its
          base-class :class:`~mvpa2.measures.base.Measure`.
      """
        Measure.__init__(self, **kwargs)

        if nproc > 1 and not externals.exists('pprocess'):
            raise RuntimeError("The 'pprocess' module is required for "
                               "multiprocess searchlights. Please either "
                               "install python-pprocess, or reduce `nproc` "
                               "to 1 (got nproc=%i)" % nproc)

        self._queryengine = queryengine
        if roi_ids is not None and not isinstance(roi_ids, str) \
                and not len(roi_ids):
            raise ValueError, \
                  "Cannot run searchlight on an empty list of roi_ids"
        self.__roi_ids = roi_ids
        self.nproc = nproc


    def __repr__(self, prefixes=[]):
        """String representation of a `Measure`

        Includes only arguments which differ from default ones
        """
        return super(BaseSearchlight, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['queryengine', 'roi_ids', 'nproc']))


    def _call(self, dataset):
        """Perform the ROI search.
        """
        # local binding
        nproc = self.nproc

        if nproc is None and externals.exists('pprocess'):
            import pprocess
            try:
                nproc = pprocess.get_number_of_cores() or 1
            except AttributeError:
                warning("pprocess version %s has no API to figure out maximal "
                        "number of cores. Using 1"
                        % externals.versions['pprocess'])
                nproc = 1
        # train the queryengine
        self._queryengine.train(dataset)

        # decide whether to run on all possible center coords or just a provided
        # subset
        if isinstance(self.__roi_ids, str):
            roi_ids = dataset.fa[self.__roi_ids].value.nonzero()[0]
        elif self.__roi_ids is not None:
            roi_ids = self.__roi_ids
            # safeguard against stupidity
            if __debug__:
                if max(roi_ids) >= len(self._queryengine):
                    raise IndexError(
                          "Maximal center_id found is %s whenever query "
                          "engine %s knows only about %d"
                          % (max(roi_ids), self._queryengine,
                             len(self._queryengine)))
        else:
            roi_ids = np.arange(len(self._queryengine))

        # pass to subclass
        results, roi_sizes = self._sl_call(dataset, roi_ids, nproc)

        if not roi_sizes is None:
            self.ca.roi_sizes = roi_sizes

        if 'mapper' in dataset.a:
            # since we know the space we can stick the original mapper into the
            # results as well
            if self.__roi_ids is None:
                results.a['mapper'] = copy.copy(dataset.a.mapper)
            else:
                # there is an additional selection step that needs to be
                # expressed by another mapper
                mapper = copy.copy(dataset.a.mapper)
                
                # NNO if the orignal mapper has no append (because it's not a
                # chainmapper, for example), we make our own chainmapper.
                #
                # THe original code was:
                # mapper.append(StaticFeatureSelection(roi_ids,
                #                                     dshape=dataset.shape[1:])) 
                feat_sel_mapper=StaticFeatureSelection(roi_ids,
                                                     dshape=dataset.shape[1:])
                if 'append' in dir(mapper):
                    mapper.append(feat_sel_mapper)
                else:
                    mapper = ChainMapper([dataset.a.mapper,
                                          feat_sel_mapper])
                    
                results.a['mapper'] = mapper

        # charge state
        self.ca.raw_results = results

        # return raw results, base-class will take care of transformations
        return results


    def _sl_call(self, dataset, roi_ids, nproc):
        """Classical generic searchlight implementation
        """
        raise NotImplementedError("Must be implemented in the derived classes")

    queryengine = property(fget=lambda self: self._queryengine)
    roi_ids = property(fget=lambda self: self.__roi_ids)


class Searchlight(BaseSearchlight):
    """The implementation of a generic searchlight measure.

    The idea for a searchlight algorithm stems from a paper by
    :ref:`Kriegeskorte et al. (2006) <KGB06>`.  As a result it
    produces a map of measures given a `datameasure` instance of
    interest, which is ran at each spatial location.
    """

    def __init__(self, datameasure, queryengine, add_center_fa=False, **kwargs):
        """
        Parameters
        ----------
        datameasure : callable
          Any object that takes a :class:`~mvpa2.datasets.base.Dataset`
          and returns some measure when called.
        add_center_fa : bool or str
          If True or a string, each searchlight ROI dataset will have a boolean
          vector as a feature attribute that indicates the feature that is the
          seed (e.g. sphere center) for the respective ROI. If True, the
          attribute is named 'roi_seed', the provided string is used as the name
          otherwise.
        **kwargs
          In addition this class supports all keyword arguments of its
          base-class :class:`~mvpa2.measures.searchlight.BaseSearchlight`.
        """
        BaseSearchlight.__init__(self, queryengine, **kwargs)
        self.datameasure = datameasure
        if isinstance(add_center_fa, str):
            self.__add_center_fa = add_center_fa
        elif add_center_fa:
            self.__add_center_fa = 'roi_seed'
        else:
            self.__add_center_fa = False

    def __repr__(self, prefixes=[]):
        return super(Searchlight, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['datameasure'])
            + _repr_attrs(self, ['add_center_fa'], default=False)
            )


    def _sl_call(self, dataset, roi_ids, nproc):
        """Classical generic searchlight implementation
        """
        # compute
        if nproc > 1:
            # split all target ROIs centers into `nproc` equally sized blocks
            nproc_needed = min(len(roi_ids), nproc)
            roi_blocks = np.array_split(roi_ids, nproc_needed)

            # the next block sets up the infrastructure for parallel computing
            # this can easily be changed into a ParallelPython loop, if we
            # decide to have a PP job server in PyMVPA
            import pprocess
            p_results = pprocess.Map(limit=nproc_needed)
            if __debug__:
                debug('SLC', "Starting off child processes for nproc=%i"
                      % nproc_needed)
            compute = p_results.manage(
                        pprocess.MakeParallel(self._proc_block))
            for block in roi_blocks:
                # should we maybe deepcopy the measure to have a unique and
                # independent one per process?
                compute(block, dataset, copy.copy(self.__datameasure))

            # collect results
            results = []
            if self.ca.is_enabled('roi_sizes'):
                roi_sizes = []
            else:
                roi_sizes = None

            for r, rsizes in p_results:
                results += r
                if not roi_sizes is None:
                    roi_sizes += rsizes
        else:
            # otherwise collect the results in a list
            results, roi_sizes = \
                    self._proc_block(roi_ids, dataset, self.__datameasure)

        if __debug__ and 'SLC' in debug.active:
            debug('SLC', '')            # just newline
            resshape = len(results) and np.asanyarray(results[0]).shape or 'N/A'
            debug('SLC', ' hstacking %d results of shape %s'
                  % (len(results), resshape))

        # but be careful: this call also serves as conversion from parallel maps
        # to regular lists!
        # this uses the Dataset-hstack
        results = hstack(results)

        if __debug__:
            debug('SLC', " hstacked shape %s" % (results.shape,))

        return results, roi_sizes


    def _proc_block(self, block, ds, measure):
        """Little helper to capture the parts of the computation that can be
        parallelized
        """
        if __debug__:
            debug_slc_ = 'SLC_' in debug.active
            debug('SLC',
                  "Starting computing block for %i elements" % len(block))
        if self.ca.is_enabled('roi_sizes'):
            roi_sizes = []
        else:
            roi_sizes = None
        results = []
        # put rois around all features in the dataset and compute the
        # measure within them
        for i, f in enumerate(block):
            # retrieve the feature ids of all features in the ROI from the query
            # engine
            roi_specs = self._queryengine[f]

            if __debug__ and  debug_slc_:
                debug('SLC_', 'For %r query returned ids %r' % (f, roi_fids))

            if is_datasetlike(roi_specs):
                # TODO: unittest
                assert(len(roi_specs) == 1)
                roi_fids = roi_specs.samples[0]
            else:
                roi_fids = roi_specs

            # slice the dataset
            roi = ds[:, roi_fids]

            if is_datasetlike(roi_specs):
                for n,v in roi_specs.fa.iteritems():
                    roi.fa[n] = v

            if self.__add_center_fa:
                # add fa to indicate ROI seed if requested
                roi_seed = np.zeros(roi.nfeatures, dtype='bool')
                roi_seed[roi_fids.index(f)] = True
                roi.fa[self.__add_center_fa] = roi_seed

            # compute the datameasure and store in results
            results.append(measure(roi))

            # store the size of the roi dataset
            if not roi_sizes is None:
                roi_sizes.append(roi.nfeatures)

            if __debug__:
                debug('SLC', "Doing %i ROIs: %i (%i features) [%i%%]" \
                    % (len(block),
                       f+1,
                       roi.nfeatures,
                       float(i+1)/len(block)*100,), cr=True)

        return results, roi_sizes


    def __set_datameasure(self, datameasure):
        """Set the datameasure"""
        self.untrain()
        self.__datameasure = datameasure

    datameasure = property(fget=lambda self: self.__datameasure,
                           fset=__set_datameasure)
    add_center_fa = property(fget=lambda self: self.__add_center_fa)

@borrowkwargs(Searchlight, '__init__', exclude=['roi_ids'])
def sphere_searchlight(datameasure, radius=1, center_ids=None,
                       space='voxel_indices', **kwargs):
    """Creates a `Searchlight` to run a scalar `Measure` on
    all possible spheres of a certain size within a dataset.

    The idea for a searchlight algorithm stems from a paper by
    :ref:`Kriegeskorte et al. (2006) <KGB06>`.

    Parameters
    ----------
    datameasure : callable
      Any object that takes a :class:`~mvpa2.datasets.base.Dataset`
      and returns some measure when called.
    radius : int
      All features within this radius around the center will be part
      of a sphere. Radius is in grid-indices, i.e. ``1`` corresponds
      to all immediate neighbors, regardless of the physical distance.
    center_ids : list of int
      List of feature ids (not coordinates) the shall serve as sphere
      centers. Alternatively, this can be the name of a feature attribute
      of the input dataset, whose non-zero values determine the feature
      ids.  By default all features will be used (it is passed as ``roi_ids``
      argument of Searchlight).
    space : str
      Name of a feature attribute of the input dataset that defines the spatial
      coordinates of all features.
    **kwargs
      In addition this class supports all keyword arguments of its
      base-class :class:`~mvpa2.measures.base.Measure`.

    Notes
    -----
    If `Searchlight` is used as `SensitivityAnalyzer` one has to make
    sure that the specified scalar `Measure` returns large
    (absolute) values for high sensitivities and small (absolute) values
    for low sensitivities. Especially when using error functions usually
    low values imply high performance and therefore high sensitivity.
    This would in turn result in sensitivity maps that have low
    (absolute) values indicating high sensitivities and this conflicts
    with the intended behavior of a `SensitivityAnalyzer`.
    """
    # build a matching query engine from the arguments
    kwa = {space: Sphere(radius)}
    qe = IndexQueryEngine(**kwa)
    # init the searchlight with the queryengine
    return Searchlight(datameasure, queryengine=qe, roi_ids=center_ids,
                       **kwargs)


#class OptimalSearchlight( object ):
#    def __init__( self,
#                  searchlight,
#                  test_radii,
#                  verbose=False,
#                  **kwargs ):
#        """
#        """
#        # results will end up here
#        self.__perfmeans = []
#        self.__perfvars = []
#        self.__chisquares = []
#        self.__chanceprobs = []
#        self.__spheresizes = []
#
#        # run searchligh for all radii in the list
#        for radius in test_radii:
#            if verbose:
#                print 'Using searchlight with radius:', radius
#            # compute the results
#            searchlight( radius, **(kwargs) )
#
#            self.__perfmeans.append( searchlight.perfmean )
#            self.__perfvars.append( searchlight.perfvar )
#            self.__chisquares.append( searchlight.chisquare )
#            self.__chanceprobs.append( searchlight.chanceprob )
#            self.__spheresizes.append( searchlight.spheresize )
#
#
#        # now determine the best classification accuracy
#        best = np.array(self.__perfmeans).argmax( axis=0 )
#
#        # select the corresponding values of the best classification
#        # in all data tables
#        self.perfmean   = best.choose(*(self.__perfmeans))
#        self.perfvar    = best.choose(*(self.__perfvars))
#        self.chisquare  = best.choose(*(self.__chisquares))
#        self.chanceprob = best.choose(*(self.__chanceprobs))
#        self.spheresize = best.choose(*(self.__spheresizes))
#
#        # store the best performing radius
#        self.bestradius = np.zeros( self.perfmean.shape, dtype='uint' )
#        self.bestradius[searchlight.mask==True] = \
#            best.choose( test_radii )[searchlight.mask==True]
#
#
#
#def makeSphericalROIMask( mask, radius, elementsize=None ):
#    """
#    """
#    # use default elementsize if none is supplied
#    if not elementsize:
#        elementsize = [ 1 for i in range( len(mask.shape) ) ]
#    else:
#        if len( elementsize ) != len( mask.shape ):
#            raise ValueError, 'elementsize does not match mask dimensions.'
#
#    # rois will be drawn into this mask
#    roi_mask = np.zeros( mask.shape, dtype='int32' )
#
#    # while increase with every ROI
#    roi_id_counter = 1
#
#    # build spheres around every non-zero value in the mask
#    for center, spheremask in \
#        algorithms.SpheresInMask( mask,
#                                  radius,
#                                  elementsize,
#                                  forcesphere = True ):
#
#        # set all elements that match the current spheremask to the
#        # current ROI index value
#        roi_mask[spheremask] = roi_id_counter
#
#        # increase ROI counter
#        roi_id_counter += 1
#
#    return roi_mask
