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
    from mvpa.base import debug

import numpy as np

from mvpa.base import externals, warning
from mvpa.base.dochelpers import borrowkwargs

from mvpa.datasets import Dataset, hstack
from mvpa.support import copy
from mvpa.mappers.base import FeatureSliceMapper
from mvpa.measures.base import DatasetMeasure
from mvpa.misc.state import StateVariable
from mvpa.misc.neighborhood import IndexQueryEngine, Sphere
from mvpa.base.dochelpers import _str, borrowkwargs

class Searchlight(DatasetMeasure):
    """An implementation of searchlight measure.

    The idea for a searchlight algorithm stems from a paper by
    :ref:`Kriegeskorte et al. (2006) <KGB06>`.
    """

    roisizes = StateVariable(enabled=False,
        doc="Number of features in each ROI.")

    def __init__(self, datameasure, queryengine, center_ids=None,
                 nproc=None, **kwargs):
        """
        Parameters
        ----------
        datameasure : callable
          Any object that takes a :class:`~mvpa.datasets.base.Dataset`
          and returns some measure when called.
        queryengine : QueryEngine
          Engine to use to discover the "neighborhood" of each feature.
          See :class:`~mvpa.misc.neighborhood.QueryEngine`.
        center_ids : None or list of int
          List of feature ids (not coordinates) the shall serve as sphere
          centers. By default all features will be used.
        nproc : None or int
          How many processes to use for computation.  Requires `pprocess`
          external module.  If None -- all available cores will be used.
        **kwargs
          In addition this class supports all keyword arguments of its
          base-class :class:`~mvpa.measures.base.DatasetMeasure`.
      """
        DatasetMeasure.__init__(self, **(kwargs))

        if nproc > 1 and not externals.exists('pprocess'):
            raise RuntimeError("The 'pprocess' module is required for "
                               "multiprocess searchlights. Please either "
                               "install python-pprocess, or reduce `nproc` "
                               "to 1 (got nproc=%i)" % nproc)

        self.__datameasure = datameasure
        self.__qe = queryengine
        if center_ids is not None and not len(center_ids):
            raise ValueError, \
                  "Cannot run searchlight on an empty list of center_ids"
        self.__center_ids = center_ids
        self.__nproc = nproc


    def _call(self, dataset):
        """Perform the ROI search.
        """
        # local binding
        nproc = self.__nproc

        if nproc is None and externals.exists('pprocess'):
            import pprocess
            try:
                nproc = pprocess.get_number_of_cores() or 1
            except AttributeError:
                warning("pprocess version %s has no API to figure out maximal "
                        "number of cores. Using 1" % externals.versions['pprocess'])
                nproc = 1
        # train the queryengine
        self.__qe.train(dataset)

        # decide whether to run on all possible center coords or just a provided
        # subset
        if self.__center_ids is not None:
            roi_ids = self.__center_ids
            # safeguard against stupidity
            if __debug__:
                if max(roi_ids) >= dataset.nfeatures:
                    raise IndexError, \
                          "Maximal center_id found is %s whenever given " \
                          "dataset has only %d features" \
                          % (max(roi_ids), dataset.nfeatures)
        else:
            roi_ids = np.arange(dataset.nfeatures)

        # compute
        if nproc > 1:
            # split all target ROIs centers into `nproc` equally sized chunks
            roi_chunks = np.array_split(roi_ids, nproc)

            # the next block sets up the infrastructure for parallel computing
            # this can easily be changed into a ParallelPython loop, if we
            # decide to have a PP job server in PyMVPA
            import pprocess
            p_results = pprocess.Map(limit=nproc)
            compute = p_results.manage(
                        pprocess.MakeParallel(self._proc_chunk))
            for chunk in roi_chunks:
                # should we maybe deepcopy the measure to have a unique and
                # independent one per process?
                compute(chunk, dataset, copy.copy(self.__datameasure))

            # collect results
            results = []
            if self.ca.is_enabled('roisizes'):
                roisizes = []
            else:
                roisizes = None

            for r, rsizes in p_results:
                results += r
                if not roisizes is None:
                    roisizes += rsizes
        else:
            # otherwise collect the results in a list
            results, roisizes = \
                    self._proc_chunk(roi_ids, dataset, self.__datameasure)

        if not roisizes is None:
            self.ca.roisizes = roisizes

        if __debug__:
            debug('SLC', '')

        # but be careful: this call also serves as conversion from parallel maps
        # to regular lists!
        # this uses the Dataset-hstack
        results = hstack(results)

        if 'mapper' in dataset.a:
            # since we know the space we can stick the original mapper into the
            # results as well
            if self.__center_ids is None:
                results.a['mapper'] = copy.copy(dataset.a.mapper)
            else:
                # there is an additional selection step that needs to be
                # expressed by another mapper
                mapper = copy.copy(dataset.a.mapper)
                mapper.append(FeatureSliceMapper(self.__center_ids,
                                                 dshape=dataset.shape[1:]))
                results.a['mapper'] = mapper

        # charge state
        self.ca.raw_results = results

        # return raw results, base-class will take care of transformations
        return results


    def _proc_chunk(self, chunk, ds, measure):
        """Little helper to capture the parts of the computation that can be
        parallelized
        """
        if self.ca.is_enabled('roisizes'):
            roisizes = []
        else:
            roisizes = None
        results = []
        # put rois around all features in the dataset and compute the
        # measure within them
        for i, f in enumerate(chunk):
            # retrieve the feature ids of all features in the ROI from the query
            # engine
            roi_fids = self.__qe[f]
            # slice the dataset
            roi = ds[:, roi_fids]

            # compute the datameasure and store in results
            results.append(measure(roi))

            # store the size of the roi dataset
            if not roisizes is None:
                roisizes.append(roi.nfeatures)

            if __debug__:
                debug('SLC', "Doing %i ROIs: %i (%i features) [%i%%]" \
                    % (len(chunk),
                       f+1,
                       roi.nfeatures,
                       float(i+1)/len(chunk)*100,), cr=True)

        return results, roisizes


@borrowkwargs(Searchlight, '__init__')
def sphere_searchlight(datameasure, radius=1, center_ids=None,
                       space='voxel_indices', **kwargs):
    """Creates a `Searchlight` to run a scalar `DatasetMeasure` on
    all possible spheres of a certain size within a dataset.

    The idea for a searchlight algorithm stems from a paper by
    :ref:`Kriegeskorte et al. (2006) <KGB06>`.

    Parameters
    ----------
    datameasure : callable
      Any object that takes a :class:`~mvpa.datasets.base.Dataset`
      and returns some measure when called.
    radius : float
      All features within this radius around the center will be part
      of a sphere.
    center_ids : list of int
      List of feature ids (not coordinates) the shall serve as sphere
      centers. By default all features will be used.
    space : str
      Name of a feature attribute of the input dataset that defines the spatial
      coordinates of all features.
    **kwargs
      In addition this class supports all keyword arguments of its
      base-class :class:`~mvpa.measures.base.DatasetMeasure`.

    Notes
    -----
    If `Searchlight` is used as `SensitivityAnalyzer` one has to make
    sure that the specified scalar `DatasetMeasure` returns large
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
    return Searchlight(datameasure, qe, center_ids=center_ids, **kwargs)


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
