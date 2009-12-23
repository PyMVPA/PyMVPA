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

import numpy as N

from mvpa.measures.base import DatasetMeasure
from mvpa.misc.state import StateVariable
from mvpa.misc.neighborhood import IndexQueryEngine, Sphere


class Searchlight(DatasetMeasure):

    roisizes = StateVariable(enabled=False,
        doc="Number of features in each ROI.")

    def __init__(self, datameasure, queryengine, center_ids=None, **kwargs):
        DatasetMeasure.__init__(self, **(kwargs))

        self.__datameasure = datameasure
        self.__qe = queryengine
        self.__center_ids = center_ids


    def _call(self, dataset):
        """Perform the ROI search.
        """
        if self.states.isEnabled('roisizes'):
            self.states.roisizes = []

        # train the queryengine
        self.__qe.train(dataset)

        # progress info stuff -- not critical
        if __debug__:
            if not self.__center_ids == None:
                nrois = len(self.__center_ids)
            else:
                nrois = dataset.nfeatures
            roi_count = 0

        # collect the results in a list -- you never know what you get
        results = []

        # decide whether to run on all possible center coords or just a provided
        # subset
        if not self.__center_ids == None:
            generator = self.__center_ids
        else:
            generator = xrange(dataset.nfeatures)

        # put rois around all features in the dataset and compute the
        # measure within them
        for f in generator:
            # retrieve the feature ids of all features in the ROI from the query
            # engine
            roi_fids = self.__qe[f]
            # slice the dataset
            roi = dataset[:, roi_fids]

            # compute the datameasure and store in results
            measure = self.__datameasure(roi)
            results.append(N.squeeze(measure))

            # store the size of the roi dataset
            if self.states.isEnabled('roisizes'):
                self.states.roisizes.append(roi.nfeatures)

            if __debug__:
                roi_count += 1
                debug('SLC', "Doing %i ROIs: %i (%i features) [%i%%]" \
                    % (nrois,
                       roi_count,
                       roi.nfeatures,
                       float(roi_count)/nrois*100,), cr=True)

        if __debug__:
            debug('SLC', '')

        # charge state
        self.states.raw_results = results

        # return raw results, base-class will take care of transformations
        return results



def sphere_searchlight(datameasure, diameter=1, center_ids=None,
                       space='voxel_indices', **kwargs):
    """Runs a scalar `DatasetMeasure` on all possible spheres of a certain size
    within a dataset.

    The idea for a searchlight algorithm stems from a paper by
    :ref:`Kriegeskorte et al. (2006) <KGB06>`.

    Parameters
    ----------
    datameasure: callable
      Any object that takes a :class:`~mvpa.datasets.base.Dataset`
      and returns some measure when called.
    diameter: int
      All features within the diameteraround the center will be part
      of a sphere.
    center_ids: list(int)
      List of feature ids (not coordinates) the shall serve as sphere
      centers. By default all features will be used.
    **kwargs
      In additions this class supports all keyword arguments of its
      base-class :class:`~mvpa.measures.base.DatasetMeasure`.

    Notes
    -----
    If `Searchlight` is used as `SensitivityAnalyzer` one has to make
    sure that the specified scalar `DatasetMeasure` returns large
    (absolute) values for high sensitivities and small (absolute) values
    for low sensitivities. Especially when using error functions usually
    low values imply high performance and therefore high sensitivity.
    This would in turn result in sensitivity maps that have low
    (absolute) values indicating high sensitivites and this conflicts
    with the intended behavior of a `SensitivityAnalyzer`.
    """
    # build a matching query engine from the arguments
    # XXX right now Sphere wants an odd integer diameter
    kwa = {space: Sphere(diameter)}
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
#        best = N.array(self.__perfmeans).argmax( axis=0 )
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
#        self.bestradius = N.zeros( self.perfmean.shape, dtype='uint' )
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
#    roi_mask = N.zeros( mask.shape, dtype='int32' )
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
