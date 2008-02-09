#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Implementation of the Searchlight algorithm"""

__docformat__ = 'restructuredtext'

import numpy as N

if __debug__:
    from mvpa.misc import debug

from mvpa.datasets.mappeddataset import MappedDataset
from mvpa.datasets.mapper import MetricMapper
from mvpa.algorithms.datameasure import SensitivityAnalyzer


class Searchlight(SensitivityAnalyzer):
    """Runs a `ScalarDatasetMeasure` on all possible spheres of a certain size
    within a dataset.

    The idea to use a searchlight as a sensitivity analyser stems from this
    paper:

      Kriegeskorte, N., Goebel, R. & Bandettini, P. (2006).
      'Information-based functional brain mapping.' Proceedings of the
      National Academy of Sciences of the United States of America 103,
      3863-3868.
    """
    def __init__(self, datameasure,
                 combinefx=N.array,
                 radius=1.0):
        """Initialize Searchlight to compute 'datameasure' for each sphere with
        a certain 'radius' in a given dataset (see __call__()).

        The results of the datameasures of all spheres are passed to 'combinefx'
        and the output of that call is returned.

        ATTENTION: If `Searchlight` is used as `SensitivityAnalyzer` one has to
        make sure that the specified `ScalarDatasetMeasure` returns large (absolute)
        values for high sensitivities and small (absolute) values for low
        sensitivities. Especially when using error functions usually low values
        imply high performance and therefore high sensitivity. This would in
        turn result in sensitivity maps that have low (absolute) values
        indicating high sensitivites and this conflicts with the intended#
        behavior of a `SensitivityAnalyzer`.
        """
        SensitivityAnalyzer.__init__(self)

        self.__datameasure = datameasure
        self.__radius = radius
        self.__combinefx = combinefx
        self.__spheresizes = []


    def _resetState(self):
        """Don't touch me"""
        self.__spheresizes = []


    def __call__(self, dataset, callables=[]):
        """Perform the spheresearch.
        """
        if not isinstance(dataset, MappedDataset) \
           or not isinstance(dataset.mapper, MetricMapper):
            raise ValueError, "Searchlight only works with MappedDatasets " \
                              "that make use of a mapper with information " \
                              "about the dataspace metrics."

        self._resetState()

        if __debug__:
            nspheres = dataset.nfeatures
            sphere_count = 0

        # collect the results in a list -- you never know what you get
        results = []

        # put spheres around all features in the dataset and compute the
        # measure within them
        for f in xrange(dataset.nfeatures):
            sphere = dataset.selectFeatures(
                dataset.mapper.getNeighbors(f, self.__radius),
                plain=True)

            # compute the datameasure and store in results
            # XXX implement callbacks!
            measure = self.__datameasure(sphere)
            results.append(measure)

            # store the size of the sphere dataset
            self.__spheresizes.append(sphere.nfeatures)

            if __debug__:
                sphere_count += 1
                debug('SLC', "Doing %i spheres: %i (%i features) [%i%%]" \
                    % (nspheres,
                       sphere_count,
                       sphere.nfeatures,
                       float(sphere_count)/nspheres*100,), cr=True)

        if __debug__:
            debug('SLC', '')

        # transform the results with the user-supplied function and return
        return self.__combinefx(results)

    spheresizes = property(fget=lambda self: self.__spheresizes)



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
