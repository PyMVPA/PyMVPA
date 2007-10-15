### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Implementation of the Searchlight algorithm -- loosely implemented
#            after:
#
#            Kriegeskorte, N., Goebel, R. & Bandettini, P. (2006).
#            'Information-based functional brain mapping.' Proceedings of the
#            National Academy of Sciences of the United States of America 103,
#            3863-3868.
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2 of the License, or (at your option) any later version.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import algorithms
import numpy as np
import support
import crossval
import sys
import stats


class Searchlight( object ):
    """ Perform a cross-validation analysis in all possible spheres of a
    certain size within a mask in the dataspace.

    The algorithm is loosely implemented after:

      Kriegeskorte, N., Goebel, R. & Bandettini, P. (2006).
      'Information-based functional brain mapping.' Proceedings of the
      National Academy of Sciences of the United States of America 103,
      3863-3868.

    The analysis results are a map of the mean classification performance
    and their variance. Additionally a chisquare value per sphere and the
    associated p-value provide information about the signal strength
    within a certain sphere.

    The object provides a number of properties that can be used to access
    input and output data of the searchlight algorithm.

    The following properties provide access to arrays that match the size of
    the mask shape. Each element contains information about the sphere that
    is centered on the respective location in the mask:

        perfmean   - mean classification performance
        perfvar    - variance of classification performance
        chisquare  - value of the chisquare test of classifications being
                     equally distributed over all cells in the
                     target x predictions contingency table
        chanceprob - probability of classifications being equally distributed
                     over all cells in the target x prediction contingency
                     table, i.e. high probability means low classification
                     performance and low signal in the data
        spheresize - number of features in the sphere


    These properties give access to the input data and other status variables:

        pattern     - the MVPAPattern object that holds the data set
        mask        - the mask that is used to generate the spheres
        clf         - the classifier instance that is used
        radius      - the sphere radius in an arbitrary unit (see elementsize
                      property)
        elementsize - a vector specifying the extent of each data element
                      along all dimensions in the dataset. This information is
                      used to translate the radius into element units
        forcesphere - if True a full sphere is used regardless of the status
                      of the status of the sphere elements in the mask. If
                      False only elements are considered as sphere elements
                      that have a non-zero value in the mask.
    """
    def __init__( self, pattern, mask,
                  classifier,
                  elementsize = None,
                  forcesphere = False,
                  verbose = False ):
        """
        Parameters:
            pattern:      
            mask:         
            classifier:   
            elementsize:  a vector specifying the extent of each data element
                          along all dimensions in the dataset. This information
                          is used to translate the radius passed to the
                          __call__() method into element units
            forcesphere:  if True a full sphere is used regardless of the
                          status of the status of the sphere elements in the
                          mask. If False only elements are considered as sphere
                          elements that have a non-zero value in the mask.
            verbose:      Is set to True some status messages are enabled.
        """
        self.__pattern = pattern
        self.__mask = mask
        self.__verbose = verbose

        self.setClassifier( classifier )

        if not elementsize:
            self.__elementsize = [ 1 for i in range( len(pattern.mapper.dsshape) ) ]
        else:
            if not len( elementsize ) == len( mask.shape ):
                raise ValueError, 'elementsize does not match mask dimensions.'
            self.__elementsize = elementsize

        self.__forcesphere = forcesphere

        if not mask.shape == pattern.mapper.dsshape:
            raise ValueError, 'Mask shape has to match the datasets dataspace shape.'

        self.__clearResults()


    def __clearResults( self ):
        # init the result maps
        self.__perfmean = np.zeros(self.pattern.mapper.dsshape)
        self.__perfvar = np.zeros(self.pattern.mapper.dsshape)
        self.__chisquare = np.zeros(self.pattern.mapper.dsshape)
        self.__chanceprob = np.zeros(self.pattern.mapper.dsshape)
        self.__spheresize = np.zeros(self.pattern.mapper.dsshape, dtype='uint')


    def setClassifier( self, classifier ):
        """ Set a classifier instance that is used to perform the
        classification.
        """
        self.__clf = classifier


    def __call__( self, radius = 1.0, **kwargs ):
        """ Perform the spheresearch for all possible spheres in the
        mask.

        Additional keyword arguments that are passed to the CrossValidation
        class.
        """
        # cleanup prior runs first
        self.__clearResults()

        if self.__verbose:
            nspheres = np.array( self.mask.nonzero() ).shape[1]
            sphere_count = 0

        # for all possible spheres in the mask
        for center, spheremask in \
            algorithms.SpheresInMask( self.__mask,
                                      radius,
                                      self.__elementsize,
                                      self.__forcesphere ):
            # select features inside the sphere
            masked = self.__pattern.selectFeaturesByMask( spheremask )

            # do the cross-validation
            cv = crossval.CrossValidation( masked, self.__clf, **(kwargs) )
            # run cross-validation
            cv()

            # store the performance value as a vector
            perf = np.array( cv.perf )

            # translate center coordinate into array slicing index
            center_index = np.transpose(
                               np.array( center, ndmin=2 ) ).tolist()

            # store the interesting information
            # mean performance
            self.__perfmean[center_index] = perf.mean()
            # performance variance
            self.__perfvar[center_index] = perf.var()
            # significantly different from chance?
            self.__chisquare[center_index], \
            self.__chanceprob[center_index] = \
                stats.chisquare( cv.contingencytbl )
            # spheresize / number of features
            self.__spheresize[center_index] = masked.nfeatures

            if self.__verbose:
                sphere_count += 1
                print "\rDoing %i spheres: %i (%i%%)" \
                    % (nspheres, sphere_count, float(sphere_count)/nspheres*100,),
                sys.stdout.flush()

        if self.__verbose:
            print ''


    # access to the results
    perfmean = property( fget=lambda self: self.__perfmean )
    perfvar = property( fget=lambda self: self.__perfvar )
    chisquare = property( fget=lambda self: self.__chisquare )
    chanceprob = property( fget=lambda self: self.__chanceprob )
    spheresize = property( fget=lambda self: self.__spheresize )

    # other data access
    pattern = property( fget=lambda self: self.__pattern )
    mask = property( fget=lambda self: self.__mask )
    clf  = property( fget=lambda self: self.__clf,
                     fset=setClassifier )
    elementsize = property( fget=lambda self: self.__elementsize )
    forcesphere = property( fget=lambda self: self.__forcesphere )



class OptimalSearchlight( object ):
    def __init__( self,
                  searchlight,
                  test_radii,
                  verbose=False,
                  **kwargs ):
        """
        """
        # results will end up here
        self.__perfmeans = []
        self.__perfvars = []
        self.__chisquares = []
        self.__chanceprobs = []
        self.__spheresizes = []

        # run searchligh for all radii in the list
        for radius in test_radii:
            if verbose:
                print 'Using searchlight with radius:', radius
            # compute the results
            searchlight( radius, **(kwargs) )

            self.__perfmeans.append( searchlight.perfmean )
            self.__perfvars.append( searchlight.perfvar )
            self.__chisquares.append( searchlight.chisquare )
            self.__chanceprobs.append( searchlight.chanceprob )
            self.__spheresizes.append( searchlight.spheresize )


        # now determine the best classification accuracy
        best = np.array(self.__perfmeans).argmax( axis=0 )

        # select the corresponding values of the best classification
        # in all data tables
        self.perfmean   = best.choose(*(self.__perfmeans))
        self.perfvar    = best.choose(*(self.__perfvars))
        self.chisquare  = best.choose(*(self.__chisquares))
        self.chanceprob = best.choose(*(self.__chanceprobs))
        self.spheresize = best.choose(*(self.__spheresizes))

        # store the best performing radius
        self.bestradius = np.zeros( self.perfmean.shape, dtype='uint' )
        self.bestradius[searchlight.mask==True] = \
            best.choose( test_radii )[searchlight.mask==True]



def makeSphericalROIMask( mask, radius, elementsize=None ):
    """
    """
    # use default elementsize if none is supplied
    if not elementsize:
        elementsize = [ 1 for i in range( len(mask.shape) ) ]
    else:
        if len( elementsize ) != len( mask.shape ):
            raise ValueError, 'elementsize does not match mask dimensions.'

    # rois will be drawn into this mask
    roi_mask = np.zeros( mask.shape, dtype='int32' )

    # while increase with every ROI
    roi_id_counter = 1

    # build spheres around every non-zero value in the mask
    for center, spheremask in \
        algorithms.SpheresInMask( mask,
                                  radius,
                                  elementsize,
                                  forcesphere = True ):

        # set all elements that match the current spheremask to the
        # current ROI index value
        roi_mask[spheremask] = roi_id_counter

        # increase ROI counter
        roi_id_counter += 1

    return roi_mask
