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
import numpy
import crossval
import sys
import scipy.stats as stats

class Searchlight( object ):
    """
    The object provides a number of properties that can be used to access
    input and output data of the searchlight algorithm.

    The following properties provide access to array that match the size of
    the mask shape. Each element contains information about the sphere that
    is centered on the respective location in the mask:

        perfmean   - mean classification performance
        perfvar    - variance of classification performance
        perfmin    - minimal classification performance of all CV-folds
        perfmax    - maximal classification performance of all CV-folds
        notchance  - two-tailed probability of the peformance being different
                     from chance level ( 1 / <number of classes> ).
        spheresize - number of features in the sphere

    These properties give access to the input data and other status variables:

        pattern     - the MVPAPattern object that holds the data set
        mask        - the mask that is used to generate the spheres
        radius      - the sphere radius in an arbitrary unit (see elementsize
                      property)
        elementsize - a vector specifying the extent of each data element
                      along all dimensions in the dataset. This information is
                      used to translate the radius into element units
        cvtype      - type of cross-validation that is used. 1 means N-1 CV,
                      2 means N-2 ...
        clf         - a class providing a classifier
        clfargs     - dictionary with keyword arguments for the classifiers
                      constructor
        forcesphere - if True a full sphere is used regardless of the status
                      of the status of the sphere elements in the mask. If
                      False only elements are considered as sphere elements
                      that have a non-zero value in the mask.
        ncvfolds    - number of cross-validation folds that are computed by
                      the searchlight algorithm for each sphere
    """
    def __init__( self, pattern, mask,
                  radius = 1.0,
                  elementsize = None,
                  forcesphere = False,
                  cvtype = 1,
                  classifier = None, **kwargs ):
        self.__pattern = pattern
        self.__mask = mask
        self.__radius = radius

        if not elementsize:
            self.__elementsize = [ 1 for i in range( len(pattern.origshape) ) ]
        else:
            if not len( elementsize ) == len( mask.shape ):
                raise ValueError, 'elementsize does not match mask dimensions.'
            self.__elementsize = elementsize

        self.__forcesphere = forcesphere
        self.__cvtype = cvtype
        self.__clf = classifier
        self.__clfargs = kwargs

        if not mask.shape == pattern.origshape:
            raise ValueError, 'Mask shape has to match the pattern origshape.'

        self.__clearResults()


    def __clearResults( self ):
        # init the result maps
        self.__perfmean = numpy.zeros(self.pattern.origshape)
        self.__perfmin = numpy.zeros(self.pattern.origshape)
        self.__perfmax = numpy.zeros(self.pattern.origshape)
        self.__perfvar = numpy.zeros(self.pattern.origshape)
        self.__notchance = numpy.zeros(self.pattern.origshape)
        self.__spheresize = numpy.zeros(self.pattern.origshape, dtype='uint')


    def run( self, verbose=False, classifier = None, **kwargs ):
        """ Perform the spheresearch for all possible spheres in the
        mask.

        By setting 'verbose' to True one can enable some status messages that
        inform about the progress while processing the spheres.

        The 'classifier' argument overrides can be used to select a new
        classification algorithm. Additional keyword are passed to the
        classifier's contructor.
        """
        # accept new classifier if any
        if classifier:
            self.__clf = classifier
            self.__clfargs = kwargs

        if not self.__clf:
            raise RuntimeError, 'No classifier set.'

        # cleanup prior runs first
        self.__clearResults()

        if verbose:
            nspheres = numpy.array( self.mask.nonzero() ).shape[1]
            sphere_count = 0

        # for all possible spheres in the mask
        for center, sphere in \
            algorithms.SpheresInMask( self.__mask,
                                      self.__radius,
                                      self.__elementsize,
                                      self.__forcesphere ):
            # select features inside the sphere
            masked = self.__pattern.selectFeatures( tuple( sphere ) )

            # do the cross-validation
            cv = crossval.CrossValidation( masked,
                                           self.__clf,
                                           **(self.__clfargs) )
            # run cross-validation
            perf = numpy.array( cv.run( cvtype=self.__cvtype ) )

            # translate center coordinate into array slicing index
            center_index = numpy.transpose( 
                               numpy.array( center, ndmin=2 ) ).tolist()

            # store the interesting information
            # mean performance
            self.__perfmean[center_index] = perf.mean()
            # performance variance
            self.__perfvar[center_index] = perf.var()
            # performance minimum
            self.__perfmin[center_index] = perf.min()
            # performance maximum
            self.__perfmax[center_index] = perf.max()
            # significantly different from chance?
            self.__notchance[center_index] = \
                stats.ttest_1samp(perf, 
                                  1.0 / len( self.pattern.reglabels ) )[1]
            # spheresize / number of features
            self.__spheresize[center_index] = sphere.shape[1]

            if verbose:
                sphere_count += 1
                print "\rDoing %i spheres: %i%%" \
                    % (nspheres, float(sphere_count)/nspheres*100,),
                sys.stdout.flush()

        if verbose:
            print ''

    def getNCVFolds( self ):
        """ Returns the number of cross-validation folds that is used by
        the searchlight algorithm.
        """
        return len( crossval.getUniqueLengthNCombinations(
                        self.pattern.originlabels,
                        self.cvtype ) )

    # access to the results
    perfmean = property( fget=lambda self: self.__perfmean )
    perfvar = property( fget=lambda self: self.__perfvar )
    perfmin = property( fget=lambda self: self.__perfmin )
    perfmax = property( fget=lambda self: self.__perfmax )
    notchance = property( fget=lambda self: self.__notchance )
    spheresize = property( fget=lambda self: self.__spheresize )

    # other data access
    pattern = property( fget=lambda self: self.__pattern )
    mask = property( fget=lambda self: self.__mask )
    radius = property( fget=lambda self: self.__radius )
    elementsize = property( fget=lambda self: self.__elementsize )
    cvtype = property( fget=lambda self: self.__cvtype )
    clf = property( fget=lambda self: self.__clf )
    clfargs = property( fget=lambda self: self.__clfargs )
    forcesphere = property( fget=lambda self: self.__forcesphere )
    ncvfolds = property( fget=getNCVFolds )
