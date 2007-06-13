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

class Searchlight( object ):
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
        self.__spheresize = numpy.zeros(self.pattern.origshape, dtype='uint')


    def run( self, verbose=False, classifier = None, **kwargs ):
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
            perf = numpy.array( cv.run( cv=self.__cvtype ) )

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
            # spheresize / number of features
            self.__spheresize[center_index] = sphere.shape[1]

            sphere_count += 1

            if verbose:
                print "\rDoing %i spheres: %i%%" \
                    % (nspheres, float(sphere_count)/nspheres*100,),
                sys.stdout.flush()

        if verbose:
            print ''

    def getNCVFolds( self ):
        return len( crossval.getUniqueLengthNCombinations(
                        self.pattern.originlabels,
                        self.cvtype ) )

    # access to the results
    perfmean = property( fget=lambda self: self.__perfmean )
    perfvar = property( fget=lambda self: self.__perfvar )
    perfmin = property( fget=lambda self: self.__perfmin )
    perfmax = property( fget=lambda self: self.__perfmax )
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
