### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Pattern handling and manipulation
#
#    Copyright (C) 2006-2007 by
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

import numpy
import operator
import random

class MVPAPattern(object):
    """ This class provides a container to store all necessary data to perform
    MVPA analyses. That is the actual pattern data, as well as the regressors
    associated with these patterns. Additionally the origin of each pattern is
    stored to be able to group patterns for cross-validation purposes.
    """

    def __init__( self, pattern, reg, origin = None, **kwargs ):
        """ Initialize the pattern data.

        The pattern data is finally loaded by calling
        MVPAPattern.addPattern(). Please see the documentation of this method
        to learn what kind of data is required or supported.

        Please ignore the additional keyword arguments. They are only necessary
        for internal stuff. You cannot trust them -- do not use them!
        """
        # initialize containers
        self.__origshape = None
        self.__patterns = None
        self.__regs = None
        self.__origins = None

        self.__origregs = None

        # when in internal mode bypass all checks and directly assign
        # the data. Otherwise call addPattern() that checks if the data
        # is in good shape
        if kwargs.has_key('internal'):
            self.__initDataNoChecks( pattern, reg, origin, **(kwargs) )
        else:
            self.addPattern( pattern, reg, origin )


    def __iadd__( self, other ):
        """ Merge the patterns of one MVPAPattern object to another (in-place).

        Please note that the patterns, regressors and origins are simply
        concatenated to create an MVPAPattern object that contains the
        patterns of both objects. No further processing is done. In particular
        the origin values are not modified: Patterns with the same origin from
        both MVPAPattern object will still share the same origin and will
        therefore be treated as they would belong together by the
        CrossValidation algorithm.
        """
        if self.origshape != other.origshape:
            raise ValueError, "Cannot add MVPAPattern, because origshapes " \
                              "do not match."

        self.__patterns = \
            numpy.concatenate( ( self.__patterns, other.__patterns ), axis=0)
        self.__regs = \
            numpy.concatenate( ( self.__regs, other.__regs ), axis=0)
        self.__origins = \
            numpy.concatenate( ( self.__origins, other.__origins ), axis=0)

        return self


    def __add__( self, other ):
        """ Merge the patterns of two MVPAPattern objects.

        Please note that the patterns, regressors and origins are simply
        concatenated to create an MVPAPattern object that contains the
        patterns of both objects. No further processing is done. In particular
        the origin values are not modified: Patterns with the same origin from
        both MVPAPattern object will still share the same origin and will
        therefore be treated as they would belong together by the
        CrossValidation algorithm.
        """
        out = MVPAPattern( self.__patterns,
                           self.__regs,
                           self.__origins,
                           internal = True,
                           origshape = self.origshape )
        out += other

        return out


    def addPattern( self, pattern, reg, origin = None ):
        """ Adds one or more pattern dataset(s).

        Pattern can be an array or sequence with an arbitrary number of
        dimensions. Internally the pattern data will be converted into a
        2d matrix ( patterns x features ). This data can later be accessed
        by using to 'pattern' property of this class.

        When adding more patterns to an object that already holds some patterns,
        the new pattern data has to match the shape of the ones that are already
        loaded. An example: one first loads pattern data with shape (10,2,3,4).
        This means 10 patterns shaped (2,3,4). Any pattern that shall be loaded
        later has to be shaped <number of patterns> x (2,3,4).

        A 1d array or simple sequence passed as 'pattern' arguments is assumed
        to be a single pattern ( 1 x <number of features> ).

        'reg' can be a single scalar or a sequence of value. These regressors
        represent the target value for the classification, e.g. class labels.
        If 'reg' is a scalar this value is used as regressor for all patterns
        in 'pattern'. If 'reg' is a sequence its length has to match the number
        of patterns and the values in 'reg' are associated with the
        corresponding patterns.

        The 'origin' parameter is used to associate patterns with each other.
        This can be useful to exclude/include patterns set in a
        cross-validation run. 'origin' is treated similar to the 'reg'
        argument: scalars and sequences are supported. Additionally 'origin'
        can also be 'None'. In this case a unique origin value is automatically
        calculated for each pattern individually (this also takes already
        present patterns into account).
        """
        # 1d arrays or simple sequences are assumed to be a single pattern
        if (not isinstance(pattern, numpy.ndarray)) or pattern.ndim < 2:
            pattern = numpy.array( pattern, ndmin=2 )

        # store the shape of a single pattern for later coordinate
        # reconstruction
        if self.__origshape == None:
            self.__origshape = pattern.shape[1:]
        else:
            if self.__origshape != pattern.shape[1:]:
                raise ValueError, "Pattern shape does not match existing" \
                                  " patterns (exist: %s, new: %s)" \
                                  % ( str(self.__origshape),
                                      str(pattern.shape[1:]) )

        # now reshape into a 2d array
        if not pattern.ndim == 2:
            pattern = pattern.reshape( len( pattern ),
                                       numpy.prod( pattern.shape[1:] ) )

        # simply assign or concatenate if already present
        if self.__patterns == None:
            self.__patterns = pattern
        else:
            self.__patterns = numpy.concatenate( (self.__patterns, pattern),
                                                axis=0 )

        # check if regs is supplied as a sequence
        try:
            if len( reg ) != len( pattern ):
                raise ValueError, "Length of 'reg' has to match the number" \
                                  " of patterns."
            # store the sequence as array
            reg = numpy.array( reg )

        except TypeError:
            # make sequence of identical value matching the number of patterns
            reg = numpy.repeat( reg, len( pattern ) )

        # simply assign or concatenate if already present
        if self.__regs == None:
            self.__regs = reg
        else:
            self.__regs = numpy.concatenate( (self.__regs, reg), axis=0 )

        # if no origin is given assume that every pattern has its own
        if origin == None:
            origin = numpy.arange( len( pattern ) )
            if self.__origins != None:
                origin += self.__origins.max() + 1
        else:
            try:
                if len( origin ) != len( pattern ):
                    raise ValueError, "Length of 'origin' has to match the" \
                                      " number of patterns."
                # store the sequence as array
                origin = numpy.array( origin )

            except TypeError:
                # make sequence of identical value matching the number of
                # patterns
                origin = numpy.repeat( origin, len( pattern ) )

        # simply assign or concatenate if already present
        if self.__origins == None:
            self.__origins = origin
        else:
            self.__origins = numpy.concatenate( (self.__origins, origin ),
                                                axis=0 )


    def __initDataNoChecks(self, pat, reg, orig, **kwargs):
        """ Do not use! """
        self.__origshape = kwargs['origshape']
        self.__patterns = pat
        self.__regs = reg
        self.__origins = orig


    def permutatedRegressors( self, status ):
        """ Permutate the regressors.

        Calling this method with 'status' set to True, the regressors are
        permutated among all patterns sharing the same origin value. Therefore
        only the association of a certain pattern with a regressor is permutated
        while keeping the absolute number of occurences of each regressor value
        with a certain origin constant.

        If 'status' is False the original regressors are restored.
        """

        if not status:
            # restore originals
            if self.__origregs == None:
                raise RuntimeError, 'Cannot restore regressors. ' \
                                    'randomizedRegressors() has never been ' \
                                    'called with status == True.'
            self.__regs = self.__origregs
            self.__origregs = None
        else:
            # permutate regs per origin

            # make a backup of the original regressors
            self.__origregs = self.__regs.copy()

            # now scramble the rest
            for o in self.originlabels:
                self.__regs[self.__origins == o ] = \
                    numpy.random.permutation( self.__regs[ self.__origins == o ] )


    def zscore( self, mean = None, std = None, origin=True ):
        """ Z-Score the pattern data.

        'mean' and 'std' can be used to pass custom values to the z-scoring.
        Both may be scalars or arrays.

        All computations are done in place. Data upcasting is done
        automatically if necessary.

        If origin is True patterns with the same origin are z-scored independent
        of patterns with other origin values, e.i. mean and standard deviation
        are calculated individually.
        """
        # cast to floating point datatype if necessary
        if str(self.__patterns.dtype).startswith('uint') \
           or str(self.__patterns.dtype).startswith('int'):
            self.__patterns = self.__patterns.astype('float64')

        def doit(pat, mean, std):
            # calculate mean if necessary
            if not mean:
                mean = pat.mean(axis=0)

            # calculate std-deviation if necessary
            if not std:
                std = pat.std(axis=0)

            # do the z-scoring
            pat -= mean
            pat /= std

            return pat

        if origin:
            for o in self.originlabels:
                self.__patterns[self.__origins == o] = \
                    doit( self.__patterns[self.__origins == o], mean, std )
        else:
            doit( self.__patterns, mean, std )


    def getPatternSample( self, nperreg ):
        """ Select a random sample of patterns.

        If 'nperreg' is an integer value, the specified number of patterns is
        randomly choosen from the group of patterns sharing a unique regressor
        value ( total number of selected patterns: nperreg x len(reglabels).

        If 'nperreg' is a list its length has to match the number of unique
        regressor labels. In this case 'nperreg' specifies the number of
        patters that shall be selected from the patterns with the corresponding
        regressor label.

        The method returns a MVPAPattern object containing the selected
        patterns.
        """
        # if interger is given take this value for all classes
        if isinstance(nperreg, int):
            nperreg = [ nperreg for i in self.reglabels ]

        sample = []
        # for each available class
        for i,r in enumerate(self.reglabels):
            # get the list of pattern ids for this class
            sample += random.sample( (self.reg == r).nonzero()[0],
                                     nperreg[i] )

        return self.selectPatterns( sample )


    def selectPatterns( self, mask ):
        """ Choose a subset of patterns.

        Returns a new MVPAPattern object containing the selected pattern
        subset.
        """
        # without having a sequence a index the masked pattern array would
        # loose its 2d layout
        if not operator.isSequenceType( mask ):
            mask = [mask]

        return MVPAPattern( self.__patterns[mask,],
                            self.__regs[mask,],
                            self.__origins[mask,],
                            internal=True,
                            origshape=self.origshape )


    def selectFeatures( self, mask ):
        """ Choose a subset of features.

        'mask' can either be a NumPy array, a tuple or a list.

        If 'mask' is an array all nonzero array elements are used to select
        features. The shape of the mask array has to match the original data
        space (see the origshape property).

        If 'mask' is a tuple, it is assumed to be structured like to output of
        Numpy.array.nonzero() ( tuple of sequences listing nonzero element
        coordinates ).

        If 'mask' is a list, it is assumed to be a list of to-be-selected
        feature ids.

        Returns a new MVPAPattern object with the selected features.
        """
        if isinstance(mask, numpy.ndarray):
            if not mask.shape == self.origshape:
                raise ValueError, 'Mask shape has to match original data ' \
                              + 'array shape (ignoring the pattern axis).' \

            # tuple of arrays containing the indexes of all nonzero elements
            # of the mask
            featuremask = [ self.getFeatureId( c ) \
                          for c in numpy.transpose( mask.nonzero() ) ]

        elif isinstance(mask, tuple):
            # mask already contains the nonzero coordinates
            if not len(mask) == len(self.origshape):
                raise ValueError, 'Number of mask dimensions has to match' \
                                + ' the original data array (ignoring the' \
                                + ' pattern axis).'
            featuremask = [ self.getFeatureId( c ) \
                          for c in numpy.transpose( mask ) ]

        elif isinstance(mask, list):
            # assumed to be a list of feature ids
            featuremask = mask

        else:
            raise ValueError, "'mask' has to be either an array with" \
                              " origshape or an n-tuple of index arrays" \
                              " (like those returned by array.nonzero())" \
                              " or a list of feature ids."

        return MVPAPattern( self.__patterns[:, featuremask],
                            self.__regs,
                            self.__origins,
                            internal = True,
                            origshape = self.origshape )


    def features2origmask( self, features ):
        """ Transforms a sequence of feature ids into a boolean mask in the
        original data shape (see the origshape property).

        Selected features are set to 'True' all others are 'False'.
        """
        # initialize empty mask
        origmask = numpy.zeros( self.origshape, dtype='bool' )

        # translate feature ids into coordinates
        coords = [ self.getCoordinate(f) for f in features ]

        # set mask to True at feature coordinates
        # A note for the reader: The tuple() is really necessary as without
        # it the whole indexing does not work.
        origmask[ tuple( numpy.transpose(coords) ) ] = True

        return origmask


    def getNumberOfPatterns( self ):
        """ Currently available number of patterns.
        """
        return self.pattern.shape[0]


    def getNumberOfFeatures( self ):
        """ Number of features per pattern.
        """
        return self.pattern.shape[1]


    def getFeatureId( self, coord ):
        """ Calculates the feature id from a coordinate value in the original
        data space.
        """
        # transform shape and coordinate into array for easy handling
        ac = numpy.array( coord )
        ao = numpy.array( self.origshape )

        # check for sane coordinates
        if (ac >= ao).all() \
           or (ac < numpy.repeat( 0, len( ac ) ) ).all():
            raise ValueError, 'Invalid coordinate: outside array ' \
                              '( coord: %s, arrayshape: %s )' % \
                              ( str(coord), str(self.origshape) )

        # this will hold the feature number
        f_id = 0

        # for all axes
        for d in range(len(ao)):
            f_id += ac[d] * ao[d+1:].prod()

        return f_id


    def getCoordinate( self, feature_id ):
        """ Computes the feature coordinates in the original data space
        for a given feature id.
        """
        # transform shape and coordinate into array for easy handling
        ao = numpy.array( self.origshape )

        # check for sane feature id
        if feature_id < 0 or feature_id >= ao.prod():
            raise ValueError, 'Invalid feature id (recieved: %i)' % feature_id

        coord = []

        # for all axes, except the last
        for d in range(1, len(ao) ):
            # offset on current axis
            submatrix_size = ao[d:].prod()
            axis_coord = feature_id / submatrix_size
            # substract what we already have
            feature_id -= axis_coord * submatrix_size
            # store
            coord.append( axis_coord )
        # store the offset on the last axis
        coord.append( feature_id % ao[-1] )

        return coord


    def getPatterns( self ):
        """ Returns the pattern matrix.

        Please note, that the size of the pattern matrix depends on the
        current feature and pattern mask.
        """
        return self.__patterns


    def getRegs( self ):
        """ Returns the regressors vector.

        Please note, that the size of the vector depends on the current
        pattern mask.
        """
        return self.__regs


    def getOrigins( self ):
        """ Returns the origin vector.

        Please note, that the size of the vector depends on the current
        pattern mask.
        """
        return self.__origins


    def getRegLabels( self ):
        """ Returns an array with all unique class labels in the regressors
        vector.
        """

        return numpy.unique( self.reg )


    def getOriginLabels( self ):
        """ Returns an array with all unique labels in the origin vector.
        """

        return numpy.unique( self.origin )


    def getPatternsPerRegLabel( self ):
        """ Returns the number of patterns per regressor label.
        """
        return [ len(self.__patterns[self.__regs == r]) \
                    for r in self.reglabels ]


    # read-only class properties
    pattern =   property( fget=getPatterns )
    reg =       property( fget=getRegs )
    origin =    property( fget=getOrigins )
    npatterns = property( fget=getNumberOfPatterns )
    nfeatures = property( fget=getNumberOfFeatures )
    origshape = property( fget=lambda self: self.__origshape )
    reglabels = property( fget=getRegLabels )
    originlabels = property( fget=getOriginLabels )
    patperreg = property( fget=getPatternsPerRegLabel )

