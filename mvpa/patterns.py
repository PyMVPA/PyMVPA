### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
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
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

import numpy

class MVPAPattern(object):

    def __init__( self, pattern, reg, origin = None ):
        """ Initialize the pattern data.

        The pattern data is finally loaded by calling 
        MVPAPattern.addPattern(). Please see the documentation of this method
        to learn what kind of data is required or supported.
        """
        self.__origshape = None
        self.__patterns = None
        self.__regs = None
        self.__origins = None
        self.__mask = None
        self.__maskedpatterns = None

        self.addPattern( pattern, reg, origin )


    def addPattern( self, pattern, reg, origin = None ):
        """ Adds one or more pattern dataset(s).

        Pattern can be an array or sequence with an arbitrary number of
        dimensions. Internally the pattern data will be converted into a
        2d matrix ( patterns x features ). This data can later be accessed
        by using to 'pattern' property of this class.

        When adding more patterns to an object that already holds soe patterns,
        the new pattern data has to match the shape of the one that are already
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
        pattern = \
            pattern.reshape( len( pattern ), numpy.prod( pattern.shape[1:] ) )

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
                # make sequence of identical value matching the number of patterns
                origin = numpy.repeat( origin, len( pattern ) )

        # simply assign or concatenate if already present
        if self.__origins == None:
            self.__origins = origin
        else:
            self.__origins = numpy.concatenate( (self.__origins, origin ), axis=0 )


    def zscore( self, mean = None, std = None ):
        """ Z-Score the pattern data.

        'mean' and 'std' can be used to pass custom values to the z-scoring.
        Both may be scalars or arrays.

        All computations are done in place. Data upcasting is done
        automatically if necessary.

        Please note, that the whole dataset is z-scored and masking does
        not affect the computation.
        """
        # cast to floating point datatype if necessary
        if str(self.__patterns.dtype).startswith('uint') \
           or str(self.__patterns.dtype).startswith('int'):
            self.__patterns = self.__patterns.astype('float64')

        # calculate mean if necessary
        if not mean:
            mean = self.__patterns.mean(axis=0)

        # calculate std-deviation if necessary
        if not std:
            std = self.__patterns.std(axis=0)

        # do the z-scoring
        self.__patterns -= mean
        self.__patterns /= std


    def getSelectedFeatures( self ):
        """ Returns a list of currently selected feature ids.

        Currently masked features are not listed.

        Please see, the getMaskInOrigShape() method to get the current mask
        in the original data space.
        """
        if self.__mask == None:
            return range( self.nfeatures )
        else:
            return self.__mask


    def removeMask(self):
        """ Disables any present feature mask. """
        setPatternMask(None)


    def setPatternMask( self, mask = None ):
        """ Mask certain features.

        'mask' can either be an NumPy array, a tuple or a list.

        If 'mask' is an array all nonzero array elements are used to select
        features and all other features are masked. The shape of the mask array
        has to match the original data space (see the origshape property).

        If 'mask' is a tuple, it is assumed to be structures like to output of
        Numpy.array.nonzero() ( tuple of sequences listing nonzero element
        coordinates ).

        If 'mask' is a list, it is assumed to be a list of to-be-selected
        feature ids.

        Calling this method without an argument (mask == None) removes any
        possibly present mask and selects all available features.

        After masking the class property 'pattern' will only return the
        selected features.
        """
        # calling with nothing removes the pattern mask
        if mask == None:
            self.__mask = None
            self.__maskedpatterns = None

            return

        if isinstance(mask, numpy.ndarray):
            if not mask.shape == self.origshape:
                raise ValueError, 'Mask shape has to match original data ' \
                              + 'array shape (ignoring the pattern axis).' \

            # tuple of arrays containing the indexes of all nonzero elements
            # of the mask
            self.__mask = [ self.getFeatureId( c ) \
                                for c in numpy.transpose( mask.nonzero() ) ]

        elif isinstance(mask, tuple):
            # mask already contains the nonzero coordinates
            if not len(mask) == len(self.origshape):
                raise ValueError, 'Number of mask dimensions has to match' \
                                + ' the original data array (ignoring the' \
                                + ' pattern axis).'
            self.__mask = [ self.getFeatureId( c ) \
                                for c in numpy.transpose( mask ) ]

        elif isinstance(mask, list):
            # assumed to be a list of feature ids
            self.__mask = mask

        else:
            raise ValueError, "'mask' has to be either an array with" \
                              " origshape or an n-tuple of index arrays" \
                              " (like those returned by array.nonzero())" \
                              " or a list of feature ids."

        # choose all elements with non-zero mask values from all patterns 
        # and convert into a 2d array (patterns x features)
        self.__maskedpatterns = self.__patterns[:, self.__mask]


    def getMaskInOrigShape( self ):
        """ Returns to current mask as an array in the original data shape.

        Currently selected features are set to 'True' all others are 'False'.
        """
        # return full matrix if no mask is there
        if self.__mask == None:
            return numpy.ones( self.origshape, dtype='bool' )

        # initialize empty mask
        origmask = numpy.zeros( self.origshape, dtype='bool' )

        # translate feature ids into coordinates
        coords = [ self.getCoordinate(f) for f in self.getSelectedFeatures() ]

        # set mask to True at feature coordinates
        # A note for the reader: The tuple() is really necessary as without
        # it the whole indexing does not work.
        origmask[ tuple( numpy.transpose(coords) ) ] = True

        return origmask


    def getNumberOfPatterns( self ):
        """ Currently available number of patterns. """
        return len( self.pattern )


    def getNumberOfFeatures( self ):
        """ Number of features per pattern.

        Please note, that the reported number depends on the current feature
        mask!
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
        """ Returns to pattern matrix.

        Please note, that the size of the pattern matrix depends on the
        current feature mask.
        """
        if self.__mask:
            return self.__maskedpatterns
        else:
            return self.__patterns


    # read-only class properties
    pattern =   property( fget=getPatterns )
    reg =       property( fget=lambda self: self.__regs )
    origin =    property( fget=lambda self: self.__origins )
    npatterns = property( fget=getNumberOfPatterns )
    nfeatures = property( fget=getNumberOfFeatures )
    origshape = property( fget=lambda self: self.__origshape )

