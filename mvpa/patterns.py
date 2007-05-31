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
import random

class Patterns(object):
    def __init__( self, pattern = [], regs = [], origin = None ):
        self.__patterns = [ i for i in pattern ]
        self.__regs = [ i for i in regs ]

        if origin == None:
            origin = [ i for i in xrange( len( pattern ) ) ]

        self.__origins = [ i for i in origin ]

        if not ( len( self.__patterns ) == len( self.__regs ) == len( self.__origins ) ):
            raise ValueError, "All sequences (pattern, regs (, origin)) have to be of equal length."


    def addPatterns( self, pattern, reg, origin ):
        """ Add patterns with a common regressor (reg) and origin.

        'pattern' has to be a sequence.
        """
        regs = [ reg for i in xrange(len(pattern)) ]
        origins = [ origin for i in xrange(len(pattern)) ]

        # do list comprehension to be able to add other sequence types
        # (or even numpy arrays) to the pattern list
        self.__patterns += [ p for p in pattern ]
        self.__regs += regs
        self.__origins += origins


    def addPackedPatterns( self, pack ):
        """
        """
        pat = [ p[0] for p in pack ]
        reg = [ p[1] for p in pack ]
        orig = [ p[2] for p in pack ]

        self.__patterns += pat
        self.__regs += reg
        self.__origins += orig


    def getPacked( self ):
        """ Pack information about pattern data, regressor value and pattern origin
        into a single datastructure.

        This function can be useful if a data pattern has to be associated with
        some properties (regressor value and origin of the pattern). This might be
        necessary if a function has to be applied to a list of patterns without
        loosing the association.

        While the regressor associates patterns with certain conditions the origins
        can be used to mark patterns to be structural similiar e.g. recorded during
        the same session. If no origin value is specified each pattern will get a
        unique origin label.

        This function returns a sequence of 3-tupels (pattern, reg, origin).

        Please see the unpackPatterns() that reverts this procedure.
        """
        return zip( self.pattern, self.reg, self.origin )


    def clear( self ):
        self.__patterns = []
        self.__regs = []
        self.__origins = []


    def shuffle( self ):
        # get the packed data
        pack = self.getPacked()

        # clear the data
        self.clear()

        # shuffle the data
        random.shuffle( pack )

        # and put it back
        self.addPackedPatterns( pack )


    def zscore( self, mean = None, std = None ):
        """ Z-Score the pattern data.

        'mean' and 'std' can be used to pass custom values to the z-scoring. Both
        may be scalars or arrays.
        """
        data = self.asarray() 
        
        # calculate mean if necessary
        if not mean:
            mean = data.mean()

        # calculate std-deviation if necessary
        if not std:
            std = data.std()

        # do the z-scoring (do not use in-place operations to ensure
        # appropriate data upcasting
        zscore = ( data - mean ) / std

        # store the zscored data
        self.__patterns = [ p for p in zscore ]


    def asarray( self ):
        """ Returns the pattern data as a NumPy array.
        """
        return numpy.array( self.pattern )


    def selectFeatures(self, mask = None):
        """ Uses all non-zero elements of a mask volume to select
        elements in data array.

        Returns a 2d array ( patterns x <number of non-zeros in mask> ).
        """
        # if there is nothing return nothing
        if not len( self.pattern ):
            return None

        # convert data into an array
        # this might be stupid as the data is finally transformed back into a list
        # but it also makes sure that all patterns have a uniform shape
        data = self.asarray()

        # make sure to always have at least 2d data
        # necessary because each pattern has to be an array as well otherwise
        # one cannot use the nonzero coordinates to slice the data
        if len( data.shape ) < 2:
            data = data.reshape( data.shape + (1,) )

        # use everything if there is no mask
        if mask == None:
            mask = numpy.ones(data.shape[1:])

        if isinstance(mask, numpy.ndarray):
            if not mask.shape == data.shape[1:]:
                raise ValueError, 'Mask shape has to match data array shape' \
                                + ' while ignoring 1st dimension, e,g. if data' \
                                + ' is (10,2,3,4) mask has to be (2,3,4).'

            # tuple of arrays containing the indexes of all nonzero elements
            # of the mask
            nz = mask.nonzero()

        elif isinstance(mask, tuple):
            # mask already contains the nonzero coordinates
            if not len(mask) == len(data.shape[1:]):
                raise ValueError, 'Number of mask dimensions has to match the' \
                                + ' data array (except 1st data array dimension.'
            nz = mask

        else:
            raise ValueError, "'mask' has to be either an array with one" \
                            + " dimension less than the data array or an" \
                            + " n-tuple of index arrays (like those returned" \
                            + " by array.nonzero())"

        # choose all elements with non-zero mask values from all patterns 
        # and convert into a 2d array (patterns x features)
        selected = numpy.array( [ p[nz] for p in data ] )

        return selected


   # read-only class properties
    packed =  property( fget=getPacked)
    pattern = property( fget=lambda self: self.__patterns )
    reg =     property( fget=lambda self: self.__regs )
    origin =  property( fget=lambda self: self.__origins )


def feature2coord( nfeat, mask ):
    """ Converts the feature id (number) into a coordinate in a given mask.
    """

    # get the non-zero mask elements
    nz = mask.nonzero()

    return tuple( [ nz[i][nfeat] for i in range(len(nz)) ] )


def samplePatterns( pack, n, exclude_orig = None ):
    """ 
    """
    pass
