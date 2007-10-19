### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Mapper using a mask array to map dataspace to featurespace
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

from mapper import Mapper
import numpy as N

class MaskMapper(Mapper):
    def __init__(self, mask):
        """ 'mask' has to be an array in the original dataspace and its nonzero
        elements are used to define the features.
        """
        self.__mask = mask


    def forward(self, data):
        """ Map data from the original dataspace into featurespace.
        """
        maskdim = len( self.__mask.shape )

        if not data.shape[(-1)*maskdim:] == self.__mask.shape:
            raise ValueError, \
                  "To be mapped data does not match the mapper mask."

        if maskdim + 1 < len(data.shape):
            raise ValueError, \
                  "Shape of the to be mapped data, does not match the " \
                  "mapper mask. Only one (optional) additional dimension " \
                  "exceeding the mask shape is supported."

        if maskdim == len(data.shape):
            return data[ self.__mask > 0 ]
        elif maskdim+1 == len(data.shape):
            return data[ :, self.__mask > 0 ]
        else:
            raise RuntimeError, 'This should not happen!'


    def reverse(self, data):
        """ Reverse map data from featurespace into the original dataspace.
        """
        if len(data.shape) > 2 or len(data.shape) < 1:
            raise ValueError, \
                  "Only 2d or 1d data can be reverse mapped."

        if len(data.shape) == 1:
            mapped = N.zeros( self.__mask.shape, dtype=data.dtype )
            mapped[self.__mask>0] = data
        elif len(data.shape) == 2:
            mapped = N.zeros( data.shape[:1] + self.__mask.shape, 
                              dtype=data.dtype )
            mapped[:, self.__mask>0] = data

        return mapped


    def getInShape(self):
        return self.__mask.shape


    def getNMappedFeatures(self):
        return len(self.__mask.nonzero()[0])


    def getMask(self, copy = True):
        """By default returns a copy of the current mask.

        If 'copy' is set to False a reference to the mask is returned instead.
        This shared mask must not be modified!
        """
        if copy:
            return self.__mask.copy()
        else:
            return self.__mask

    # XXX it might become __get_item__ access method
    def getFeatureCoordinate( self, feature_id ):
        """ Returns a features coordinate in the original data space
        for a given feature id.
        """
        return self.getFeatureCoordinates()[feature_id]


    def getFeatureCoordinates( self ):
        """ Returns a 2d array where each row contains the coordinate of the
        feature with the corresponding id.
        """
        return N.transpose(self.__mask.nonzero())


    def getFeatureId( self, coord ):
        """ Translate a feature mask coordinate into a feature ID.

        Warning: This method is painfully slow, avoid if possible!
        """
        coord = list(coord)

        featcoords = N.transpose(self.__mask.nonzero()).tolist()

        for i, c in enumerate( featcoords ):
            if c == coord:
                return i

        raise ValueError, "There is no used feature at this mask coordinate."


    def buildMaskFromFeatureIds( self, ids ):
        """ Returns a mask with all features in ids selected from the
        current feature set.
        """
        fmask = N.repeat(False, self.nfeatures )
        fmask[ids] = True
        return self.reverse( fmask )


    # Read-only props
    dsshape = property( fget=getDataspaceShape )
    nfeatures = property( fget=getNMappedFeatures )

