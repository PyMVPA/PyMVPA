#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the MIT License.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the COPYING
#    file that comes with this package for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Mapper using a mask array to map dataspace to featurespace"""

from mapper import Mapper
import numpy as N

class MaskMapper(Mapper):
    """Mapper which uses a binary mask to select "Features" """

    def __init__(self, mask):
        """ 'mask' has to be an array in the original dataspace and its nonzero
        elements are used to define the features.
        """
        Mapper.__init__(self)
        self.__mask = self.__maskdim = self.__masksize = \
                      self.__masknonzerosize = self.__forwardmap = \
                      self.__masknonzero = None # to make pylint happy
        self._initMask(mask)

    def _initMask(self, mask):
        """Initialize internal state with mask-derived information

        It is needed to initialize structures for the fast
        and reverse lookup to don't impose performance hit on any
        future operation
        """
        self.__mask = mask
        self.__maskdim = len(mask.shape)
        self.__masksize = N.prod(mask.shape)

        # Following introduces space penalty but are needed
        # for efficient processing.
        # Store all coordinates for backward mapping
        self.__masknonzero = mask.nonzero()
        self.__masknonzerosize = len(self.__masknonzero[0])

        # Store forward mapping (ie from coord into outId)
        # TODO to save space might take appropriate int type
        #     depending on masknonzerosize
        # it could be done with a dictionary, but since mask
        # might be relatively big, it is better to simply use
        # a chunk of RAM ;-)
        self.__forwardmap = N.zeros(mask.shape, dtype=N.uint64)
        # under assumption that we +1 values in forwardmap so that
        # 0 can be used to signal outside of mask
        for voxelIndex in xrange(self.__masknonzerosize):
            coordIn = self.getInId(voxelIndex)
            self.__forwardmap[tuple(coordIn)] = voxelIndex + 1

    def forward(self, data):
        """ Map data from the original dataspace into featurespace.
        """
        datadim = len(data.shape)
        if not data.shape[(-1)*self.__maskdim:] == self.__mask.shape:
            raise ValueError, \
                  "To be mapped data does not match the mapper mask."

        # XXX yoh Q: can't we mask 3D (no samples) into 2D with a single sample?
        if self.__maskdim + 1 < len(data.shape):
            raise ValueError, \
                  "Shape of the to be mapped data, does not match the " \
                  "mapper mask. Only one (optional) additional dimension " \
                  "exceeding the mask shape is supported."
        # XXX: this condition is masked out by the previous check... so is
        # previous needed?
        if self.__maskdim == datadim:
            return data[ self.__mask > 0 ]
        elif self.__maskdim+1 == datadim:
            # XXX !!! yoh -- changed >0 to != 0 especially since nonzero()
            #         so negative values also get considered
            return data[ :, self.__mask != 0 ]
        else:
            raise RuntimeError, 'This should not happen!'

    def reverse(self, data):
        """ Reverse map data from featurespace into the original dataspace.
        """
        datadim = len(data.shape)
        if not datadim in [1, 2]:
            raise ValueError, \
                  "Only 2d or 1d data can be reverse mapped."

        if datadim == 1:
            mapped = N.zeros(self.__mask.shape, dtype=data.dtype)
            mapped[self.__mask != 0] = data
        elif datadim == 2:
            mapped = N.zeros(data.shape[:1] + self.__mask.shape,
                             dtype=data.dtype)
            mapped[:, self.__mask != 0] = data

        return mapped

    def getInShape(self):
        """InShape is a shape of original mask"""
        return self.__mask.shape

    def getInSize(self):
        """InShape is a shape of original mask"""
        return self.__masksize

    def getOutShape(self):
        """OutShape is a shape of target dataset"""
        # should worry about state-full class.
        # TODO: add exception 'InvalidStateError' which is raised
        #       by some class if state is not yet defined:
        #         classifier has not yet been trained
        #         mapped yet see the dataset
        raise NotImplementedError

    def getOutSize(self):
        """OutSize is a number of non-0 elements in the mask"""
        return self.__masknonzerosize

    def getMask(self, copy = True):
        """By default returns a copy of the current mask.

        If 'copy' is set to False a reference to the mask is returned instead.
        This shared mask must not be modified!
        """
        if copy:
            return self.__mask.copy()
        else:
            return self.__mask

    def getInId(self, outId):
        """ Returns a features coordinate in the original data space
        for a given feature id.

        XXX it might become __get_item__ access method

        """
        # XXX Might be improved by storing also transpose of
        # __masknonzero
        return N.array([self.__masknonzero[i][outId]
                        for i in xrange(self.__maskdim)])

    def getInIds(self):
        """ Returns a 2d array where each row contains the coordinate of the
        feature with the corresponding id.
        """
        return N.transpose(self.__masknonzero)

    def getOutId(self, coord):
        """ Translate a feature mask coordinate into a feature ID.
        """
        try:
            outId = self.__forwardmap[tuple(coord)]
        except TypeError:
            raise ValueError, \
                  "Coordinates %s are of incorrect dimension. " % `coord` + \
                  "The mask has %d dimensions." % self.__maskdim
        except IndexError:
            raise ValueError, \
                  "Coordinates %s are out of mask boundary. " % `coord` + \
                  "The mask is of %s shape." % `self.__mask.shape`

        if not outId:
            raise ValueError, \
                  "The point %s didn't belong to the mask" % (`coord`)
        else:
            return outId-1


    def buildMaskFromFeatureIds(self, ids):
        """ Returns a mask with all features in ids selected from the
        current feature set.
        """
        fmask = N.repeat(False, self.nfeatures)
        fmask[ids] = True
        return self.reverse(fmask)

    # Read-only props
    # TODO: refactor the property names? make them vproperty?
    dsshape = property(fget=getInShape)
    nfeatures = property(fget=getOutSize)


