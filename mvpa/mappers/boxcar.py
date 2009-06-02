# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Data mapper"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.base.dochelpers import enhancedDocString
from mvpa.mappers.base import Mapper
from mvpa.misc.support import isInVolume

if __debug__:
    from mvpa.base import debug

class BoxcarMapper(Mapper):
    """Mapper to combine multiple samples into a single sample.

    .. note::

      This mapper is somewhat unconventional since it doesn't preserve number
      of samples (ie the size of 0-th dimension).
    """

    _COLLISION_RESOLUTIONS = ['mean']

    def __init__(self, startpoints, boxlength, offset=0,
                 collision_resolution='mean'):
        """
        :Parameters:
          startpoints: sequence
            Index values along the first axis of 'data'.
          boxlength: int
            The number of elements after 'startpoint' along the first axis of
            'data' to be considered for the boxcar.
          offset: int
            The offset between the provided starting point and the actual start
            of the boxcar.
          collision_resolution : 'mean'
            if a sample belonged to multiple output samples, then on reverse,
            how to resolve the value
        """
        Mapper.__init__(self)

        startpoints = N.asanyarray(startpoints)
        if N.issubdtype(startpoints.dtype, 'i'):
            self.startpoints = startpoints
        else:
            if __debug__:
                debug('MAP', "Boxcar: obtained startpoints are not of int type."
                      " Rounding and changing dtype")
            self.startpoints = N.asanyarray(N.round(startpoints), dtype='i')

        # Sanity checks
        if boxlength < 1:
            raise ValueError, "Boxlength lower than 1 makes no sense."
        if boxlength - int(boxlength) != 0:
            raise ValueError, "boxlength must be an integer value."

        self.boxlength = int(boxlength)
        self.offset = offset
        self.__selectors = None

        if not collision_resolution in self._COLLISION_RESOLUTIONS:
            raise ValueError, "Unknown method to resolve the collision." \
                  " Valid are %s" % self._COLLISION_RESOLUTIONS
        self.__collision_resolution = collision_resolution


    __doc__ = enhancedDocString('BoxcarMapper', locals(), Mapper)


    def __repr__(self):
        s = super(BoxcarMapper, self).__repr__()
        return s.replace("(", "(boxlength=%d, offset=%d, startpoints=%s, "
                         "collision_resolution='%s'" %
                         (self.boxlength, self.offset, str(self.startpoints),
                          str(self.__collision_resolution)), 1)


    def forward(self, data):
        """Project an ND matrix into N+1D matrix

        This method also handles the special of forward mapping a single 'raw'
        sample. Such a sample is extended (by concatenating clones of itself) to
        cover a full boxcar. This functionality is only availably after a full
        data array has been forward mapped once.

        :Returns:
          array: (#startpoint, ...)
        """
        # in case the mapper is already charged
        if not self.__selectors is None:
            # if we have a single 'raw' sample (not a boxcar)
            # extend it to cover the full box -- useful if one
            # wants to forward map a mask in raw dataspace (e.g.
            # fMRI ROI or channel map) into an appropriate mask vector
            if data.shape == self._outshape[2:]:
                return N.asarray([data] * self.boxlength)

        self._inshape = data.shape

        startpoints = self.startpoints
        offset = self.offset
        boxlength = self.boxlength

        # check for illegal boxes
        for sp in self.startpoints:
            if ( sp + offset + boxlength - 1 > len(data)-1 ) \
               or ( sp + offset < 0 ):
                raise ValueError, \
                      'Illegal box: start: %i, offset: %i, length: %i' \
                      % (sp, offset, boxlength)

        # build a list of list where each sublist contains the indexes of to be
        # averaged data elements
        self.__selectors = [ N.arange(i + offset, i + offset + boxlength) \
                             for i in startpoints ]
        selected = N.asarray([ data[ box ] for box in self.__selectors ])
        self._outshape = selected.shape

        return selected


    def reverse(self, data):
        """Uncombine features back into original space.

        Samples which were not touched by forward will get value 0 assigned
        """
        if data.shape == self._outshape:
            # reconstruct to full input space from the provided data
            # done below
            pass
        elif data.shape == self._outshape[1:]:
            # single sample was given, simple return it again.
            # this is done because other mappers also work with 'single'
            # samples
            return data
        else:
            raise ValueError, "BoxcarMapper operates either on single samples" \
                  " %s or on the full dataset in 'reverse()' which must have " \
                  "shape %s. Got data of shape %s" \
                  % (self._outshape[1:], self._outshape, data.shape)

        # the rest of this method deals with reconstructing the full input
        # space from the boxcar samples
        assert(data.shape[0] == len(self.__selectors)) # am I right? :)

        output = N.zeros(self._inshape, dtype=data.dtype)
        output_counts = N.zeros((self._inshape[0],), dtype=int)

        for i, selector in enumerate(self.__selectors):
            output[selector, ...] += data[i, ...]
            output_counts[selector] += 1

        # scale output
        if self.__collision_resolution == 'mean':
            # which samples how multiple sources?
            g1 = output_counts > 1
            # average them
            # doing complicated transposing to be able to process array with
            # nd > 2
            output_ = output[g1].T
            output_ /= output_counts[g1]
            output[g1] = output_.T

        return output


    def getInSize(self):
        """Returns the number of original samples which were combined.
        """

        return self._inshape[0]

    def isValidOutId(self, outId):
        """Validate if OutId is valid

        """
        try:
            return isInVolume(outId, self._outshape[1:])
        except:
            return False

    def isValidInId(self, inId):
        """Validate if InId is valid

        """
        try:
            return isInVolume(inId, self._inshape[1:])
        except:
            return False


    def getOutSize(self):
        """Returns the number of output samples.
        """

        return N.prod(self._outshape[1:])


    def selectOut(self, outIds):
        """Just complain for now"""
        raise NotImplementedError, \
            "For feature selection use MaskMapper on output of the %s mapper" \
            % self.__class__.__name__



