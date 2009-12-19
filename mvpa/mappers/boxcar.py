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
from mvpa.clfs.base import Classifier, accepts_dataset_as_samples

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
        self._outshape = None

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

        # build a list of list where each sublist contains the indexes of to be
        # averaged data elements
        self.__selectors = [ N.arange(i + offset, i + offset + boxlength) \
                             for i in startpoints ]


    @accepts_dataset_as_samples
    def _train(self, data):
        startpoints = self.startpoints
        boxlength = self.boxlength
        if __debug__:
            offset = self.offset
            for sp in startpoints:
                if ( sp + offset + boxlength - 1 > len(data)-1 ) \
                   or ( sp + offset < 0 ):
                    raise ValueError, \
                          'Illegal box: start: %i, offset: %i, length: %i' \
                          % (sp, offset, boxlength)
        self._outshape = (len(startpoints), boxlength) + data.shape[1:]


    def __repr__(self):
        s = super(BoxcarMapper, self).__repr__()
        return s.replace("(", "(boxlength=%d, offset=%d, startpoints=%s, "
                         "collision_resolution='%s'" %
                         (self.boxlength, self.offset, str(self.startpoints),
                          str(self.__collision_resolution)), 1)


    def _forward_data(self, data):
        """Project an ND matrix into N+1D matrix

        This method also handles the special of forward mapping a single 'raw'
        sample. Such a sample is extended (by concatenating clones of itself) to
        cover a full boxcar. This functionality is only availably after a full
        data array has been forward mapped once.

        :Returns:
          array: (#startpoint, ...)
        """
        # if we have a single 'raw' sample (not a boxcar)
        # extend it to cover the full box -- useful if one
        # wants to forward map a mask in raw dataspace (e.g.
        # fMRI ROI or channel map) into an appropriate mask vector
        if self._outshape and data.shape == self._outshape[2:]:
            return N.array([data] * self.boxlength)

        return N.asarray([data[box] for box in self.__selectors])


    def reverse1(self, data):
        if __debug__:
            if not data.shape == self._outshape[1:]:
                raise ValueError("BoxcarMapper has not been train to "
                                 "reverse-map %s-shaped data, but %s."
                                 % (data.shape, self._outshape[1:]))

        # reimplemented since it is really only that
        return data


    def _reverse_data(self, data):
        if not data.shape[1:] == self._outshape[1:]:
            raise ValueError("BoxcarMapper has not been train to "
                             "reverse-map %s-shaped data, but %s."
                             % (data.shape[1:], self._outshape[1:]))
        # stack them all together -- this will cause overlapping boxcars to
        # result in multiple identical samples
        return N.vstack(data)

#        if data.shape == self._outshape:
#            # reconstruct to full input space from the provided data
#            # done below
#            pass
#        elif data.shape == self._outshape[1:]:
#            # single sample was given, simple return it again.
#            # this is done because other mappers also work with 'single'
#            # samples
#            return data
#        else:
#            raise ValueError, "BoxcarMapper operates either on single samples" \
#                  " %s or on the full dataset in 'reverse()' which must have " \
#                  "shape %s. Got data of shape %s" \
#                  % (self._outshape[1:], self._outshape, data.shape)
#
#        # the rest of this method deals with reconstructing the full input
#        # space from the boxcar samples
#        assert(data.shape[0] == len(self.__selectors)) # am I right? :)
#
#        output = N.zeros(self._inshape, dtype=data.dtype)
#        output_counts = N.zeros((self._inshape[0],), dtype=int)
#
#        for i, selector in enumerate(self.__selectors):
#            output[selector, ...] += data[i, ...]
#            output_counts[selector] += 1
#
#        # scale output
#        if self.__collision_resolution == 'mean':
#            # which samples how multiple sources?
#            g1 = output_counts > 1
#            # average them
#            # doing complicated transposing to be able to process array with
#            # nd > 2
#            output_ = output[g1].T
#            output_ /= output_counts[g1]
#            output[g1] = output_.T

        return output
