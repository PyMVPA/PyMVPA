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

from mvpa.mappers.base import Mapper
from mvpa.clfs.base import accepts_dataset_as_samples

if __debug__:
    from mvpa.base import debug

class BoxcarMapper(Mapper):
    """Mapper to combine multiple samples into a single sample.

    .. note::

      This mapper is somewhat unconventional since it doesn't preserve number
      of samples (ie the size of 0-th dimension).
    """
    # TODO: extend with the possibility to provide real onset vectors and a
    #       samples attribute that is used to determine the actual sample that
    #       is matching a particular onset. The difference between target onset
    #       and sample could be stored as an additional sample attribute. Some
    #       utility functionality (outside BoxcarMapper) could be used to merge
    #       arbitrary sample attributes into the samples matrix (with
    #       appropriate mapper adjustment, e.g. CombinedMapper).
    def __init__(self, startpoints, boxlength, offset=0):
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

        # build a list of list where each sublist contains the indexes of to be
        # averaged data elements
        self.__selectors = [ slice(i + offset, i + offset + boxlength) \
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
        return s.replace("(", "(boxlength=%d, offset=%d, startpoints=%s" %
                         (self.boxlength, self.offset, str(self.startpoints)),
                         1)


    def forward1(self, data):
        # if we have a single 'raw' sample (not a boxcar)
        # extend it to cover the full box -- useful if one
        # wants to forward map a mask in raw dataspace (e.g.
        # fMRI ROI or channel map) into an appropriate mask vector
        if not self._outshape:
            return RuntimeError("BoxcarMapper needs to be trained before "
                                ".forward1() can be used.")
        if not data.shape == self._outshape[2:]:
            return ValueError("Data shape %s does not match sample shape %s."
                              % (data.shape, self._outshape[2:]))

        return N.vstack([data[N.newaxis]] * self.boxlength)


    def _forward_data(self, data):
        """Project an ND matrix into N+1D matrix

        This method also handles the special of forward mapping a single 'raw'
        sample. Such a sample is extended (by concatenating clones of itself) to
        cover a full boxcar. This functionality is only availably after a full
        data array has been forward mapped once.

        :Returns:
          array: (#startpoint, ...)
        """
        return N.vstack([data[box][N.newaxis] for box in self.__selectors])


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
