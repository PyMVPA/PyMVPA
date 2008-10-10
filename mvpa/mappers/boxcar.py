#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
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

    This mapper is somewhat unconventional since it doesn't preserve
    number of samples (ie the size of 0-th dimension...
    """

    _COLLISION_RESOLUTIONS = ['mean']

    def __init__(self, startpoints, boxlength, offset=0,
                 collision_resolution='mean'):
        """Initialize the BoxcarMapper

        Parameters:
          startpoints:    A sequence of index value along the first axis of
                          'data'.
          boxlength:      The number of elements after 'startpoint' along the
                          first axis of 'data' to be considered for averaging.
          offset:         The offset between the starting point and the
                          averaging window (boxcar).
          collision_resolution : string
            if a sample belonged to multiple output samples, then on reverse,
            how to resolve the value (choices: 'mean')
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

        if boxlength < 1:
            raise ValueError, "Boxlength lower than 1 makes no sense."

        self.boxlength = boxlength
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

        :Returns:
          array: (#startpoint, ...)
        """

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

        # XXX  check if use of slicing gives any benefits and if it works at all
        #self.__selectors = [ slice( i + offset, i + offset + boxlength) \
        self.__selectors = [ N.arange(i + offset, i + offset + boxlength) \
                             for i in startpoints ]

        # XXX average each box
        #selected = [ fx( data[ N.array(box) ], axis=0 ) for box in selector ]
        selected = N.asarray([ data[ box ] for box in self.__selectors ])
        self._outshape = selected.shape

        return selected


    def reverse(self, data):
        """Uncombine features back into original space.

        Samples which were not touched by forward will get value 0 assigned
        """
        if data.shape != self._outshape:
            raise ValueError, "BoxcarMapper operates on full dataset in " \
                  "'reverse()' which must have shape %s" % self._outshape

        assert(data.shape[0] == len(self.__selectors)) # am I right? :)

        output = N.zeros(self._inshape, dtype=data.dtype)
        output_counts = N.zeros((self._inshape[0],),dtype=int)

        for i, selector in enumerate(self.__selectors):
            output[selector,...] += data[i, ...]
            output_counts[selector] += 1

        # scale output
        if self.__collision_resolution == 'mean':
            g1 = output_counts>1
            # XXX couldn't broadcast if we have multiple columns...
            # need to operate on transposed output, so that last running
            # dimensions are the same
            output_ = output.T
            output_[:,g1] /= output_counts[g1]
            output = output_.T

        return output

    def getInShape(self):
        """Returns a shape of original sample.
        """
        return self._inshape[1:]


    def getOutShape(self):
        """Returns a shape of combined sample.
        """
        return self._outshape[1:]


    def getInSize(self):
        """Returns the number of original samples which were combined.
        """

        return self._inshape[0]

    def isValidOutId(self, outId):
        """Validate if OutId is valid

        """
        try:
            return isInVolume(outId, self.getOutShape())
        except:
            return False

    def isValidInId(self, inId):
        """Validate if InId is valid

        """
        try:
            return isInVolume(inId, self.getInShape())
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



