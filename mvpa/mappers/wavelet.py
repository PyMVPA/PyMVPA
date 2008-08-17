#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wavelet mappers"""

import pywt
import numpy as N

from mvpa.mappers.base import Mapper
from mvpa.base.dochelpers import enhancedDocString

if __debug__:
    from mvpa.base import debug

# WaveletPacket and WaveletTransformation mappers share lots of common
# functionality at the moment

class _WaveletMapper(Mapper):
    """Generic class for Wavelet mappers (decomposition and packet)
    """

    def __init__(self, dim=1, wavelet='sym4', mode='per', maxlevel=None):
        """Initialize WaveletPacket mapper

        :Parameters:
          dim : int or tuple of int
            dimensions to work across (for now just scalar value, ie 1D
            transformation) is supported
          wavelet : basestring
            one from the families available withing pywt package
          mode : basestring
            periodization mode
          maxlevel : int or None
            number of levels to use. If None - automatically selected by pywt
        """
        Mapper.__init__(self)

        self._dim = dim
        """Dimension to work along"""

        self._maxlevel = maxlevel
        """Maximal level of decomposition. None for automatic"""

        if not wavelet in pywt.wavelist():
            raise ValueError, \
                  "Unknown family of wavelets '%s'. Please use one " \
                  "available from the list %s" % (wavelet, pywt.wavelist())
        self._wavelet = wavelet
        """Wavelet family to use"""

        if not mode in pywt.MODES.modes:
            raise ValueError, \
                  "Unknown periodization mode '%s'. Please use one " \
                  "available from the list %s" % (mode, pywt.MODES.modes)
        self._mode = mode
        """Periodization mode"""

    def forward(self, data):
        data = N.asanyarray(data)
        self._inshape = data.shape
        self._intimepoints = data.shape[self._dim]
        res = self._forward(data)
        self._outshape = res.shape
        return res


    def reverse(self, data):
        data = N.asanyarray(data)
        return self._reverse(data)

    def _forward(self, *args):
        raise NotImplementedError

    def _reverse(self, *args):
        raise NotImplementedError


    def getInShape(self):
        """Returns a one-tuple with the number of original features."""
        return self._inshape[1:]


    def getOutShape(self):
        """Returns a tuple with the shape of output components."""
        return self._outshape[1:]


    def getInSize(self):
        """Returns the number of original features."""
        return self._inshape[1:]


    def getOutSize(self):
        """Returns the number of wavelet components."""
        return self._outshape[1:]


    def selectOut(self, outIds):
        """Choose a subset of components...

        just use MaskMapper on top?"""
        raise NotImplementedError, "Please use in conjunction with MaskMapper"


    __doc__ = enhancedDocString('_WaveletMapper', locals(), Mapper)


def _getIndexes(shape, dim):
    """Generator for coordinate tuples providing slice for all in `dim`

    XXX Somewhat sloppy implementation... but works...
    """
    if len(shape) < dim:
        raise ValueError, "Dimension %d is incorrect for a shape %s" % \
              (dim, shape)
    n = len(shape)
    curindexes = [0] * n
    curindexes[dim] = slice(None)       # all elements for dimension dim
    while True:
        yield tuple(curindexes)
        for i in xrange(n):
            if i == dim and dim == n-1:
                return                  # we reached it -- thus time to go
            if curindexes[i] == shape[i] - 1:
                if i == n-1:
                    return
                curindexes[i] = 0
            else:
                if i != dim:
                    curindexes[i] += 1
                    break


class WaveletPacketMapper(_WaveletMapper):
    """Convert signal into an overcomplete representaion using Wavelet packet
    """

    def _forward(self, data):
        if __debug__:
            debug('MAP', "Converting signal using DWP")

        wp = None
        levels_length = None                # total length at each level
        levels_lengths = None                # list of lengths per each level
        for indexes in _getIndexes(data.shape, self._dim):
            if __debug__:
                debug('MAP_', " %s" % (indexes,), lf=False, cr=True)
            WP = pywt.WaveletPacket(
                data[indexes],
                wavelet=self._wavelet,
                mode=self._mode, maxlevel=self._maxlevel)

            if levels_length is None:
                levels_length = [None] * WP.maxlevel
                levels_lengths = [None] * WP.maxlevel

            levels_datas = []
            for level in xrange(WP.maxlevel):
                level_nodes = WP.get_level(level+1)
                level_datas = [node.data for node in level_nodes]

                level_lengths = [len(x) for x in level_datas]
                level_length = N.sum(level_lengths)

                if levels_lengths[level] is None:
                    levels_lengths[level] = level_lengths
                elif levels_lengths[level] != level_lengths:
                    raise RuntimeError, \
                          "ADs of same level of different samples should have same number of elements." \
                          " Got %s, was %s" % (level_lengths, levels_lengths[level])

                if levels_length[level] is None:
                    levels_length[level] = level_length
                elif levels_length[level] != level_length:
                    raise RuntimeError, \
                          "Levels of different samples should have same number of elements." \
                          " Got %d, was %d" % (level_length, levels_length[level])

                level_data = N.hstack(level_datas)
                levels_datas.append(level_data)

            # assert(len(data) == levels_length)
            # assert(len(data) >= Ntimepoints)
            if wp is None:
                newdim = list(data.shape)
                newdim[self._dim] = N.sum(levels_length)
                wp = N.empty( tuple(newdim) )
            wp[indexes] = N.hstack(levels_datas)

        self.levels_lengths, self.levels_length = levels_lengths, levels_length
        if __debug__:
            debug('MAP_', "")
            debug('MAP', "Done convertion into wp. Total size %s" % str(wp.shape))
        return wp

    def _reverse(self, data):
        raise NotImplementedError


class WaveletTransformationMapper(_WaveletMapper):
    """Convert signal into wavelet representaion
    """

    def _forward(self, data):
        """Decompose signal into wavelets's coefficients via dwt
        """
        if __debug__:
            debug('MAP', "Converting signal using DWT")
        wd = None
        coeff_lengths = None
        for indexes in _getIndexes(data.shape, self._dim):
            if __debug__:
                debug('MAP_', " %s" % (indexes,), lf=False, cr=True)
            coeffs = pywt.wavedec(
                data[indexes],
                wavelet=self._wavelet,
                mode=self._mode,
                level=self._maxlevel)
            # Silly Yarik embedds extraction of statistics right in place
            #stats = []
            #for coeff in coeffs:
            #    stats_ = [N.std(coeff),
            #              N.sqrt(N.dot(coeff, coeff)),
            #              ]# + list(N.histogram(coeff, normed=True)[0]))
            #    stats__ = list(coeff) + stats_[:]
            #    stats__ += list(N.log(stats_))
            #    stats__ += list(N.sqrt(stats_))
            #    stats__ += list(N.array(stats_)**2)
            #    stats__ += [  N.median(coeff), N.mean(coeff), scipy.stats.kurtosis(coeff) ]
            #    stats.append(stats__)
            #coeffs = stats
            coeff_lengths_ = N.array([len(x) for x in coeffs])
            if coeff_lengths is None:
                coeff_lengths = coeff_lengths_
            assert((coeff_lengths == coeff_lengths_).all())
            if wd is None:
                newdim = list(data.shape)
                newdim[self._dim] = N.sum(coeff_lengths)
                wd = N.empty( tuple(newdim) )
            coeff = N.hstack(coeffs)
            wd[indexes] = coeff
        if __debug__:
            debug('MAP_', "")
            debug('MAP', "Done DWT. Total size %s" % str(wd.shape))
        self.lengths = coeff_lengths
        return wd

    def _reverse(self, wd):
        if __debug__:
            debug('MAP', "Performing iDWT")
        signal = None
        wd_offsets = [0] + list(N.cumsum(self.lengths))
        Nlevels = len(self.lengths)
        Ntime_points = self._intimepoints #len(time_points)
        # unfortunately sometimes due to padding iDWT would return longer
        # sequences, thus we just limit to the right ones

        for indexes in _getIndexes(wd.shape, self._dim):
            if __debug__:
                debug('MAP_', " %s" % (indexes,), lf=False, cr=True)
            wd_sample = wd[indexes]
            wd_coeffs = [wd_sample[wd_offsets[i]:wd_offsets[i+1]] for i in xrange(Nlevels)]
            # need to compose original list
            time_points = pywt.waverec(
                wd_coeffs, wavelet=self._wavelet, mode=self._mode)
            if signal is None:
                newdim = list(wd.shape)
                newdim[self._dim] = Ntime_points
                signal = N.empty(newdim)
            signal[indexes] = time_points[:Ntime_points]
        if __debug__:
            debug('MAP_', "")
            debug('MAP', "Done iDWT. Total size %s" % (signal.shape, ))
        return signal


