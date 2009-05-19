# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wavelet mappers"""

from mvpa.base import externals

if externals.exists('pywt', raiseException=True):
    # import conditional to be able to import the whole module while building
    # the docs even if pywt is not installed
    import pywt

import numpy as N

from mvpa.base import warning
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
        """Initialize _WaveletMapper mapper

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
    curindexes[dim] = Ellipsis#slice(None)       # all elements for dimension dim
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

    def __init__(self, level=None, **kwargs):
        """Initialize WaveletPacketMapper mapper

        :Parameters:
          level : int or None
            What level to decompose at. If 'None' data for all levels
            is provided, but due to different sizes, they are placed
            in 1D row.
        """

        _WaveletMapper.__init__(self,**kwargs)

        self.__level = level


    # XXX too much of duplications between such methods -- it begs
    #     refactoring
    def __forwardSingleLevel(self, data):
        if __debug__:
            debug('MAP', "Converting signal using DWP (single level)")

        wp = None

        level = self.__level
        wavelet = self._wavelet
        mode = self._mode
        dim = self._dim

        level_paths = None
        for indexes in _getIndexes(data.shape, self._dim):
            if __debug__:
                debug('MAP_', " %s" % (indexes,), lf=False, cr=True)
            WP = pywt.WaveletPacket(
                data[indexes], wavelet=wavelet,
                mode=mode, maxlevel=level)

            level_nodes = WP.get_level(level)
            if level_paths is None:
                # Needed for reconstruction
                self.__level_paths = N.array([node.path for node in level_nodes])
            level_datas = N.array([node.data for node in level_nodes])

            if wp is None:
                newdim = data.shape
                newdim = newdim[:dim] + level_datas.shape + newdim[dim+1:]
                if __debug__:
                    debug('MAP_', "Initializing storage of size %s for single "
                          "level (%d) mapping of data of size %s" % (newdim, level, data.shape))
                wp = N.empty( tuple(newdim) )

            wp[indexes] = level_datas

        return wp


    def __forwardMultipleLevels(self, data):
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


    def _forward(self, data):
        if __debug__:
            debug('MAP', "Converting signal using DWP")

        if self.__level is None:
            return self.__forwardMultipleLevels(data)
        else:
            return self.__forwardSingleLevel(data)

    #
    # Reverse mapping
    #
    def __reverseSingleLevel(self, wp):

        # local bindings
        level_paths = self.__level_paths

        # define wavelet packet to use
        WP = pywt.WaveletPacket(
            data=None, wavelet=self._wavelet,
            mode=self._mode, maxlevel=self.__level)

        # prepare storage
        signal_shape = wp.shape[:1] + self.getInSize()
        signal = N.zeros(signal_shape)
        Ntime_points = self._intimepoints
        for indexes in _getIndexes(signal_shape,
                                   self._dim):
            if __debug__:
                debug('MAP_', " %s" % (indexes,), lf=False, cr=True)

            for path, level_data in zip(level_paths, wp[indexes]):
                WP[path] = level_data

            signal[indexes] = WP.reconstruct(True)[:Ntime_points]

        return signal


    def _reverse(self, data):
        if __debug__:
            debug('MAP', "Converting signal back using DWP")

        if self.__level is None:
            raise NotImplementedError
        else:
            if not externals.exists('pywt wp reconstruct'):
                raise NotImplementedError, \
                      "Reconstruction for a single level for versions of " \
                      "pywt < 0.1.7 (revision 103) is not supported"
            if not externals.exists('pywt wp reconstruct fixed'):
                warning("Reconstruction using available version of pywt might "
                        "result in incorrect data in the tails of the signal")
            return self.__reverseSingleLevel(data)





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


