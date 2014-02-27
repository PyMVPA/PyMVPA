# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wavelet transformation"""

from mvpa2.base import externals

if externals.exists('pywt', raise_=True):
    # import conditional to be able to import the whole module while building
    # the docs even if pywt is not installed
    import pywt

import numpy as np

from mvpa2.base import warning
from mvpa2.mappers.base import Mapper

if __debug__:
    from mvpa2.base import debug

# WaveletPacket and WaveletTransformation mappers share lots of common
# functionality at the moment

class _WaveletMapper(Mapper):
    """Generic class for Wavelet mappers (decomposition and packet)
    """

    def __init__(self, dim=1, wavelet='sym4', mode='per', maxlevel=None):
        """Initialize _WaveletMapper mapper

        Parameters
        ----------
        dim : int or tuple of int
          dimensions to work across (for now just scalar value, ie 1D
          transformation) is supported
        wavelet : str
          one from the families available withing pywt package
        mode : str
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


    def _forward_data(self, data):
        data = np.asanyarray(data)
        self._inshape = data.shape
        self._intimepoints = data.shape[self._dim]
        res = self._wm_forward(data)
        self._outshape = res.shape
        return res


    def _reverse_data(self, data):
        data = np.asanyarray(data)
        return self._wm_reverse(data)


    def _wm_forward(self, *args):
        raise NotImplementedError


    def _wm_reverse(self, *args):
        raise NotImplementedError



##REF: Name was automagically refactored
def _get_indexes(shape, dim):
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

        Parameters
        ----------
        level : int or None
          What level to decompose at. If 'None' data for all levels
          is provided, but due to different sizes, they are placed
          in 1D row.
        """

        _WaveletMapper.__init__(self,**kwargs)

        self.__level = level


    # XXX too much of duplications between such methods -- it begs
    #     refactoring
    ##REF: Name was automagically refactored
    def __forward_single_level(self, data):
        if __debug__:
            debug('MAP', "Converting signal using DWP (single level)")

        wp = None

        level = self.__level
        wavelet = self._wavelet
        mode = self._mode
        dim = self._dim

        level_paths = None
        for indexes in _get_indexes(data.shape, self._dim):
            if __debug__:
                debug('MAP_', " %s" % (indexes,), lf=False, cr=True)
            WP = pywt.WaveletPacket(
                data[indexes], wavelet=wavelet,
                mode=mode, maxlevel=level)

            level_nodes = WP.get_level(level)
            if level_paths is None:
                # Needed for reconstruction
                self.__level_paths = np.array([node.path for node in level_nodes])
            level_datas = np.array([node.data for node in level_nodes])

            if wp is None:
                newdim = data.shape
                newdim = newdim[:dim] + level_datas.shape + newdim[dim+1:]
                if __debug__:
                    debug('MAP_', "Initializing storage of size %s for single "
                          "level (%d) mapping of data of size %s" % (newdim, level, data.shape))
                wp = np.empty( tuple(newdim) )

            wp[indexes] = level_datas

        return wp


    ##REF: Name was automagically refactored
    def __forward_multiple_levels(self, data):
        wp = None
        levels_length = None                # total length at each level
        levels_lengths = None                # list of lengths per each level
        for indexes in _get_indexes(data.shape, self._dim):
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
                level_length = np.sum(level_lengths)

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

                level_data = np.hstack(level_datas)
                levels_datas.append(level_data)

            # assert(len(data) == levels_length)
            # assert(len(data) >= Ntimepoints)
            if wp is None:
                newdim = list(data.shape)
                newdim[self._dim] = np.sum(levels_length)
                wp = np.empty( tuple(newdim) )
            wp[indexes] = np.hstack(levels_datas)

        self.levels_lengths, self.levels_length = levels_lengths, levels_length
        if __debug__:
            debug('MAP_', "")
            debug('MAP', "Done convertion into wp. Total size %s" % str(wp.shape))
        return wp


    def _wm_forward(self, data):
        if __debug__:
            debug('MAP', "Converting signal using DWP")

        if self.__level is None:
            return self.__forward_multiple_levels(data)
        else:
            return self.__forward_single_level(data)

    #
    # Reverse mapping
    #
    ##REF: Name was automagically refactored
    def __reverse_single_level(self, wp):

        # local bindings
        level_paths = self.__level_paths

        # define wavelet packet to use
        WP = pywt.WaveletPacket(
            data=None, wavelet=self._wavelet,
            mode=self._mode, maxlevel=self.__level)

        # prepare storage
        signal_shape = wp.shape[:1] + self._inshape[1:]
        signal = np.zeros(signal_shape)
        Ntime_points = self._intimepoints
        for indexes in _get_indexes(signal_shape,
                                   self._dim):
            if __debug__:
                debug('MAP_', " %s" % (indexes,), lf=False, cr=True)

            for path, level_data in zip(level_paths, wp[indexes]):
                WP[path] = level_data

            signal[indexes] = WP.reconstruct(True)[:Ntime_points]

        return signal


    def _wm_reverse(self, data):
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
                warning("%s: Reverse mapping with this version of 'pywt' might "
                        "result in incorrect data in the tails of the signal. "
                        "Please check for an update of 'pywt', or be careful "
                        "when interpreting the edges of the reverse mapped "
                        "data." % self.__class__.__name__)
            return self.__reverse_single_level(data)



class WaveletTransformationMapper(_WaveletMapper):
    """Convert signal into wavelet representaion
    """

    def _wm_forward(self, data):
        """Decompose signal into wavelets's coefficients via dwt
        """
        if __debug__:
            debug('MAP', "Converting signal using DWT")
        wd = None
        coeff_lengths = None
        for indexes in _get_indexes(data.shape, self._dim):
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
            #    stats_ = [np.std(coeff),
            #              np.sqrt(np.dot(coeff, coeff)),
            #              ]# + list(np.histogram(coeff, normed=True)[0]))
            #    stats__ = list(coeff) + stats_[:]
            #    stats__ += list(np.log(stats_))
            #    stats__ += list(np.sqrt(stats_))
            #    stats__ += list(np.array(stats_)**2)
            #    stats__ += [  np.median(coeff), np.mean(coeff), scipy.stats.kurtosis(coeff) ]
            #    stats.append(stats__)
            #coeffs = stats
            coeff_lengths_ = np.array([len(x) for x in coeffs])
            if coeff_lengths is None:
                coeff_lengths = coeff_lengths_
            assert((coeff_lengths == coeff_lengths_).all())
            if wd is None:
                newdim = list(data.shape)
                newdim[self._dim] = np.sum(coeff_lengths)
                wd = np.empty( tuple(newdim) )
            coeff = np.hstack(coeffs)
            wd[indexes] = coeff
        if __debug__:
            debug('MAP_', "")
            debug('MAP', "Done DWT. Total size %s" % str(wd.shape))
        self.lengths = coeff_lengths
        return wd


    def _wm_reverse(self, wd):
        if __debug__:
            debug('MAP', "Performing iDWT")
        signal = None
        wd_offsets = [0] + list(np.cumsum(self.lengths))
        nlevels = len(self.lengths)
        Ntime_points = self._intimepoints #len(time_points)
        # unfortunately sometimes due to padding iDWT would return longer
        # sequences, thus we just limit to the right ones

        for indexes in _get_indexes(wd.shape, self._dim):
            if __debug__:
                debug('MAP_', " %s" % (indexes,), lf=False, cr=True)
            wd_sample = wd[indexes]
            wd_coeffs = [wd_sample[wd_offsets[i]:wd_offsets[i+1]] for i in xrange(nlevels)]
            # need to compose original list
            time_points = pywt.waverec(
                wd_coeffs, wavelet=self._wavelet, mode=self._mode)
            if signal is None:
                newdim = list(wd.shape)
                newdim[self._dim] = Ntime_points
                signal = np.empty(newdim)
            signal[indexes] = time_points[:Ntime_points]
        if __debug__:
            debug('MAP_', "")
            debug('MAP', "Done iDWT. Total size %s" % (signal.shape, ))
        return signal
