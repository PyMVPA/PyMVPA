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

if __debug__:
    from mvpa.base import debug

# WaveletPacket and WaveletDecomposition mappers share lots of common
# functionality at the moment

class _WaveletMapper(Mapper):
    """Generic class for Wavelet mappers (decomposition and packet)
    """
    # heavy TODO to make proper mapper

    def __init__(self, wavelet='sym4', mode='per', maxlevel=None):
        """Initialize WaveletPacket mapper

        :Parameters:
          wavelet : basestring
            one from the families available withing pywt package
          mode : basestring
            periodization mode
          maxlevel : int or None
            number of levels to use. If None - automatically selected by pywt
        """
        Mapper.__init__(self)

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

    def forward(self, *args):
        raise NotImplementedError

    def inverse(self, *args):
        raise NotImplementedError


class WaveletPacketMapper(_WaveletMapper):
    """Convert signal into an overcomplete representaion using Wavelet packet
    """

    def forward(self, data):
        if __debug__:
            debug('MAP', "Converting signal using DWP")
        Nsamples, Ntimepoints, Nchans = data.shape
        wp = None
        levels_length = None                # total length at each level
        levels_lengths = None                # list of lengths per each level
        for sample_id in xrange(data.shape[0]):
            for chan_id in xrange(data.shape[2]):
                if __debug__:
                    debug('MAP_', " %d/%d" % (sample_id, chan_id), lf=False, cr=True)
                WP = pywt.WaveletPacket(
                    data[sample_id, :, chan_id],
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
                    wp = N.empty( (Nsamples, N.sum(levels_length),  Nchans) )
                wp[sample_id, :, chan_id] = N.hstack(levels_datas)

        self.levels_lengths, self.levels_length = levels_lengths, levels_length
        if __debug__:
            debug('MAP_', "")
            debug('MAP', "Done convertion into wp. Total size %s" % str(wp.shape))
        return wp

    def reverse(self, data):
        raise NotImplementedError


class WaveletDecompositionMapper(_WaveletMapper):
    """Convert signal into wavelet representaion
    """

    def forward(self, data):
        """Decompose signal into wavelets's coefficients via dwt
        """
        if __debug__:
            debug('MAP', "Converting signal using DWT")
        if len(data.shape) != 3:
            raise ValueError, \
                  "For now only 3D datasets (samples x timepoints x channels) are supported"
        Nsamples, Ntimepoints, Nchans = data.shape
        wd = None
        coeff_lengths = None
        for sample_id in xrange(Nsamples):
            for chan_id in xrange(Nchans):
                if __debug__:
                    debug('MAP_', " %d/%d" % (sample_id, chan_id), lf=False, cr=True)
                coeffs = pywt.wavedec(
                    data[sample_id, :, chan_id],
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
                    wd = N.empty( (Nsamples, coeff_lengths.sum(), Nchans) )
                coeff = N.hstack(coeffs)
                wd[sample_id, :, chan_id] = coeff
        if __debug__:
            debug('MAP_', "")
            debug('MAP', "Done DWT. Total size %s" % str(wd.shape))
        self.lengths = coeff_lengths    # XXX move under some proper attrib
        return wd

    def reverse(self, wd):
        if __debug__:
            debug('MAP', "Performing iDWT")
        Nsamples, Ncoefs, Nchans = wd.shape
        signal = None
        wd_offsets = [0] + list(N.cumsum(self.lengths))
        Nlevels = len(self.lengths)
        for sample_id in xrange(Nsamples):
            for chan_id in xrange(Nchans):
                if __debug__:
                    debug('MAP_', " %d/%d" % (sample_id, chan_id), lf=False, cr=True)
                wd_sample = wd[sample_id, :, chan_id]
                wd_coeffs = [wd_sample[wd_offsets[i]:wd_offsets[i+1]] for i in xrange(Nlevels)]
                # need to compose original list
                time_points = pywt.waverec(
                    wd_coeffs, wavelet=self._wavelet, mode=self._mode)
                Ntime_points = len(time_points)
                if signal is None:
                    signal = N.empty( (Nsamples, Ntime_points, Nchans) )
                signal[sample_id, :, chan_id] = time_points
        if __debug__:
            debug('MAP_', "")
            debug('MAP', "Done iDWT. Total size %s" % (signal.shape, ))
        return signal


