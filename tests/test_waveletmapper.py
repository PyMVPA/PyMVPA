#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Wavelet mappers"""

from mvpa.base import externals
externals.exists('pywt', raiseException=True)

import unittest
from mvpa.misc.copy import deepcopy
import numpy as N

from mvpa.mappers.boxcar import BoxcarMapper
from mvpa.mappers.wavelet import *
from mvpa.datasets import Dataset

from tests_warehouse import datasets

class WaveletMappersTests(unittest.TestCase):

    def testSimpleWDM(self):
        """
        """
        ds = datasets['uni2small']
        d2d = ds.samples
        ws = 9                          # size of timeline for wavelet
        sp = N.arange(ds.nsamples-ws*2) + ws

        # create 3D instance (samples x timepoints x channels)
        bcm = BoxcarMapper(sp, ws)
        d3d = bcm(d2d)

        # use wavelet mapper
        wdm = WaveletDecompositionMapper()
        d3d_wd = wdm(d3d)

        print d2d.shape, d3d.shape, d3d_wd.shape

    def testCompareToOld(self):
        """Good just to compare if I didn't screw up anything... treat
        it as a regression test
        """
        import mvpa.mappers.wavelet_ as wavelet_

        ds = datasets['uni2medium']
        d2d = ds.samples
        ws = 16                          # size of timeline for wavelet
        sp = N.arange(ds.nsamples-ws*2) + ws

        # create 3D instance (samples x timepoints x channels)
        bcm = BoxcarMapper(sp, ws)
        d3d = bcm(d2d)

        # use wavelet mapper
        for wdm, wdm_ in ((WaveletDecompositionMapper(),
                           wavelet_.WaveletDecompositionMapper()),
                          (WaveletPacketMapper(),
                           wavelet_.WaveletPacketMapper()),):
            d3d_wd = wdm(d3d)
            d3d_wd_ = wdm_(d3d)

            self.failUnless((d3d_wd == d3d_wd_).all(),
                            msg="We should have got same result with old and new code. Got %s and %s" % (d3d_wd, d3d_wd_))


def suite():
    return unittest.makeSuite(WaveletMappersTests)


if __name__ == '__main__':
    import runner

