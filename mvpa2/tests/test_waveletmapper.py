# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Wavelet mappers"""

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets
skip_if_no_external('pywt')

from mvpa2.base import externals

import unittest
from mvpa2.support.copy import deepcopy
import numpy as np

from mvpa2.mappers.boxcar import BoxcarMapper
from mvpa2.mappers.wavelet import *
from mvpa2.datasets import Dataset


class WaveletMappersTests(unittest.TestCase):

    def test_simple_wdm(self):
        """
        """
        ds = datasets['uni2medium']
        d2d = ds.samples
        ws = 15                          # size of timeline for wavelet
        sp = np.arange(ds.nsamples-ws*2) + ws

        # create 3D instance (samples x timepoints x channels)
        bcm = BoxcarMapper(sp, ws)
        d3d = bcm.forward(d2d)

        # use wavelet mapper
        wdm = WaveletTransformationMapper()
        d3d_wd = wdm.forward(d3d)
        d3d_swap = d3d.swapaxes(1,2)

        self.assertRaises(ValueError, WaveletTransformationMapper,
                              wavelet='bogus')
        self.assertRaises(ValueError, WaveletTransformationMapper,
                              mode='bogus')

        # use wavelet mapper
        for wdm, wdm_swap in ((WaveletTransformationMapper(),
                               WaveletTransformationMapper(dim=2)),
                              (WaveletPacketMapper(),
                               WaveletPacketMapper(dim=2))):
          for dd, dd_swap in ((d3d, d3d_swap),
                              (d2d, None)):
            dd_wd = wdm.forward(dd)
            if dd_swap is not None:
                dd_wd_swap = wdm_swap.forward(dd_swap)

                self.assertTrue((dd_wd == dd_wd_swap.swapaxes(1,2)).all(),
                                msg="We should have got same result with swapped "
                                "dimensions and explicit mentioining of it. "
                                "Got %s and %s" % (dd_wd, dd_wd_swap))

            # some sanity checks
            self.assertTrue(dd_wd.shape[0] == dd.shape[0])

            if not isinstance(wdm, WaveletPacketMapper):
                # we can do reverse only for DWT
                dd_rev = wdm.reverse(dd_wd)
                # inverse transform might be not exactly as the
                # input... but should be very close ;-)
                self.assertEqual(dd_rev.shape, dd.shape,
                                     msg="Shape should be the same after iDWT")

                diff = np.linalg.norm(dd - dd_rev)
                ornorm = np.linalg.norm(dd)
                self.assertTrue(diff/ornorm < 1e-10)


    def test_simple_wp1_level(self):
        """
        """

        ds = datasets['uni2large']
        d2d = ds.samples
        ws = 50                          # size of timeline for wavelet
        sp = (np.arange(ds.nsamples - ws*2) + ws)[:4]

        # create 3D instance (samples x timepoints x channels)
        bcm = BoxcarMapper(sp, ws)
        d3d = bcm.forward(d2d)

        # use wavelet mapper
        wdm = WaveletPacketMapper(level=2, wavelet='sym2')
        d3d_wd = wdm.forward(d3d)

        # Check dimensionality
        d3d_wds, d3ds = d3d_wd.shape, d3d.shape
        self.assertTrue(len(d3d_wds) == len(d3ds)+1)
        self.assertTrue(d3d_wds[1] * d3d_wds[2] >= d3ds[1])
        self.assertTrue(d3d_wds[0] == d3ds[0])
        self.assertTrue(d3d_wds[-1] == d3ds[-1])
        #print d2d.shape, d3d.shape, d3d_wd.shape

        if externals.exists('pywt wp reconstruct'):
            # Test reverse -- should be identical
            # we can do reverse only for DWT
            d3d_rev = wdm.reverse(d3d_wd)

            # inverse transform might be not exactly as the
            # input... but should be very close ;-)
            self.assertEqual(d3d_rev.shape, d3d.shape,
                                 msg="Shape should be the same after iDWT")

            diff = np.linalg.norm(d3d - d3d_rev)
            ornorm = np.linalg.norm(d3d)

            skip_if_no_external('pywt wp reconstruct fixed')
            self.assertTrue(diff/ornorm < 1e-10)
        else:
            self.assertRaises(NotImplementedError, wdm.reverse, d3d_wd)


    ##REF: Name was automagically refactored
    def _test_compare_to_old(self):  # pragma: no cover
        """Good just to compare if I didn't screw up anything... treat
        it as a regression test
        """
        import mvpa2.mappers.wavelet_ as wavelet_

        ds = datasets['uni2medium']
        d2d = ds.samples
        ws = 16                          # size of timeline for wavelet
        sp = np.arange(ds.nsamples-ws*2) + ws

        # create 3D instance (samples x timepoints x channels)
        bcm = BoxcarMapper(sp, ws)
        d3d = bcm.forward(d2d)

        # use wavelet mapper
        for wdm, wdm_ in ((WaveletTransformationMapper(),
                           wavelet_.WaveletTransformationMapper()),
                          (WaveletPacketMapper(),
                           wavelet_.WaveletPacketMapper()),):
            d3d_wd = wdm(d3d)
            d3d_wd_ = wdm_(d3d)

            self.assertTrue((d3d_wd == d3d_wd_).all(),
                msg="We should have got same result with old and new code. "
                    "Got %s and %s" % (d3d_wd, d3d_wd_))


def suite():  # pragma: no cover
    return unittest.makeSuite(WaveletMappersTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

