# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA EEP dataset"""

import os.path
import numpy as N

from mvpa import pymvpa_dataroot
from mvpa.base import externals
from mvpa.datasets.eep import eep_dataset, EEPBin

from mvpa.testing.tools import assert_equal, assert_true, \
     assert_array_almost_equal

def test_eep_load():
    eb = EEPBin(os.path.join(pymvpa_dataroot, 'eep.bin'))

    ds = [ eep_dataset(source, targets=[1, 2]) for source in
            (eb, os.path.join(pymvpa_dataroot, 'eep.bin')) ]

    for d in ds:
        assert_equal(d.nsamples, 2)
        assert_equal(d.nfeatures, 128)
        assert_equal(N.unique(d.fa.channels[4*23:4*23+4]), 'Pz')
        assert_array_almost_equal([N.arange(-0.002, 0.005, 0.002)] * 32,
                                  d.a.mapper.reverse1(d.fa.timepoints))


def test_eep_bin():
    eb = EEPBin(os.path.join(pymvpa_dataroot, 'eep.bin'))

    assert_equal(eb.nchannels, 32)
    assert_equal(eb.nsamples, 2)
    assert_equal(eb.ntimepoints, 4)
    assert_true(eb.t0 - eb.dt < 0.00000001)
    assert_equal(len(eb.channels), 32)
    assert_equal(eb.data.shape, (2, 32, 4))


    # XXX put me back whenever there is a proper resamples()
#     def test_resampling(self):
#         ds = eep_dataset(os.path.join(pymvpa_dataroot, 'eep.bin'),
#                          targets=[1, 2])
#         channelids = N.array(ds.a.channelids).copy()
#         self.failUnless(N.round(ds.samplingrate) == 500.0)
# 
#         if not externals.exists('scipy'):
#             return
# 
#         # should puke when called with nothing
#         self.failUnlessRaises(ValueError, ds.resample)
# 
#         # now for real -- should divide nsamples into half
#         rds = ds.resample(sr=250, inplace=False)
#         # We should have not changed anything
#         self.failUnless(N.round(ds.samplingrate) == 500.0)
# 
#         # by default do 'inplace' resampling
#         ds.resample(sr=250)
#         for d in [rds, ds]:
#             self.failUnless(N.round(d.samplingrate) == 250)
#             self.failUnless(d.nsamples == 2)
#             self.failUnless(N.abs((d.a.dt - 1.0/250)/d.a.dt)<1e-5)
#             self.failUnless(N.all(d.a.channelids == channelids))
#             # lets now see if we still have a mapper
#             self.failUnless(d.O.shape == (2, len(channelids), 2))
