# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for filter mappers'''


from mvpa2.testing import *
skip_if_no_external('scipy')

import numpy as np

from mvpa2.datasets import Dataset, vstack
from mvpa2.mappers.filters import FFTResampleMapper, iir_filter

def test_resample():
    time = np.linspace(0, 2*np.pi, 100)
    ds = Dataset(np.vstack((np.sin(time), np.cos(time))).T,
                 sa = {'time': time,
                       'section': np.repeat(range(10), 10)})
    assert_equal(ds.shape, (100, 2))

    # downsample
    num = 10
    rm = FFTResampleMapper(num, window=('gauss', 50),
                           position_attr='time',
                           attr_strategy='sample')
    mds = rm.forward(ds)
    assert_equal(mds.shape, (num, ds.nfeatures))
    # didn't change the orig
    assert_equal(len(ds), 100)

    # check position-based resampling
    ds_partial = ds[0::10]
    mds_partial = rm.forward(ds_partial)
    # despite different input sampling should yield the same output timepoints
    assert_array_almost_equal(mds.sa.time, mds_partial.sa.time)
    # exclude the first points to prevent edge effects, but the data should be
    # very similar too
    assert_array_almost_equal(mds.samples[2:], mds_partial.samples[2:], decimal=2)
    # simple sample of sa's should give meaningful stuff
    assert_array_equal(mds.sa.section, range(10))

    # and now for a dataset with chunks
    cds = vstack([ds.copy(), ds.copy()])
    cds.sa['chunks'] = np.repeat([0,1], len(ds))
    rm = FFTResampleMapper(num, attr_strategy='sample', chunks_attr='chunks',
                           window=('gauss', 50))
    mcds = rm.forward(cds)
    assert_equal(mcds.shape, (20, 2))
    assert_array_equal(mcds.sa.section, np.tile(range(10),2))
    # each individual chunks should be identical to previous dataset
    assert_array_almost_equal(mds.samples, mcds.samples[:10])
    assert_array_almost_equal(mds.samples, mcds.samples[10:])


def test_iirfilter():
    # dataset with one feature from two waves
    t = np.linspace(0, 1.0, 2001)
    xlow = np.sin(2 * np.pi * 5 * t)
    xhigh = np.sin(2 * np.pi * 250 * t)
    x = xlow + xhigh
    ds = Dataset(x, sa={'sid': np.arange(len(x))}, fa={'fid':['theone']})

    # butterworth filter with a cutoff between the waves
    from scipy import signal
    b, a = signal.butter(8, 0.125)
    mds = iir_filter(ds, b, a, padlen=150)
    # check we get just the slow wave out (compensate for edge artifacts)
    assert_false(np.sum(np.abs(mds.samples[100:-100,0] - xlow[100:-100]) > 0.001))
    assert_equal(len(ds.sa), len(mds.sa))
    assert_equal(len(ds.fa), len(mds.fa))
    assert_array_equal(ds.fa.fid, mds.fa.fid)
    assert_array_equal(ds.sa.sid, mds.sa.sid)
