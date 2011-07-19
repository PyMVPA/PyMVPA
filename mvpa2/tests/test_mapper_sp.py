# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for detrending mapper (requiring SciPy)."""

import numpy as np

from mvpa2.testing.tools import *

skip_if_no_external('scipy')

from mvpa2.datasets import Dataset, dataset_wizard
from mvpa2.mappers.detrend import PolyDetrendMapper, poly_detrend

def test_polydetrend():
    samples_forwhole = np.array( [[1.0, 2, 3, 4, 5, 6],
                                 [-2.0, -4, -6, -8, -10, -12]], ndmin=2 ).T
    samples_forchunks = np.array( [[1.0, 2, 3, 3, 2, 1],
                                  [-2.0, -4, -6, -6, -4, -2]], ndmin=2 ).T
    chunks = [0, 0, 0, 1, 1, 1]
    chunks_bad = [ 0, 0, 1, 1, 1, 0]
    target_whole = np.array( [[-3.0, -2, -1, 1, 2, 3],
                             [-6, -4, -2,  2, 4, 6]], ndmin=2 ).T
    target_chunked = np.array( [[-1.0, 0, 1, 1, 0, -1],
                               [2, 0, -2, -2, 0, 2]], ndmin=2 ).T


    ds = Dataset(samples_forwhole)

    # this one will auto-train the mapper on first use
    dm = PolyDetrendMapper(polyord=1, space='police')
    mds = dm.forward(ds)
    # features are linear trends, so detrending should remove all
    assert_array_almost_equal(mds.samples, np.zeros(mds.shape))
    # we get the information where each sample is assumed to be in the
    # space spanned by the polynomials
    assert_array_equal(mds.sa.police, np.arange(len(ds)))

    # hackish way to get the previous regressors into a dataset
    ds.sa['opt_reg_const'] = dm._regs[:,0]
    ds.sa['opt_reg_lin'] = dm._regs[:,1]
    # using these precomputed regressors, we should get the same result as
    # before even if we do not generate a regressor for linear
    dm_optreg = PolyDetrendMapper(polyord=0,
                                  opt_regs=['opt_reg_const', 'opt_reg_lin'])
    mds_optreg = dm_optreg.forward(ds)
    assert_array_almost_equal(mds_optreg, np.zeros(mds.shape))


    ds = Dataset(samples_forchunks)
    # 'constant' detrending removes the mean
    mds = PolyDetrendMapper(polyord=0).forward(ds)
    assert_array_almost_equal(
            mds.samples,
            samples_forchunks - np.mean(samples_forchunks, axis=0))
    # if there is no GLOBAL linear trend it should be identical to mean removal
    # even if trying to remove linear
    mds2 = PolyDetrendMapper(polyord=1).forward(ds)
    assert_array_almost_equal(mds, mds2)

    # chunk-wise detrending
    ds = dataset_wizard(samples_forchunks, chunks=chunks)
    dm = PolyDetrendMapper(chunks_attr='chunks', polyord=1, space='police')
    mds = dm.forward(ds)
    # features are chunkswise linear trends, so detrending should remove all
    assert_array_almost_equal(mds.samples, np.zeros(mds.shape))
    # we get the information where each sample is assumed to be in the
    # space spanned by the polynomials, which is the identical linspace in both
    # chunks
    assert_array_equal(mds.sa.police, range(3) * 2)
    # non-matching number of samples cannot be mapped
    assert_raises(ValueError, dm.forward, ds[:-1])
    # however, if the dataset knows about the space it is possible
    ds.sa['police'] = mds.sa.police
    # XXX this should be
    #mds2 = dm(ds[1:-1])
    #assert_array_equal(mds[1:-1], mds2)
    # XXX but right now is
    assert_raises(NotImplementedError, dm.forward, ds[1:-1])

    # Detrend must preserve the size of dataset
    assert_equal(mds.shape, ds.shape)

    # small additional test for break points
    # although they are no longer there
    ds = dataset_wizard(np.array([[1.0, 2, 3, 1, 2, 3]], ndmin=2).T,
                 targets=chunks, chunks=chunks)
    mds = PolyDetrendMapper(chunks_attr='chunks', polyord=1).forward(ds)
    assert_array_almost_equal(mds.samples, np.zeros(mds.shape))

    # test of different polyord on each chunk
    target_mixed = np.array( [[-1.0, 0, 1, 0, 0, 0],
                             [2.0, 0, -2, 0, 0, 0]], ndmin=2 ).T
    ds = dataset_wizard(samples_forchunks.copy(), targets=chunks, chunks=chunks)
    mds = PolyDetrendMapper(chunks_attr='chunks', polyord=[0,1]).forward(ds)
    assert_array_almost_equal(mds, target_mixed)

    # test irregluar spacing of samples, but with corrective time info
    samples_forwhole = np.array( [[1.0, 4, 6, 8, 2, 9],
                                 [-2.0, -8, -12, -16, -4, -18]], ndmin=2 ).T
    ds = Dataset(samples_forwhole, sa={'time': samples_forwhole[:,0]})
    # linear detrending that makes use of temporal info from dataset
    dm = PolyDetrendMapper(polyord=1, space='time')
    mds = dm.forward(ds)
    assert_array_almost_equal(mds.samples, np.zeros(mds.shape))

    # and now the same stuff, but with chunking and ordered by time
    samples_forchunks = np.array( [[1.0, 3, 3, 2, 2, 1],
                                  [-2.0, -6, -6, -4, -4, -2]], ndmin=2 ).T
    chunks = [0, 1, 0, 1, 0, 1]
    time = [4, 4, 12, 8, 8, 12]
    ds = Dataset(samples_forchunks.copy(), sa={'chunks': chunks, 'time': time})
    mds = PolyDetrendMapper(chunks_attr='chunks', polyord=1, space='time').forward(ds)

    # the whole thing must not affect the source data
    assert_array_equal(ds, samples_forchunks)
    # but if done inplace that is no longer true
    poly_detrend(ds, chunks_attr='chunks', polyord=1, space='time')
    assert_array_equal(ds, mds)
