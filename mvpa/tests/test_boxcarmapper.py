# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Boxcar mapper"""


import numpy as N

from mvpa.testing.tools import ok_, assert_raises, assert_false, assert_equal, \
        assert_true, assert_array_equal

from mvpa.mappers.boxcar import BoxcarMapper
from mvpa.datasets import Dataset


def test_simpleboxcar():
    data = N.atleast_2d(N.arange(10)).T
    sp = N.arange(10)

    # check if stupid thing don't work
    assert_raises(ValueError, BoxcarMapper, sp, 0)

    # now do an identity transformation
    bcm = BoxcarMapper(sp, 1)
    trans = bcm.forward(data)
    # ,0 is a feature below, so we get explicit 2D out of 1D
    assert_array_equal(trans[:,0], data)

    # now check for illegal boxes
    if __debug__:
        # condition is checked only in __debug__
        assert_raises(ValueError, BoxcarMapper(sp, 2).train, data)

    # now something that should work
    nbox = 9
    boxlength = 2
    sp = N.arange(nbox)
    bcm = BoxcarMapper(sp, boxlength)
    trans = bcm(data)
    # check that is properly upcasts the dimensionality
    assert_equal(trans.shape, (nbox, boxlength) + data.shape[1:])
    # check actual values, squeezing the last dim for simplicity
    assert_array_equal(trans.squeeze(), N.vstack((N.arange(9), N.arange(9)+1)).T)


    # now test for proper data shape
    data = N.ones((10,3,4,2))
    sp = [ 2, 4, 3, 5 ]
    trans = BoxcarMapper(sp, 4)(data)
    assert_equal(trans.shape, (4,4,3,4,2))

    # test reverse
    data = N.arange(240).reshape(10, 3, 4, 2)
    sp = [ 2, 4, 3, 5 ]
    boxlength = 2
    m = BoxcarMapper(sp, boxlength)
    m.train(data)
    mp = m.forward(data)
    assert_equal(mp.shape, (4, 2, 3, 4, 2))

    # try full reconstruct
    mr = m.reverse(mp)
    # shape has to match
    assert_equal(mr.shape, (len(sp) * boxlength,) + data.shape[1:])
    # only known samples are part of the results
    assert_true((mr >= 24).all())
    assert_true((mr < 168).all())

    # check proper reconstruction of non-conflicting sample
    assert_array_equal(mr[0].ravel(), N.arange(48, 72))

    # check proper reconstruction of samples being part of multiple
    # mapped samples
    assert_array_equal(mr[1].ravel(), N.arange(72, 96))

    # test reverse of a single sample
    singlesample = N.arange(48).reshape(2, 3, 4, 2)
    assert_array_equal(singlesample, m.reverse1(singlesample))
    # should not work for shape mismatch, but it does work and is useful when
    # reverse mapping sample attributes
    #assert_raises(ValueError, m.reverse, singlesample[0])

    # check broadcasting of 'raw' samples into proper boxcars on forward()
    bc = m.forward1(N.arange(24).reshape(3, 4, 2))
    assert_array_equal(bc, N.array(2 * [N.arange(24).reshape(3, 4, 2)]))


def test_datasetmapping():
    # 6 samples, 4 features
    data = N.arange(24).reshape(6,4)
    ds = Dataset(data,
                 sa={'timepoints': N.arange(6),
                     'multidim': data.copy()},
                 fa={'fid': N.arange(4)})
    # with overlapping and non-overlapping boxcars
    startpoints = [0, 1, 4]
    boxlength = 2
    bm = BoxcarMapper(startpoints, boxlength, inspace='boxy')
    # train is critical
    bm.train(ds)
    mds = bm.forward(ds)
    assert_equal(len(mds), len(startpoints))
    assert_equal(mds.nfeatures, boxlength)
    # all samples attributes remain, but the can rotated/compressed into
    # multidimensional attributes
    assert_equal(sorted(mds.sa.keys()), ['boxy_onsetidx'] + sorted(ds.sa.keys()))
    assert_equal(mds.sa.multidim.shape,
                 (len(startpoints), boxlength, ds.nfeatures))
    assert_equal(mds.sa.timepoints.shape, (len(startpoints), boxlength))
    assert_array_equal(mds.sa.timepoints.flatten(),
                       N.array([(s, s+1) for s in startpoints]).flatten())
    assert_array_equal(mds.sa.boxy_onsetidx, startpoints)
    # feature attributes also get rotated and broadcasted
    assert_array_equal(mds.fa.fid, [ds.fa.fid, ds.fa.fid])
    # and finally there is a new one
    assert_array_equal(mds.fa.boxy_offsetidx,
                       N.repeat(N.arange(boxlength), 4).reshape(2,-1))

    # now see how it works on reverse()
    rds = bm.reverse(mds)
    # we got at least something of all original attributes back
    assert_equal(sorted(rds.sa.keys()), sorted(ds.sa.keys()))
    assert_equal(sorted(rds.fa.keys()), sorted(ds.fa.keys()))
    # it is not possible to reconstruct the full samples array
    # some samples even might show up multiple times (when there are overlapping
    # boxcars
    assert_array_equal(rds.samples,
                       N.array([[ 0,  1,  2,  3],
                                [ 4,  5,  6,  7],
                                [ 4,  5,  6,  7],
                                [ 8,  9, 10, 11],
                                [16, 17, 18, 19],
                                [20, 21, 22, 23]]))
    assert_array_equal(rds.sa.timepoints, [0, 1, 1, 2, 4, 5])
    assert_array_equal(rds.sa.multidim, ds.sa.multidim[rds.sa.timepoints])
    # but feature attributes should be fully recovered
    assert_array_equal(rds.fa.fid, ds.fa.fid)
