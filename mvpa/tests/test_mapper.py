# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for basic mappers'''

import numpy as N
# for repr
from numpy import array

from numpy.testing import assert_array_equal
from nose.tools import ok_, assert_raises, assert_false, assert_equal, \
        assert_true

from mvpa.mappers.flatten import FlattenMapper
from mvpa.mappers.base import FeatureSliceMapper, ChainMapper
from mvpa.support.copy import copy
from mvpa.datasets.base import Dataset
from mvpa.base.collections import ArrayCollectable

# arbitrary ndarray subclass for testing
class myarray(N.ndarray):
    pass

def test_flatten():
    samples_shape = (2, 2, 4)
    data_shape = (4,) + samples_shape
    data = N.arange(N.prod(data_shape)).reshape(data_shape).view(myarray)
    pristinedata = data.copy()
    target = [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
              [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
              [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]]
    target = N.array(target).view(myarray)
    index_target = N.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3],
                            [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3],
                            [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 0, 3],
                            [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3]])


    # array subclass survives
    ok_(isinstance(data, myarray))

    # actually, there should be no difference between a plain FlattenMapper and
    # a chain that only has a FlattenMapper as the one element
    for fm in [FlattenMapper(inspace='voxel'),
               ChainMapper([FlattenMapper(inspace='voxel'),
                            FeatureSliceMapper(slice(None))])]:
        # not working if untrained
        assert_raises(RuntimeError,
                      fm.forward1,
                      N.arange(N.sum(samples_shape) + 1))

        fm.train(data)

        ok_(isinstance(fm.forward(data), myarray))
        ok_(isinstance(fm.forward1(data[2]), myarray))
        assert_array_equal(fm.forward(data), target)
        assert_array_equal(fm.forward1(data[2]), target[2])
        assert_raises(ValueError, fm.forward, N.arange(4))

        # all of that leaves that data unmodified
        assert_array_equal(data, pristinedata)

        # reverse mapping
        ok_(isinstance(fm.reverse(target), myarray))
        ok_(isinstance(fm.reverse1(target[0]), myarray))
        ok_(isinstance(fm.reverse(target[1:2]), myarray))
        assert_array_equal(fm.reverse(target), data)
        assert_array_equal(fm.reverse1(target[0]), data[0])
        assert_array_equal(fm.reverse(target[1:2]), data[1:2])
        assert_raises(ValueError, fm.reverse, N.arange(14))

        # check one dimensional data, treated as scalar samples
        oned = N.arange(5)
        fm.train(Dataset(oned))
        # needs 2D
        assert_raises(ValueError, fm.forward, oned)
        # doesn't match mapper, since Dataset turns `oned` into (5,1)
        assert_raises(ValueError, fm.forward, oned)
        assert_equal(Dataset(oned).nfeatures, 1)

        # try dataset mode, with some feature attribute
        fattr = N.arange(N.prod(samples_shape)).reshape(samples_shape)
        ds = Dataset(data, fa={'awesome': fattr.copy()})
        assert_equal(ds.samples.shape, data_shape)
        fm.train(ds)
        dsflat = fm.forward(ds)
        ok_(isinstance(dsflat, Dataset))
        ok_(isinstance(dsflat.samples, myarray))
        assert_array_equal(dsflat.samples, target)
        assert_array_equal(dsflat.fa.awesome, N.arange(N.prod(samples_shape)))
        assert_true(isinstance(dsflat.fa['awesome'], ArrayCollectable))
        # test index creation
        assert_array_equal(index_target, dsflat.fa.voxel)

        # and back
        revds = fm.reverse(dsflat)
        ok_(isinstance(revds, Dataset))
        ok_(isinstance(revds.samples, myarray))
        assert_array_equal(revds.samples, data)
        assert_array_equal(revds.fa.awesome, fattr)
        assert_true(isinstance(revds.fa['awesome'], ArrayCollectable))
        assert_false('voxel' in revds.fa)



def test_subset():
    data = N.array(
            [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
            [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]])
    # float array doesn't work
    sm = FeatureSliceMapper(N.ones(16))
    assert_raises(IndexError, sm.forward, data)

    # full mask
    sm = FeatureSliceMapper(slice(None))
    # should not change single samples
    assert_array_equal(sm.forward(data[0:1].copy()), data[0:1])
    # or multi-samples
    assert_array_equal(sm.forward(data.copy()), data)
    sm.train(data)
    # same on reverse
    assert_array_equal(sm.reverse(data[0:1].copy()), data[0:1])
    # or multi-samples
    assert_array_equal(sm.reverse(data.copy()), data)

    ## test basic properties
    #assert_equal(sm.get_outsize(), 16)
    #assert_array_equal(sm.get_mask(), N.arange(16))

    # identical mappers
    sm_none = FeatureSliceMapper(slice(None))
    sm_int = FeatureSliceMapper(N.arange(16))
    sm_bool = FeatureSliceMapper(N.ones(16, dtype='bool'))
    sms = [sm_none, sm_int, sm_bool]

    # test subsets
    sids = [3,4,5,6]
    bsubset = N.zeros(16, dtype='bool')
    bsubset[sids] = True
    subsets = [sids, slice(3,7), bsubset, [3,3,4,4,6,6,6,5]]
    # all test subset result in equivalent masks, hence should do the same to
    # the mapper and result in identical behavior
    for st in sms:
        print st
        for i, sub in enumerate(subsets):
            # shallow copy
            orig = copy(st)
            subsm = FeatureSliceMapper(sub)
            # should do copy-on-write for all important stuff!!
            assert_true(orig.is_mergable(subsm))
            orig += subsm
            print orig
            # test if selection did its job
            print i
            if i == 3:
                # special case of multiplying features
                assert_array_equal(orig.forward1(data[0].copy()), subsets[i])
            else:
                assert_array_equal(orig.forward1(data[0].copy()), sids)

    ## all of the above shouldn't change the original mapper
    #assert_array_equal(sm.get_mask(), N.arange(16))

    # check for some bug catcher
    # no 3D input
    #assert_raises(IndexError, sm.forward, N.ones((3,2,1)))
    # no input of wrong length
    assert_raises(ValueError, sm.forward, N.ones(4))
    # same on reverse
    #assert_raises(ValueError, sm.reverse, N.ones(16))
    # invalid ids
    #assert_false(subsm.is_valid_inid(-1))
    #assert_false(subsm.is_valid_inid(16))


def test_repr():
    # this time give mask only by its target length
    sm = FeatureSliceMapper(slice(None), inspace='myspace')

    # check reproduction
    sm_clone = eval(repr(sm))
    assert_equal(repr(sm_clone), repr(sm))


def test_chainmapper():
    # the chain needs at lest one mapper
    assert_raises(ValueError, ChainMapper, [])
    # a typical first mapper is to flatten
    cm = ChainMapper([FlattenMapper()])

    # few container checks
    assert_equal(len(cm), 1)
    assert_true(isinstance(cm[0], FlattenMapper))

    # now training
    # come up with data
    samples_shape = (2, 2, 4)
    data_shape = (4,) + samples_shape
    data = N.arange(N.prod(data_shape)).reshape(data_shape)
    pristinedata = data.copy()
    target = [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
              [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
              [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]]
    target = N.array(target)

    # if it is not trained it knows nothing
    assert_equal(cm.get_outsize(), None)
    cm.train(data)
    assert_equal(cm.get_outsize(), 16)

    # a new mapper should appear when doing feature selection
    cm.append(FeatureSliceMapper(range(1,16)))
    assert_equal(cm.forward1(data[0]).shape, (15,))
    assert_equal(len(cm), 2)
    # multiple slicing
    cm.append(FeatureSliceMapper([9,14]))
    assert_equal(cm.forward1(data[0]).shape, (2,))
    assert_equal(len(cm), 3)

    # we cannot add a mapper with the wrong size
    #EXCEPTION HAS BEEN REMOVED, to get rid of get_outsize()
    #assert_raises(ValueError, cm.append, FeatureSubsetMapper(16))

    # check reproduction
    cm_clone = eval(repr(cm))
    assert_equal(repr(cm_clone), repr(cm))

    # what happens if we retrain the whole beast an same data as before
    cm.train(data)
    assert_equal(cm.forward1(data[0]).shape, (2,))
    assert_equal(len(cm), 3)

    # let's map something
    mdata = cm.forward(data)
    assert_array_equal(mdata, target[:,[10,15]])
    # and back
    rdata = cm.reverse(mdata)
    # original shape
    assert_equal(rdata.shape, data.shape)
    # content as far it could be restored
    assert_array_equal(rdata[rdata > 0], data[rdata > 0])
    assert_equal(N.sum(rdata > 0), 8)
