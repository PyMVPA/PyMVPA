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

from numpy.testing import assert_array_equal
from nose.tools import ok_, assert_raises, assert_false, assert_equal, \
        assert_true

from mvpa.mappers.flatten import FlattenMapper
from mvpa.mappers.base import FeatureSubsetMapper
from mvpa.support.copy import copy

def test_flatten():
    samples_shape = (2, 2, 4)
    data_shape = (4,) + samples_shape
    data = N.arange(N.prod(data_shape)).reshape(data_shape)
    pristinedata = data.copy()
    target = [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
              [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
              [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]]
    target = N.array(target)

    fm = FlattenMapper()
    # not working if untrained
    assert_raises(RuntimeError, fm.forward, N.arange(N.sum(samples_shape) + 1))

    fm.train(data)

    assert_array_equal(fm.forward(data), target)
    assert_array_equal(fm.forward(data[2]), target[2])
    assert_raises(ValueError, fm.forward, N.arange(4))
    assert_raises(ValueError, fm.forward, N.array(data[0], order='F'))

    # check coordinate2id conversion
    got_coord = N.zeros((N.prod(samples_shape)), dtype='bool')
    for i in xrange(samples_shape[0]):
        for j in xrange(samples_shape[1]):
            for k in xrange(samples_shape[2]):
                id = fm.get_outids([(i, j, k)])[0]
                assert_equal(len(id), 1)
                # mark that we got this particular ID
                got_coord[id[0]] = True
    # each id has to have occurred
    ok_(got_coord.all())
    # invalid caught?
    assert_raises(ValueError, fm.get_outids, samples_shape)
    assert_raises(ValueError, fm.get_outids, (0,0,0,0))

    # all of that leaves that data unmodified
    assert_array_equal(data, pristinedata)

    # reverse mapping
    assert_array_equal(fm.reverse(target), data)
    assert_array_equal(fm.reverse(target[0]), data[0])
    assert_array_equal(fm.reverse(target[1:2]), data[1:2])
    assert_raises(ValueError, fm.reverse, N.arange(14))
    assert_raises(ValueError, fm.reverse, N.array(target, order='F'))

    # check one dimensional data, treated as scalar samples
    oned = N.arange(5)
    oned_target = [[0],[1],[2],[3],[4]]
    fm.train(oned)
    assert_array_equal(fm.forward(oned), oned_target)
    assert_array_equal(fm.reverse(N.array(oned_target)), oned)
    assert_equal(fm.get_outids([0])[0], [0])
    assert_raises(ValueError, fm.get_outids, [5])
    assert_raises(ValueError, fm.get_outids, [(0,0)])

    # check one dimensional samples
    oneds = N.array([range(5)])
    oneds_target = [[0, 1, 2, 3, 4]]
    fm.train(oneds)
    assert_array_equal(fm.forward(oneds), oneds_target)
    assert_array_equal(fm.forward(oneds[0]), oneds_target[0])
    assert_array_equal(fm.reverse(N.array(oneds_target)), oneds)
    assert_array_equal(fm.reverse(N.array(oneds_target[0])), oneds[0])
    assert_equal(fm.get_outids([0])[0], [0])
    assert_equal(fm.get_outids([3])[0], [3])
    assert_raises(ValueError, fm.get_outids, [5])
    assert_raises(ValueError, fm.get_outids, [(0,0)])




def test_subset():
    data = N.array(
            [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
            [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]])
    # full mask
    sm = FeatureSubsetMapper(N.ones(16))
    # should not change single samples
    assert_array_equal(sm.forward(data[0].copy()), data[0])
    # or multi-samples
    assert_array_equal(sm.forward(data.copy()), data)
    # same on reverse
    assert_array_equal(sm.reverse(data[0].copy()), data[0])
    # or multi-samples
    assert_array_equal(sm.reverse(data.copy()), data)

    # test basic properties
    assert_equal(sm.get_outsize(), 16)
    assert_array_equal(sm.get_mask(), N.ones(16))

    # id transformation
    for id in range(16):
        ok_(sm.is_valid_inid(id))
        ok_(sm.is_valid_outid(id))  
        assert_equal(sm.get_outids([id])[0], [id])

    # test subsets
    sids = [3,4,5,6]
    bsubset = N.zeros(16, dtype='bool')
    bsubset[sids] = True
    subsets = [sids, slice(3,7), bsubset, [3,3,4,4,6,6,6,5]]
    # all test subset result in equivalent masks, hence should do the same to
    # the mapper and result in identical behavior
    for sub in subsets:
        # shallow copy
        subsm = copy(sm)
        # should do copy-on-write for all important stuff!!
        subsm.select_out(sub)
        # test if selection did its job
        assert_array_equal(subsm.get_mask(), bsubset)
        assert_equal(subsm.get_outsize(), 4)
        assert_array_equal([subsm.is_valid_inid(i) for i in range(16)], bsubset)
        assert_array_equal([subsm.is_valid_outid(i) for i in range(16)],
                           [True] * 4 + [False] * 12)
        assert_array_equal(subsm.forward(data[0].copy()), sids)

    # all of the above shouldn't change the original mapper
    assert_array_equal(sm.get_mask(), N.ones(16))
    # but without COW it should be affected as well
    subsm = copy(sm)
    subsm.select_out(sids, cow=False)
    assert_equal(subsm.get_outsize(), 4)
    assert_array_equal(sm.get_mask(copy=False), subsm.get_mask(copy=False))
    # however, the original mapper is now invalid, hence only for internal use
    assert_equal(sm.get_outsize(), 16)

    # check for some bug catchers
    assert_raises(ValueError, FeatureSubsetMapper, N.ones((2,1)))
    # no 3D input
    assert_raises(ValueError, subsm.forward, N.ones((3,2,1)))
    # no input of wrong length
    assert_raises(ValueError, subsm.forward, N.ones(4))
    # same on reverse
    assert_raises(ValueError, subsm.reverse, N.ones(16))
    # invalid ids
    assert_false(subsm.is_valid_inid(-1))
    assert_false(subsm.is_valid_inid(16))
    assert_raises(ValueError, subsm.get_outids, [16])


def test_coordspaces():
    sm = FeatureSubsetMapper(N.ones(5), inspace='myspace')

    # id transformation
    # works without spaces (see above) should still work with them
    for id in range(5):
        ok_(sm.is_valid_inid(id))
        ok_(sm.is_valid_outid(id))
        assert_equal(sm.get_outids([id])[0], [id])
    # we should achieve them same when providing the same information as space
    # specific coords
    for id in range(5):
        ok_(sm.is_valid_inid(id))
        ok_(sm.is_valid_outid(id))
        # space coord can be anything, whatever a mapper might digest
        out = sm.get_outids(myspace=[id], someother="blabla")
        assert_equal(out[0], [id])
        # space coords should be processed and no longer be part of the dicts
        assert_false(out[1].has_key('myspace'))
        assert_true(out[1].has_key('someother'))
    # however, if the mapper doesn't know about a space, it should leave it
    # alone and confess that there is nothing it can do
    for id in range(5):
        out = sm.get_outids(unknown=[id])
        assert_equal(out[0], [])
        assert_true(out[1].has_key('unknown'))
