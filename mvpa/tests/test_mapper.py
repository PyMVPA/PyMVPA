# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for baisc mappers'''

import numpy as N

from numpy.testing import assert_array_equal
from nose.tools import ok_, assert_raises, assert_false, assert_equal

from mvpa.mappers.flatten import FlattenMapper


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
                id = fm.get_outid((i, j, k))
                # mark that we got this particular ID
                got_coord[id] = True
    # each id has to have occurred
    ok_(got_coord.all())
    # invalid caught?
    assert_raises(ValueError, fm.get_outid, samples_shape)
    assert_raises(ValueError, fm.get_outid, (0,0,0,0))

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
    assert_equal(fm.get_outid(0), 0)
    assert_raises(ValueError, fm.get_outid, 5)
    assert_raises(ValueError, fm.get_outid, (0,0))

    # check one dimensional samples
    oneds = N.array([range(5)])
    oneds_target = [[0, 1, 2, 3, 4]]
    fm.train(oneds)
    assert_array_equal(fm.forward(oneds), oneds_target)
    assert_array_equal(fm.forward(oneds[0]), oneds_target[0])
    assert_array_equal(fm.reverse(N.array(oneds_target)), oneds)
    assert_array_equal(fm.reverse(N.array(oneds_target[0])), oneds[0])
    assert_equal(fm.get_outid(0), 0)
    assert_equal(fm.get_outid(3), 3)
    assert_raises(ValueError, fm.get_outid, 5)
    assert_raises(ValueError, fm.get_outid, (0,0))



