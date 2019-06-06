# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA dense array mapper"""


from mvpa2.mappers.flatten import mask_mapper
from mvpa2.featsel.base import StaticFeatureSelection

from mvpa2.testing.tools import assert_raises, assert_equal, assert_array_equal

import numpy as np

def test_forward_dense_array_mapper():
    mask = np.ones((3,2), dtype='bool')
    map_ = mask_mapper(mask)

    # test shape reports
    assert_equal(map_.forward1(mask).shape, (6,))

    # test 1sample mapping
    assert_array_equal(map_.forward1(np.arange(6).reshape(3,2)),
                       [0,1,2,3,4,5])

    # test 4sample mapping
    foursample = map_.forward(np.arange(24).reshape(4,3,2))
    assert_array_equal(foursample,
                       [[0,1,2,3,4,5],
                        [6,7,8,9,10,11],
                        [12,13,14,15,16,17],
                        [18,19,20,21,22,23]])

    # check incomplete masks
    mask[1,1] = 0
    map_ = mask_mapper(mask)
    assert_equal(map_.forward1(mask).shape, (5,))
    assert_array_equal(map_.forward1(np.arange(6).reshape(3,2)),
                       [0,1,2,4,5])

    # check that it doesn't accept wrong dataspace
    assert_raises(ValueError, map_.forward, np.arange(4).reshape(2,2))

    # check fail if neither mask nor shape
    assert_raises(ValueError, mask_mapper)

    # check that a full mask is automatically created when providing shape
    m = mask_mapper(shape=(2, 3, 4))
    mp = m.forward1(np.arange(24).reshape(2, 3, 4))
    assert_array_equal(mp, np.arange(24))


def test_reverse_dense_array_mapper():
    mask = np.ones((3,2), dtype='bool')
    mask[1,1] = 0
    map_ = mask_mapper(mask)

    rmapped = map_.reverse1(np.arange(1,6))
    assert_equal(rmapped.shape, (3,2))
    assert_equal(rmapped[1,1], 0)
    assert_equal(rmapped[2,1], 5)


    # check that it doesn't accept wrong dataspace
    assert_raises(ValueError, map_.forward, np.arange(6))

    rmapped2 = map_.reverse(np.arange(1,11).reshape(2,5))
    assert_equal(rmapped2.shape, (2,3,2))
    assert_equal(rmapped2[0,1,1], 0 )
    assert_equal(rmapped2[1,1,1], 0 )
    assert_equal(rmapped2[0,2,1], 5 )
    assert_equal(rmapped2[1,2,1], 10 )


def test_mapper_aliases():
    mm=mask_mapper(np.ones((3,4,2), dtype='bool'))
    assert_array_equal(mm.forward(np.ones((2,3,4,2))),
                       mm.forward(np.ones((2,3,4,2))))


def test_selects():
    mask = np.ones((3,2), dtype='bool')
    mask[1,1] = 0
    mask0 = mask.copy()
    data = np.arange(6).reshape(mask.shape)
    map_ = mask_mapper(mask)

    # check if any exception is thrown if we get
    # out of the outIds
    #assert_raises(IndexError, map_.select_out, [0,1,2,6])

    # remove 1,2
    map_.append(StaticFeatureSelection([0,3,4]))
    assert_array_equal(map_.forward1(data), [0, 4, 5])
    # remove 1 more
    map_.append(StaticFeatureSelection([0,2]))
    assert_array_equal(map_.forward1(data), [0, 5])

    # check if original mask wasn't perturbed
    assert_array_equal(mask, mask0)

    # check if original mask wasn't perturbed
    assert_array_equal(mask, mask0)
