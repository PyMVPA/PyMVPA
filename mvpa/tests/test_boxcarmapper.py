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
from numpy.testing import assert_array_equal
from nose.tools import ok_, assert_raises, assert_false, assert_equal, \
        assert_true

from mvpa.mappers.boxcar import BoxcarMapper
from mvpa.datasets import Dataset


def test_simpleboxcar():
    """Just the same tests as for transformWithBoxcar.

    Mention that BoxcarMapper doesn't apply function with each boxcar
    """
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
    # should not work for shape mismatch
    assert_raises(ValueError, m.reverse, singlesample[0])

    # check broadcasting of 'raw' samples into proper boxcars on forward()
    bc = m.forward1(N.arange(24).reshape(3, 4, 2))
    assert_array_equal(bc, N.array(2 * [N.arange(24).reshape(3, 4, 2)]))
