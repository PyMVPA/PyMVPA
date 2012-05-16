# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for complementary unittest-ing tools"""

import numpy as np

from mvpa2.testing.tools import *


def test_assert_objectarray_equal():
    # explicit dtype so we could test with numpy < 1.6
    a = np.array([np.array([0, 1]), np.array(1)], dtype=object)
    b = np.array([np.array([0, 1]), np.array(1)], dtype=object)

    # good with self
    assert_objectarray_equal(a, a)
    # good with a copy
    assert_objectarray_equal(a, a.copy())
    # good while operating with an identical one
    # see http://projects.scipy.org/numpy/ticket/2117
    assert_objectarray_equal(a, b)

    # now check if we still fail for a good reason
    for b in (
            np.array(1),
            np.array([1]),
            np.array([np.array([0, 1]), np.array((1, 2))], dtype=object),
            np.array([np.array([0, 1]), np.array(1.1)], dtype=object),
            np.array([np.array([0, 1]), np.array(1.0)], dtype=object),
            np.array([np.array([0, 1]), np.array(1, dtype=object)], dtype=object),
            ):
        assert_raises(AssertionError, assert_objectarray_equal, a, b)
