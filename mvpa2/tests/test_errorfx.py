# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for various use cases users reported mis-behaving"""

import unittest
import numpy as np

from mvpa2.testing.tools import ok_, assert_array_equal, assert_true, \
        assert_false, assert_equal, assert_not_equal, reseed_rng

from mvpa2.misc.errorfx import auc_error

def test_auc_error():
    # two basic cases
    # perfect
    assert_equal(auc_error([-1, -1, 1, 1], [0, 0, 1, 1]), 1)
    # anti-perfect
    assert_equal(auc_error([-1, -1, 1, 1], [1, 1, 0, 0]), 0)

    # chance -- we aren't taking care ATM about randomly broken
    # ties, e.g. if both labels have the same estimate :-/
    # TODO:
    #assert_equal(auc_error([-1, 1, -1, 1], [0, 0, 1, 1]), 0.5)
