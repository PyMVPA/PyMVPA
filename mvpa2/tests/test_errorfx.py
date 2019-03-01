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

from mvpa2.testing.tools import (
    assert_false, assert_equal, assert_almost_equal, reseed_rng,
    assert_raises,
)

from mvpa2.misc.errorfx import (
    auc_error,
    mean_tpr, mean_fnr,
    mean_match_accuracy,
)

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


@reseed_rng()
def test_mean_tpr_balanced():
    # in case of the balanced sets we should expect to match mean_match_accuracy
    for nclass in range(2, 4):
        for nsample in range(1, 3):
            target = np.repeat(np.arange(nclass), nsample)
            # perfect match
            assert_equal(mean_match_accuracy(target, target), 1.0)
            assert_equal(mean_tpr(target, target), 1.0)
            # perfect mismatch -- shift by nsample, so no target matches
            estimate = np.roll(target, nsample)
            assert_equal(mean_match_accuracy(target, estimate), 0)
            assert_equal(mean_tpr(target, estimate), 0)
            # do few permutations and see if both match
            for i in range(5):
                np.random.shuffle(estimate)
                assert_equal(
                    mean_tpr(target, estimate),
                    mean_match_accuracy(target, estimate))
                assert_almost_equal(
                    mean_tpr(target, estimate), 1-mean_fnr(target, estimate))


def test_mean_tpr():
    # Let's test now on some disbalanced sets
    assert_raises(ValueError, mean_tpr, [1], [])
    assert_raises(ValueError, mean_tpr, [], [1])
    assert_raises(ValueError, mean_tpr, [], [])

    # now interesting one where there were no target when it was in predicted
    assert_raises(ValueError, mean_tpr, [1], [0])
    assert_raises(ValueError, mean_tpr, [0, 1], [0, 0])
    # but it should be ok to have some targets not present in prediction
    assert_equal(mean_tpr([0, 0], [0, 1]), .5)
    # the same regardless how many samples in 0-class, if all misclassified
    # (winner by # of samples takes all)
    assert_equal(mean_tpr([0, 0, 0], [0, 0, 1]), .5)
    # whenever mean-accuracy would be different
    assert_almost_equal(mean_match_accuracy([0, 0, 0], [0, 0, 1]), 2/3.)
