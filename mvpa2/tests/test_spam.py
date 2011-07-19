# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SpAM."""

from mvpa.misc import data_generators

from mvpa.testing import *
from mvpa.testing.datasets import datasets


def test_basic():
    """Disabled the test since we have no SpAM yet
    """
    raise SkipTest, "skipped SPaM since no SPaM"
    from mvpa.clfs.spam import SpAM

    dataset = datasets['sin_modulated']
    clf = SpAM()
    clf.train(dataset)
    y = clf.predict(dataset.samples)
    assert_array_equal(y.shape, dataset.targets.shape)

