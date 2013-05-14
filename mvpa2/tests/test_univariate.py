# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA searchlight algorithm"""

import unittest
import numpy as np

from mvpa2.base import cfg
from mvpa2.datasets.base import Dataset

from mvpa2.testing import sweepargs, reseed_rng, assert_equal, \
                            assert_array_equal, assert_true
#from mvpa2.datasets import Dataset
from mvpa2.measures.univariate import compound_winner_take_all_measure, \
                                      compound_mean_measure, \
                                      compound_univariate_mean_measure, \
                                      WinnerTakeAllMeasure

# if you need some classifiers
#from mvpa2.testing.clfs import *

class UnivariateTests(unittest.TestCase):
    def test_univariate(self):
        ns = 4
        nf = 3

        ds = Dataset(np.reshape(np.arange(ns * nf), (ns, nf)),
                     sa=dict(targets=[0, 0, 1, 1], x=[3, 2, 1, 0]))

        for i, sign in enumerate([-1, 1]):

            m = compound_winner_take_all_measure(sign)
            assert_array_equal(m(ds).samples, i)

            assert_true(m.is_trained)

        m = compound_univariate_mean_measure()
        assert_array_equal(m(ds).samples, [[1.5, 2.5, 3.5], [7.5, 8.5, 9.5]])

        m = compound_mean_measure()
        assert_array_equal(m(ds).samples, [[2.5], [8.5]])

        m = WinnerTakeAllMeasure()
        assert_array_equal(m(ds).samples, [[3, 3, 3]])

        assert_array_equal(m(ds).fa.wta_targets, [1, 1, 1])

def suite():
    return unittest.makeUnivariateTests(HyperAlignmentTests)


if __name__ == '__main__':
    import runner

