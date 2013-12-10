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
from mvpa2.measures.winner import feature_winner_measure, \
                                  feature_loser_measure, \
                                  sample_winner_measure, \
                                  sample_loser_measure, \
                                  group_sample_winner_measure, \
                                  group_sample_loser_measure

from mvpa2.testing import assert_array_equal, assert_true

# if you need some classifiers
#from mvpa2.testing.clfs import *

class WinnerTests(unittest.TestCase):
    def test_winner(self):
        ns = 4
        nf = 3
        n = ns * nf
        ds = Dataset(np.reshape(np.mod(np.arange(0, n * 5, 5) + .5 * n, n), (ns, nf)),
                     sa=dict(targets=[0, 0, 1, 1], x=[3, 2, 1, 0]),
                     fa=dict(v=[3, 2, 1], w=['a', 'b', 'c']))

        measures2out = {feature_winner_measure : [1, 0, 2, 1],
                        feature_loser_measure: [2, 1, 0, 2],
                        sample_winner_measure: [1, 0, 2],
                        sample_loser_measure:[2, 1, 3],
                        group_sample_winner_measure:[0, 0, 0],
                        group_sample_loser_measure: [1, 0, 0]}

        for m, out in measures2out.iteritems():
            assert_array_equal(m()(ds).samples.ravel(), np.asarray(out))

def suite():  # pragma: no cover
    return unittest.makeSuite(WinnerTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

