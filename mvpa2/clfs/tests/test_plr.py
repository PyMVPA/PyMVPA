# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA logistic regression classifier"""

import numpy as np

from mvpa2.clfs.plr import PLR
from mvpa2.testing import *
from mvpa2.testing.datasets import datasets


class PLRTests(unittest.TestCase):

    def test_plr(self):
        data = datasets['dumb2']

        clf = PLR()

        clf.train(data)

        # prediction has to be perfect
        self.assertTrue((clf.predict(data.samples) == data.targets).all())

    def test_plr_state(self):
        data = datasets['dumb2']

        clf = PLR()

        clf.train(data)
        # Also get "sensitivity".  Was introduced to check a bug with
        # processing dataset with numeric labels
        sa = clf.get_sensitivity_analyzer()
        sens = sa(data)

        clf.ca.enable('estimates')
        clf.ca.enable('predictions')

        p = clf.predict(data.samples)

        self.assertTrue((p == clf.ca.predictions).all())
        self.assertTrue(np.array(clf.ca.estimates).shape == np.array(p).shape)


def suite():  # pragma: no cover
    return unittest.makeSuite(PLRTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

