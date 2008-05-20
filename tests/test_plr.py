#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA logistic regression classifier"""

import unittest
from mvpa.datasets import Dataset
from mvpa.clfs.plr import PLR
import numpy as N
from mvpa.misc.data_generators import dumbFeatureBinaryDataset


class PLRTests(unittest.TestCase):

    def testPLR(self):
        data = dumbFeatureBinaryDataset()

        clf = PLR()

        clf.train(data)

        # prediction has to be perfect
        self.failUnless((clf.predict(data.samples) == data.labels).all())

    def testPLRState(self):
        data = dumbFeatureBinaryDataset()

        clf = PLR()

        clf.train(data)

        clf.states.enable('values')
        clf.states.enable('predictions')

        p = clf.predict(data.samples)

        self.failUnless((p == clf.predictions).all())
        self.failUnless(N.array(clf.values).shape == N.array(p).shape)


def suite():
    return unittest.makeSuite(PLRTests)


if __name__ == '__main__':
    import runner

