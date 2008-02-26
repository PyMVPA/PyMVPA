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
from mvpa.datasets.dataset import Dataset
from mvpa.clfs.plr import PLR
import numpy as N

def dumbFeatureDataset():
    data = [[1,0],[1,1],[2,0],[2,1],[3,0],[3,1],[4,0],[4,1],
            [5,0],[5,1],[6,0],[6,1],[7,0],[7,1],[8,0],[8,1],
            [9,0],[9,1],[10,0],[10,1],[11,0],[11,1],[12,0],[12,1]]
    regs = [0 for i in range(12)] \
         + [1 for i in range(12)]

    return Dataset(samples=data, labels=regs)

class PLRTests(unittest.TestCase):

    def testPLR(self):
        data = dumbFeatureDataset()

        clf = PLR()

        clf.train(data)

        # prediction has to be perfect
        self.failUnless((clf.predict(data.samples) == data.labels).all())

    def testPLRState(self):
        data = dumbFeatureDataset()

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
    import test_runner

