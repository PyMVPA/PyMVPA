#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA least angle regression (LARS) classifier"""

import unittest
from mvpa.datasets import Dataset
from mvpa.clfs.lars import LARS
import numpy as N
from scipy.stats import pearsonr
from mvpa.misc.data_generators import dumbFeatureBinaryDataset

class LARSTests(unittest.TestCase):

    def testLARS(self):
        # not the perfect dataset with which to test, but
        # it will do for now.
        data = dumbFeatureBinaryDataset()

        clf = LARS()

        clf.train(data)

        # prediction has to be almost perfect
        # test with a correlation
        pre = clf.predict(data.samples)
        cor = pearsonr(pre,data.labels)
        self.failUnless(cor[0] > .8)

    def testLARSState(self):
        data = dumbFeatureBinaryDataset()

        clf = LARS()

        clf.train(data)

        clf.states.enable('predictions')

        p = clf.predict(data.samples)

        self.failUnless((p == clf.predictions).all())


def suite():
    return unittest.makeSuite(LARSTests)


if __name__ == '__main__':
    import runner

