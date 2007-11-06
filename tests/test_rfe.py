#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Unit tests for PyMVPA recursive feature elimination"""

import unittest
import numpy as N

from mvpa.datasets.maskeddataset import MaskedDataset
from mvpa.clf.rfe import RFE, StopNBackHistoryCriterion
#from mvpa.clf.svm import SVM


class RFETests(unittest.TestCase):

    def testStopCriterion(self):
        stopcrit = StopNBackHistoryCriterion()
        # for empty history -- no best but just go
        self.failUnless(stopcrit([]) == (False, False))
        # we got the best if we have just 1
        self.failUnless(stopcrit([1]) == (False, True))
        # we got the best if we have the last minimal
        self.failUnless(stopcrit([1, 0.9, 0.8]) == (False, True))
        # should not stop if we got 10 more after minimal
        self.failUnless(stopcrit(
            [1, 0.9, 0.8]+[0.9]*(stopcrit.steps-1)) == (False, False))
        # should stop if we got 10 more after minimal
        self.failUnless(stopcrit(
            [1, 0.9, 0.8]+[0.9]*stopcrit.steps) == (True, False))

        # test for alternative func
        stopcrit = StopNBackHistoryCriterion(func=max)
        self.failUnless(stopcrit([0.8, 0.9, 1.0]) == (False, True))
        self.failUnless(stopcrit([0.8, 0.9, 1.0]+[0.9]*9) == (False, False))
        self.failUnless(stopcrit([0.8, 0.9, 1.0]+[0.9]*10) == (True, False))

        

def suite():
    return unittest.makeSuite(RFETests)


if __name__ == '__main__':
    import test_runner

