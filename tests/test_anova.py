#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA univariate ANOVA sensitivity analyzer."""

import unittest
import numpy as N

from mvpa.datasets.dataset import Dataset
from mvpa.algorithms.anova import OneWayAnova


def dumbFeatureDataset():
    data = [[0,1],[1,1],[0,2],[1,2],[0,3],[1,3],[0,4],[1,4],
            [0,5],[1,5],[0,6],[1,6],[0,7],[1,7],[0,8],[1,8],
            [0,9],[1,9],[0,10],[1,10],[0,11],[1,11],[0,12],[1,12]]
    regs = [1] * 8 + [2] * 8 + [3] * 8

    return Dataset(samples=data, labels=regs)



class ANOVATests(unittest.TestCase):

    def testANOVA(self):
        data = dumbFeatureDataset()
        aov = OneWayAnova()

        # compute f-scores
        f = aov(data)

        self.failUnless(f.shape == (2,))
        self.failUnless(f[0] == 0.0)
        self.failUnless(f[1] > 0.0)


def suite():
    return unittest.makeSuite(ANOVATests)


if __name__ == '__main__':
    import runner

