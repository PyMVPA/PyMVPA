#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SplittingSensitivityAnalyzer"""

import unittest

import numpy as N

from mvpa.datasets.dataset import Dataset
from mvpa.algorithms.linsvmweights import LinearSVMWeights
from mvpa.clfs.svm import LinearNuSVMC
from mvpa.datasets.splitter import NFoldSplitter
from mvpa.algorithms.splitsensana import SplittingSensitivityAnalyzer

from tests_warehouse import *

class SplitSensitivityAnalyserTests(unittest.TestCase):

    def setUp(self):
        self.dataset = normalFeatureDataset(perlabel=50, nlabels=2, nfeatures=4)


    def testAnalyzer(self):
        svm = LinearNuSVMC()
        svm_weigths = LinearSVMWeights(svm)

        sana = SplittingSensitivityAnalyzer(
                    svm_weigths,
                    NFoldSplitter(cvtype=1),
                    enable_states=['maps'])

        maps = sana(self.dataset)

        self.failUnless(len(maps) == 4)
        self.failUnless(sana.states.isKnown('maps'))
        allmaps = N.array(sana.maps)
        self.failUnless(allmaps[:,0].mean() == maps[0])
        self.failUnless(allmaps.shape == (5,4))


def suite():
    return unittest.makeSuite(SplitSensitivityAnalyserTests)


if __name__ == '__main__':
    import test_runner

