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
from mvpa.datasets.splitter import NFoldSplitter
from mvpa.measures.splitsensitivity import SplitSensitivity

from tests_warehouse import *
from tests_warehouse_clfs import *


class SplitSensitivityAnalyserTests(unittest.TestCase):

    def setUp(self):
        self.dataset = normalFeatureDataset(perlabel=50, nlabels=2, nfeatures=4)


    if clfs.has_key('LinearSVMC'):
        @sweepargs(svm=clfs['LinearSVMC'])
        def testAnalyzer(self, svm):
            svm_weigths = svm.getSensitivityAnalyzer()

            sana = SplitSensitivity(
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
    import runner

