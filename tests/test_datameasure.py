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
from mvpa.algorithms.featsel import FixedNElementTailSelector
from mvpa.algorithms.linsvmweights import LinearSVMWeights
from mvpa.clfs import *
from mvpa.clfs.svm import LinearNuSVMC
from mvpa.datasets.splitter import NFoldSplitter
from mvpa.algorithms.datameasure import *

from tests_warehouse import *

class SensitivityAnalysersTests(unittest.TestCase):

    def setUp(self):
        self.nonbogus = [1, 3]          # informative features
        self.nfeatures = 5              # total features
        means = N.zeros((2,self.nfeatures))
        # pure multivariate -- single bit per feature
        for i in xrange(len(self.nonbogus)):
            means[i, self.nonbogus[i]] = 1.0
        self.dataset = normalFeatureDataset(perlabel=50, nlabels=2,
                                            nfeatures=self.nfeatures,
                                            means=means, snr=3.0)


    def testBIG(self):
        svm = LinearNuSVMC()

        #svm_weigths = LinearSVMWeights(svm)

        # assumming many defaults it is as simple as
        sana = selectAnalyzer( SplitClassifier(clf=svm) )
        # and we get sensitivity analyzer which works on splits and uses
        # linear svm sensitivity
        map_ = sana(self.dataset)
        print `sana`
        self.failUnless(len(map_) == self.dataset.nfeatures)
        self.failUnless(sana.clf["trained_confusion"].percentCorrect>90,
                        msg="We must have trained more or less correctly")
        self.failUnlessEqual(list(FixedNElementTailSelector(self.nfeatures - len(self.nonbogus))(map_)),
                             list(self.nonbogus),
                             msg="At the end we should have selected the right features")


def suite():
    return unittest.makeSuite(SensitivityAnalysersTests)


if __name__ == '__main__':
    import test_runner

