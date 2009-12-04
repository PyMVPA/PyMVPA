# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for new Kernel-based SVMs"""

import unittest
import numpy as N
from mvpa.datasets import Dataset
from mvpa.clfs.libsvmc import SVM as lsSVM
from mvpa.clfs.sg import SVM as sgSVM

from tests_warehouse import datasets, sweepargs

class SVMKernelTests(unittest.TestCase):
    
    @sweepargs(clf=[lsSVM(), sgSVM()])
    def testBasicClfTrainPredict(self, clf):
        d = datasets['uni4medium']
        clf.train(d)
        clf.predict(d)
        pass

def suite():
    return unittest.makeSuite(KernelTests)


if __name__ == '__main__':
    import runner

