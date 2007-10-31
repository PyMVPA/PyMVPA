#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Unit tests for Feature Selection Validation"""

import mvpa.ifs as ifs
import mvpa.featsel
import mvpa.maskeddataset
import mvpa.svm as svm
import unittest
import numpy as N

class FeatSelValidationTests(unittest.TestCase):

    def setUp(self):
        # prepare demo dataset and mask
        self.mask = N.ones((3,3,3))
        data = N.random.normal(0,1,(100,) + (self.mask.shape))
        reg = N.arange(100) / 50
        orig = range(5) * 20
        self.pattern = mvpa.maskeddataset.MaskedDataset(data, reg, orig)


    def testIncrementalSearch(self):
        fselect = ifs.IFS( verbose = True)
        clf = svm.SVM()

        fsv = mvpa.featsel.FeatSelValidation( self.pattern )

        fsv( fselect, clf )

        self.failUnless (len(fsv.selections) == len(fsv.perfs) == 5)
        # pattern is noise, so generalization should be chance
        self.failUnless(0.35 < N.mean(fsv.perfs) < 0.65)

        mrm = fsv.getMeanRatingMap()

        self.failUnless(mrm.shape == self.mask.shape)
        self.failUnless( (mrm != 0).all() )


def suite():
    return unittest.makeSuite(FeatSelValidationTests)


if __name__ == '__main__':
    import test_runner

