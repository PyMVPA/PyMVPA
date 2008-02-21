#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA classifier cross-validation"""

import unittest
import numpy as N

from mvpa.datasets.dataset import Dataset
from mvpa.clfs.knn import kNN
from mvpa.datasets.splitter import NFoldSplitter
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.clfs.transerror import TransferError

from tests_warehouse import pureMultivariateSignal, getMVPattern

class CrossValidationTests(unittest.TestCase):


    def testSimpleNMinusOneCV(self):
        data = getMVPattern(3)

        self.failUnless( data.nsamples == 120 )
        self.failUnless( data.nfeatures == 2 )
        self.failUnless(
            (data.labels == [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]*6 ).all() )
        self.failUnless(
            (data.chunks == \
                [ k for k in range(1,7) for i in range(20) ] ).all() )

        transerror = TransferError(kNN())
        cv = CrossValidatedTransferError(transerror,
                                         NFoldSplitter(cvtype=1))

        results = cv(data)
        self.failUnless( results < 0.2 and results >= 0.0 )


    def testNoiseClassification(self):
        # get a dataset with a very high SNR
        data = getMVPattern(10)

        # do crossval with default errorfx and 'mean' combiner
        transerror = TransferError(kNN())
        cv = CrossValidatedTransferError(transerror, NFoldSplitter(cvtype=1)) 

        # must return a scalar value
        result = cv(data)

        # must be perfect
        self.failUnless( result < 0.05 )

        # do crossval with permuted regressors
        cv = CrossValidatedTransferError(transerror,
                  NFoldSplitter(cvtype=1, permute=True, nrunspersplit=10) )
        results = cv(data)

        # must be at chance level
        pmean = N.array(results).mean()
        self.failUnless( pmean < 0.58 and pmean > 0.42 )


def suite():
    return unittest.makeSuite(CrossValidationTests)


if __name__ == '__main__':
    import test_runner

