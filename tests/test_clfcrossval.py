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
from mvpa.algorithms.clfcrossval import ClfCrossValidation
from mvpa.clfs.transerror import TransferError, ConfusionMatrix

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
        cv = ClfCrossValidation(transerror,
                                NFoldSplitter(cvtype=1))

        results = cv(data)
        self.failUnless( results < 0.2 and results >= 0.0 )


    def testNoiseClassification(self):
        # get a dataset with a very high SNR
        data = getMVPattern(10)

        # do crossval with default errorfx and 'mean' combiner
        transerror = TransferError(kNN())
        cv = ClfCrossValidation(transerror, NFoldSplitter(cvtype=1)) 

        # must return a scalar value
        result = cv(data)

        # must be perfect
        self.failUnless( result < 0.05 )

        # do crossval with permuted regressors
        cv = ClfCrossValidation(transerror,
                  NFoldSplitter(cvtype=1, permute=True, nrunspersplit=10) )
        results = cv(data)

        # must be at chance level
        pmean = N.array(results).mean()
        self.failUnless( pmean < 0.58 and pmean > 0.42 )


    def testConfusionMatrix(self):
        data = N.array([1,2,1,2,2,2,3,2,1], ndmin=2).T
        reg = N.array([1,1,1,2,2,2,3,3,3])

        cm = ConfusionMatrix()
        self.failUnlessRaises(ZeroDivisionError, lambda x:x.percentCorrect, cm)
        """No samples -- raise exception"""

        cm.add(reg, N.array([1,2,1,2,2,2,3,2,1]))

        self.failUnlessEqual(len(cm.sets), 1,
            msg="Should have a single set so far")
        self.failUnlessEqual(cm.matrix.shape, (3,3),
            msg="should be square matrix (len(reglabels) x len(reglabels)")

        self.failUnlessRaises(ValueError, cm.add, reg, N.array([1]))
        """ConfusionMatrix must complaint if number of samples different"""

        # check table content
        self.failUnless((cm.matrix == [[2,1,0],[0,3,0],[1,1,1]]).all())

        # lets add with new labels (not yet known)
        cm.add(reg, N.array([1,4,1,2,2,2,4,2,1]))

        self.failUnlessEqual(cm.labels, [1,2,3,4],
                             msg="We should have gotten 4th label")

        matrices = cm.matrices          # separate CM per each given set
        self.failUnlessEqual(len(matrices), 2,
                             msg="Have gotten two splits")

        self.failUnless((matrices[0].matrix + matrices[1].matrix == cm.matrix).all(),
                        msg="Total votes should match the sum across split CMs")

        # check pretty print
        # just a silly test to make sure that printing works
        self.failUnless(len(str(cm))>100)
        # and that it knows some parameters for printing
        self.failUnless(len(cm.__str__(summary=True, percents=True,
                                       header=False,
                                       print_empty=True))>100)

        # lets check iadd -- just itself to itself
        cm += cm
        self.failUnlessEqual(len(cm.matrices), 4, msg="Must be 4 sets now")

        # lets check add -- just itself to itself
        cm2 = cm + cm
        self.failUnlessEqual(len(cm2.matrices), 8, msg="Must be 8 sets now")
        self.failUnlessEqual(cm2.percentCorrect, cm.percentCorrect,
                             msg="Percent of corrrect should remain the same ;-)")


def suite():
    return unittest.makeSuite(CrossValidationTests)


if __name__ == '__main__':
    import test_runner

