#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA recursive feature elimination"""

import unittest
import numpy as N

from mvpa.datasets.maskeddataset import MaskedDataset
from mvpa.algorithms.rfe import RFE
from mvpa.algorithms.featsel import \
     StopNBackHistoryCriterion, XPercentFeatureSelector, \
     FixedNumberFeatureSelector
from mvpa.algorithms.linsvmweights import LinearSVMWeights
from mvpa.clf.svm import LinearNuSVMC
from mvpa.clf.transerror import TransferError

from mvpa.misc.state import UnknownStateError

class RFETests(unittest.TestCase):

    def getData(self):
        data = N.random.standard_normal(( 100, 3, 6, 6 ))
        labels = N.concatenate( ( N.repeat( 0, 50 ),
                                  N.repeat( 1, 50 ) ) )
        chunks = N.repeat( range(5), 10 )
        chunks = N.concatenate( (chunks, chunks) )
        mask = N.ones( (3, 6, 6) )
        mask[0,0,0] = 0
        mask[1,3,2] = 0
        return MaskedDataset(samples=data, labels=labels,
                             chunks=chunks, mask=mask)


    def testStopCriterion(self):
        """Test stopping criterions"""
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


    def testFeatureSelector(self):
        """Test feature selector"""
        # remove 10% weekest
        selector = XPercentFeatureSelector(10)
        dataset = N.array([3.5, 10, 7, 5, -0.4, 0, 0, 2, 10, 9])
        target10 = N.array([3.5, 10, 7, 5, 0, 0, 2, 10, 9])
        target20 = N.array([3.5, 10, 7, 5, 2, 10, 9])

        self.failUnlessRaises(UnknownStateError, selector.__getitem__, 'ndiscarded')
        self.failUnless((selector(dataset) == target10).all())
        selector.perc_discard = 20      # discard 20%
                                        # but since there are 2 0s
        self.failUnless((selector(dataset) == target20).all())
        self.failUnless(selector['ndiscarded'] == 3) # se 3 were discarded

        selector = FixedNumberFeatureSelector(1)
        dataset = N.array([3.5, 10, 7, 5, -0.4, 0, 0, 2, 10, 9])
        dataset1 = selector(dataset)
        self.failUnless((dataset1 == target10).all())
        self.failUnless((selector(dataset1) == target20).all())
        self.failUnless(selector['ndiscarded'] == 2)

        # Lets remove with 0s explicitely specified
        selector.number_discard = 3
        self.failUnless((selector(dataset) == target20).all())
        self.failUnless(selector['ndiscarded'] == 3)

        # XXX more needed


    def testRFE(self):
        svm = LinearNuSVMC()

        # sensitivity analyser and transfer error quantifier use the SAME clf!
        sens_ana = LinearSVMWeights(svm)
        trans_error = TransferError(svm)
        rfe = RFE(sens_ana,
                  trans_error,
                  train_clf=False)

        wdata = self.getData()
        tdata = self.getData()

        sdata = rfe(wdata, tdata)

        print sdata.nfeatures



def suite():
    return unittest.makeSuite(RFETests)


if __name__ == '__main__':
    import test_runner

