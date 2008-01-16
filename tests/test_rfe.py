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
from sets import Set

from mvpa.datasets.maskeddataset import MaskedDataset
from mvpa.algorithms.datameasure import SensitivityAnalyzer
from mvpa.algorithms.rfe import RFE
from mvpa.algorithms.featsel import \
     SensitivityBasedFeatureSelection, \
     FeatureSelectionPipeline, \
     NBackHistoryStopCrit, FractionTailSelector, FixedErrorThresholdStopCrit, \
     MultiStopCrit, \
     FixedNElementTailSelector, BestDetector
from mvpa.algorithms.linsvmweights import LinearSVMWeights
from mvpa.clfs.svm import LinearNuSVMC
from mvpa.clfs.transerror import TransferError
from mvpa.misc.transformers import Absolute

from mvpa.misc.state import UnknownStateError

class SillySensitivityAnalyzer(SensitivityAnalyzer):
    """Simple one which just returns xrange[-N/2, N/2], where N is the
    number of features
    """

    def __init__(self, mult=1, **kwargs):
        SensitivityAnalyzer.__init__(self, **kwargs)
        self.__mult = mult

    def __call__(self, dataset, callables=[]):
        """Train linear SVM on `dataset` and extract weights from classifier.
        """
        return( self.__mult *( N.arange(dataset.nfeatures) - int(dataset.nfeatures/2) ))


class RFETests(unittest.TestCase):

    def getData(self):
        data = N.random.standard_normal(( 100, 3, 2, 4 ))
        labels = N.concatenate( ( N.repeat( 0, 50 ),
                                  N.repeat( 1, 50 ) ) )
        chunks = N.repeat( range(5), 10 )
        chunks = N.concatenate( (chunks, chunks) )
        return MaskedDataset(samples=data, labels=labels, chunks=chunks)


    def testBestDetector(self):
        bd = BestDetector()

        # for empty history -- no best
        self.failUnless(bd([]) == False)
        # we got the best if we have just 1
        self.failUnless(bd([1]) == True)
        # we got the best if we have the last minimal
        self.failUnless(bd([1, 0.9, 0.8]) == True)

        # test for alternative func
        bd = BestDetector(func=max)
        self.failUnless(bd([0.8, 0.9, 1.0]) == True)
        self.failUnless(bd([0.8, 0.9, 1.0]+[0.9]*9) == False)
        self.failUnless(bd([0.8, 0.9, 1.0]+[0.9]*10) == False)

        # test to detect earliest and latest minimum
        bd = BestDetector(lastminimum=True)
        self.failUnless(bd([3, 2, 1, 1, 1, 2, 1]) == True)
        bd = BestDetector()
        self.failUnless(bd([3, 2, 1, 1, 1, 2, 1]) == False)


    def testNBackHistoryStopCrit(self):
        """Test stopping criterion"""
        stopcrit = NBackHistoryStopCrit()
        # for empty history -- no best but just go
        self.failUnless(stopcrit([]) == False)
        # should not stop if we got 10 more after minimal
        self.failUnless(stopcrit(
            [1, 0.9, 0.8]+[0.9]*(stopcrit.steps-1)) == False)
        # should stop if we got 10 more after minimal
        self.failUnless(stopcrit(
            [1, 0.9, 0.8]+[0.9]*stopcrit.steps) == True)

        # test for alternative func
        stopcrit = NBackHistoryStopCrit(BestDetector(func=max))
        self.failUnless(stopcrit([0.8, 0.9, 1.0]+[0.9]*9) == False)
        self.failUnless(stopcrit([0.8, 0.9, 1.0]+[0.9]*10) == True)

        # test to detect earliest and latest minimum
        stopcrit = NBackHistoryStopCrit(BestDetector(lastminimum=True))
        self.failUnless(stopcrit([3, 2, 1, 1, 1, 2, 1]) == False)
        stopcrit = NBackHistoryStopCrit(steps=4)
        self.failUnless(stopcrit([3, 2, 1, 1, 1, 2, 1]) == True)


    def testFixedErrorThresholdStopCrit(self):
        """Test stopping criterion"""
        stopcrit = FixedErrorThresholdStopCrit(0.5)

        self.failUnless(stopcrit([]) == False)
        self.failUnless(stopcrit([0.8, 0.9, 0.5]) == False)
        self.failUnless(stopcrit([0.8, 0.9, 0.4]) == True)
        # only last error has to be below to stop
        self.failUnless(stopcrit([0.8, 0.4, 0.6]) == False)


    def testMultiStopCrit(self):
        """Test multiple stop criteria"""
        stopcrit = MultiStopCrit([FixedErrorThresholdStopCrit(0.5),
                                  NBackHistoryStopCrit(steps=4)])

        # default 'or' mode
        # nback triggers
        self.failUnless(stopcrit([1, 0.9, 0.8]+[0.9]*4) == True)
        # threshold triggers
        self.failUnless(stopcrit([1, 0.9, 0.2]) == True)

        # alternative 'and' mode
        stopcrit = MultiStopCrit([FixedErrorThresholdStopCrit(0.5),
                                  NBackHistoryStopCrit(steps=4)],
                                 mode = 'and')
        # nback triggers not
        self.failUnless(stopcrit([1, 0.9, 0.8]+[0.9]*4) == False)
        # threshold triggers not
        self.failUnless(stopcrit([1, 0.9, 0.2]) == False)
        # only both satisfy
        self.failUnless(stopcrit([1, 0.9, 0.4]+[0.4]*4) == True)


    def testFeatureSelector(self):
        """Test feature selector"""
        # remove 10% weekest
        selector = FractionTailSelector(0.1)
        dataset = N.array([3.5, 10, 7, 5, -0.4, 0, 0, 2, 10, 9])
        # == rank [4, 5, 6, 7, 0, 3, 2, 9, 1, 8]
        target10 = N.array([0, 1, 2, 3, 5, 6, 7, 8, 9])
        target30 = N.array([0, 1, 2, 3, 7, 8, 9])

        self.failUnlessRaises(UnknownStateError,
                              selector.__getattribute__, 'ndiscarded')
        self.failUnless((selector(dataset) == target10).all())
        selector.felements = 0.30      # discard 30%
        self.failUnless(selector.felements == 0.3)
        self.failUnless((selector(dataset) == target30).all())
        self.failUnless(selector.ndiscarded == 3) # se 3 were discarded

        selector = FixedNElementTailSelector(1)
        dataset = N.array([3.5, 10, 7, 5, -0.4, 0, 0, 2, 10, 9])
        self.failUnless((selector(dataset) == target10).all())

        selector.nelements = 3
        self.failUnless(selector.nelements == 3)
        self.failUnless((selector(dataset) == target30).all())
        self.failUnless(selector.ndiscarded == 3)


    def testSensitivityBasedFeatureSelection(self):
        svm = LinearNuSVMC()

        # sensitivity analyser and transfer error quantifier use the SAME clf!
        sens_ana = LinearSVMWeights(svm)

        # of features to remove
        Nremove = 2

        # because the clf is already trained when computing the sensitivity
        # map, prevent retraining for transfer error calculation
        # Use absolute of the svm weights as sensitivity
        fe = SensitivityBasedFeatureSelection(Absolute(sens_ana),
                feature_selector=FixedNElementTailSelector(2),
                enable_states=["sensitivity", "selected_ids"])

        wdata = self.getData()
        wdata_nfeatures = wdata.nfeatures
        tdata = self.getData()
        tdata_nfeatures = tdata.nfeatures

        sdata, stdata = fe(wdata, tdata)

        # fail if orig datasets are changed
        self.failUnless(wdata.nfeatures == wdata_nfeatures)
        self.failUnless(tdata.nfeatures == tdata_nfeatures)

        # silly check if nfeatures got a single one removed
        self.failUnlessEqual(wdata.nfeatures, sdata.nfeatures+Nremove,
            msg="We had to remove just a single feature")

        self.failUnlessEqual(tdata.nfeatures, stdata.nfeatures+Nremove,
            msg="We had to remove just a single feature in testing as well")

        self.failUnlessEqual(len(fe.sensitivity), wdata_nfeatures,
            msg="Sensitivity have to have # of features equal to original")

        self.failUnlessEqual(len(fe.selected_ids), sdata.nfeatures,
            msg="# of selected features must be equal the one in the result dataset")


    def testFeatureSelectionPipeline(self):
        sens_ana = SillySensitivityAnalyzer()

        wdata = self.getData()
        wdata_nfeatures = wdata.nfeatures
        tdata = self.getData()
        tdata_nfeatures = tdata.nfeatures

        # test silly one first ;-)
        self.failUnlessEqual(sens_ana(wdata)[0], -int(wdata_nfeatures/2))

        # first remove 25% == 6, and then 4, total removing 10
        feature_selections = [SensitivityBasedFeatureSelection(
                                sens_ana,
                                FractionTailSelector(0.25)),
                              SensitivityBasedFeatureSelection(
                                sens_ana,
                                FixedNElementTailSelector(4))
                              ]

        # create a FeatureSelection pipeline
        feat_sel_pipeline = FeatureSelectionPipeline(
            feature_selections=feature_selections,
            enable_states=['nfeatures', 'selected_ids'])

        sdata, stdata = feat_sel_pipeline(wdata, tdata)

        self.failUnlessEqual(len(feat_sel_pipeline.feature_selections),
                             len(feature_selections),
                             msg="Test the property feature_selections")

        self.failUnlessEqual(feat_sel_pipeline.nfeatures,
                             [wdata_nfeatures, wdata_nfeatures-6],
                             msg="Test if nfeatures get assigned properly")

        self.failUnlessEqual(list(feat_sel_pipeline.selected_ids),
                             list(range(10, wdata_nfeatures)))


    def testRFE(self):
        svm = LinearNuSVMC()

        # sensitivity analyser and transfer error quantifier use the SAME clf!
        sens_ana = LinearSVMWeights(svm)
        trans_error = TransferError(svm)
        # because the clf is already trained when computing the sensitivity
        # map, prevent retraining for transfer error calculation
        # Use absolute of the svm weights as sensitivity
        rfe = RFE(Absolute(sens_ana),
                  trans_error,
                  feature_selector=FixedNElementTailSelector(1),
                  train_clf=False)

        wdata = self.getData()
        wdata_nfeatures = wdata.nfeatures
        tdata = self.getData()
        tdata_nfeatures = tdata.nfeatures

        sdata, stdata = rfe(wdata, tdata)

        # fail if orig datasets are changed
        self.failUnless(wdata.nfeatures == wdata_nfeatures)
        self.failUnless(tdata.nfeatures == tdata_nfeatures)

        # check that the features set with the least error is selected
        if len(rfe.errors):
            e = N.array(rfe.errors)
            self.failUnless(sdata.nfeatures == wdata_nfeatures - e.argmin())
        else:
            self.failUnless(sdata.nfeatures == wdata_nfeatures)

        # silly check if nfeatures is in decreasing order
        nfeatures = N.array(rfe.nfeatures).copy()
        nfeatures.sort()
        self.failUnless( (nfeatures[::-1] == rfe.nfeatures).all() )

        # check if history has elements for every step
        self.failUnless(Set(rfe.history)
                        == Set(range(len(N.array(rfe.errors)))))

        # Last (the largest number) can be present multiple times even
        # if we remove 1 feature at a time -- just need to stop well
        # in advance when we have more than 1 feature left ;)
        self.failUnless(rfe.nfeatures[-1]
                        == len(N.where(rfe.history
                                       ==max(rfe.history))[0]))

        # XXX add a test where sensitivity analyser and transfer error do not
        # use the same classifier



def suite():
    return unittest.makeSuite(RFETests)


if __name__ == '__main__':
    import test_runner

