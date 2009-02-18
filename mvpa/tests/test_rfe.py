# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA recursive feature elimination"""

from sets import Set

from mvpa.datasets.splitters import NFoldSplitter
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.datasets.masked import MaskedDataset
from mvpa.measures.base import FeaturewiseDatasetMeasure
from mvpa.featsel.rfe import RFE
from mvpa.featsel.base import \
     SensitivityBasedFeatureSelection, \
     FeatureSelectionPipeline
from mvpa.featsel.helpers import \
     NBackHistoryStopCrit, FractionTailSelector, FixedErrorThresholdStopCrit, \
     MultiStopCrit, NStepsStopCrit, \
     FixedNElementTailSelector, BestDetector, RangeElementSelector

from mvpa.clfs.meta import FeatureSelectionClassifier, SplitClassifier
from mvpa.clfs.transerror import TransferError
from mvpa.misc.transformers import Absolute

from mvpa.misc.state import UnknownStateError

from tests_warehouse import *
from tests_warehouse_clfs import *

class SillySensitivityAnalyzer(FeaturewiseDatasetMeasure):
    """Simple one which just returns xrange[-N/2, N/2], where N is the
    number of features
    """

    def __init__(self, mult=1, **kwargs):
        FeaturewiseDatasetMeasure.__init__(self, **kwargs)
        self.__mult = mult

    def __call__(self, dataset):
        """Train linear SVM on `dataset` and extract weights from classifier.
        """
        return( self.__mult *( N.arange(dataset.nfeatures) - int(dataset.nfeatures/2) ))


class RFETests(unittest.TestCase):

    def getData(self):
        return datasets['uni2medium_train']

    def getDataT(self):
        return datasets['uni2medium_test']


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


    def testNStepsStopCrit(self):
        """Test stopping criterion"""
        stopcrit = NStepsStopCrit(2)

        self.failUnless(stopcrit([]) == False)
        self.failUnless(stopcrit([0.8, 0.9]) == True)
        self.failUnless(stopcrit([0.8]) == False)


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
        data = N.array([3.5, 10, 7, 5, -0.4, 0, 0, 2, 10, 9])
        # == rank [4, 5, 6, 7, 0, 3, 2, 9, 1, 8]
        target10 = N.array([0, 1, 2, 3, 5, 6, 7, 8, 9])
        target30 = N.array([0, 1, 2, 3, 7, 8, 9])

        self.failUnlessRaises(UnknownStateError,
                              selector.__getattribute__, 'ndiscarded')
        self.failUnless((selector(data) == target10).all())
        selector.felements = 0.30      # discard 30%
        self.failUnless(selector.felements == 0.3)
        self.failUnless((selector(data) == target30).all())
        self.failUnless(selector.ndiscarded == 3) # se 3 were discarded

        selector = FixedNElementTailSelector(1)
        #                   0   1   2  3   4    5  6  7  8   9
        data = N.array([3.5, 10, 7, 5, -0.4, 0, 0, 2, 10, 9])
        self.failUnless((selector(data) == target10).all())

        selector.nelements = 3
        self.failUnless(selector.nelements == 3)
        self.failUnless((selector(data) == target30).all())
        self.failUnless(selector.ndiscarded == 3)

        # test range selector
        # simple range 'above'
        self.failUnless((RangeElementSelector(lower=0)(data) == \
                         N.array([0,1,2,3,7,8,9])).all())

        self.failUnless((RangeElementSelector(lower=0,
                                              inclusive=True)(data) == \
                         N.array([0,1,2,3,5,6,7,8,9])).all())

        self.failUnless((RangeElementSelector(lower=0, mode='discard',
                                              inclusive=True)(data) == \
                         N.array([4])).all())

        # simple range 'below'
        self.failUnless((RangeElementSelector(upper=2)(data) == \
                         N.array([4,5,6])).all())

        self.failUnless((RangeElementSelector(upper=2,
                                              inclusive=True)(data) == \
                         N.array([4,5,6,7])).all())

        self.failUnless((RangeElementSelector(upper=2, mode='discard',
                                              inclusive=True)(data) == \
                         N.array([0,1,2,3,8,9])).all())


        # ranges
        self.failUnless((RangeElementSelector(lower=2, upper=9)(data) == \
                         N.array([0,2,3])).all())

        self.failUnless((RangeElementSelector(lower=2, upper=9,
                                              inclusive=True)(data) == \
                         N.array([0,2,3,7,9])).all())

        self.failUnless((RangeElementSelector(upper=2, lower=9, mode='discard',
                                              inclusive=True)(data) ==
                         RangeElementSelector(lower=2, upper=9,
                                              inclusive=False)(data)).all())

        # non-0 elements -- should be equivalent to N.nonzero()[0]
        self.failUnless((RangeElementSelector()(data) == \
                         N.nonzero(data)[0]).all())


    @sweepargs(clf=clfswh['has_sensitivity', '!meta'])
    def testSensitivityBasedFeatureSelection(self, clf):

        # sensitivity analyser and transfer error quantifier use the SAME clf!
        sens_ana = clf.getSensitivityAnalyzer()

        # of features to remove
        Nremove = 2

        # because the clf is already trained when computing the sensitivity
        # map, prevent retraining for transfer error calculation
        # Use absolute of the svm weights as sensitivity
        fe = SensitivityBasedFeatureSelection(sens_ana,
                feature_selector=FixedNElementTailSelector(2),
                enable_states=["sensitivity", "selected_ids"])

        wdata = self.getData()
        wdata_nfeatures = wdata.nfeatures
        tdata = self.getDataT()
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
        tdata = self.getDataT()
        tdata_nfeatures = tdata.nfeatures

        # test silly one first ;-)
        self.failUnlessEqual(sens_ana(wdata)[0], -int(wdata_nfeatures/2))

        # OLD: first remove 25% == 6, and then 4, total removing 10
        # NOW: test should be independent of the numerical number of features
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

        desired_nfeatures = int(N.ceil(wdata_nfeatures*0.75))
        self.failUnlessEqual(feat_sel_pipeline.nfeatures,
                             [wdata_nfeatures, desired_nfeatures],
                             msg="Test if nfeatures get assigned properly."
                             " Got %s!=%s" % (feat_sel_pipeline.nfeatures,
                                              [wdata_nfeatures, desired_nfeatures]))

        self.failUnlessEqual(list(feat_sel_pipeline.selected_ids),
                             range(int(wdata_nfeatures*0.25)+4, wdata_nfeatures))


    # TODO: should later on work for any clfs_with_sens
    @sweepargs(clf=clfswh['has_sensitivity', '!meta'][:1])
    def testRFE(self, clf):

        # sensitivity analyser and transfer error quantifier use the SAME clf!
        sens_ana = clf.getSensitivityAnalyzer()
        trans_error = TransferError(clf)
        # because the clf is already trained when computing the sensitivity
        # map, prevent retraining for transfer error calculation
        # Use absolute of the svm weights as sensitivity
        rfe = RFE(sens_ana,
                  trans_error,
                  feature_selector=FixedNElementTailSelector(1),
                  train_clf=False)

        wdata = self.getData()
        wdata_nfeatures = wdata.nfeatures
        tdata = self.getDataT()
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


    def testJamesProblem(self):
        percent = 80
        dataset = datasets['uni2small']
        rfesvm_split = LinearCSVMC()
        FeatureSelection = \
            RFE(sensitivity_analyzer=rfesvm_split.getSensitivityAnalyzer(),
                transfer_error=TransferError(rfesvm_split),
                feature_selector=FractionTailSelector(
                    percent / 100.0,
                    mode='select', tail='upper'), update_sensitivity=True)

        clf = FeatureSelectionClassifier(
            clf = LinearCSVMC(),
            # on features selected via RFE
            feature_selection = FeatureSelection)
             # update sensitivity at each step (since we're not using the
             # same CLF as sensitivity analyzer)
        clf.states.enable('feature_ids')

        cv = CrossValidatedTransferError(
            TransferError(clf),
            NFoldSplitter(cvtype=1),
            enable_states=['confusion'],
            expose_testdataset=True)
        #cv = SplitClassifier(clf)
        try:
            error = cv(dataset)
            self.failUnless(error < 0.2)
        except:
            self.fail('CrossValidation cannot handle classifier with RFE '
                      'feature selection')




def suite():
    return unittest.makeSuite(RFETests)


if __name__ == '__main__':
    import runner

