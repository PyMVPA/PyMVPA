# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA recursive feature elimination"""

import numpy as np

from mvpa2.generators.base import Repeater
from mvpa2.generators.partition import NFoldPartitioner, OddEvenPartitioner
from mvpa2.generators.permutation import AttributePermutator
from mvpa2.generators.splitters import Splitter
from mvpa2.datasets.base import Dataset
from mvpa2.mappers.base import ChainMapper
from mvpa2.mappers.fx import maxofabs_sample, mean_sample, BinaryFxNode, FxMapper
from mvpa2.misc.errorfx import mean_mismatch_error
from mvpa2.misc.transformers import l2_normed
from mvpa2.misc.data_generators import normal_feature_dataset
from mvpa2.featsel.rfe import RFE
from mvpa2.featsel.base import \
     SensitivityBasedFeatureSelection
from mvpa2.featsel.helpers import \
     NBackHistoryStopCrit, FractionTailSelector, FixedErrorThresholdStopCrit, \
     MultiStopCrit, NStepsStopCrit, \
     FixedNElementTailSelector, BestDetector, RangeElementSelector

from mvpa2.clfs.meta import FeatureSelectionClassifier, SplitClassifier
from mvpa2.clfs.transerror import ConfusionBasedError
from mvpa2.misc.attrmap import AttributeMap
from mvpa2.clfs.stats import MCNullDist
from mvpa2.measures.base import ProxyMeasure, CrossValidation
from mvpa2.measures.anova import OneWayAnova
from mvpa2.measures.fx import targets_dcorrcoef

from mvpa2.base.state import UnknownStateError

from mvpa2.testing import *
from mvpa2.testing.clfs import *
from mvpa2.testing.tools import reseed_rng
from mvpa2.testing.datasets import datasets


class RFETests(unittest.TestCase):
    def get_data(self):
        return datasets['uni2medium']


    def test_best_detector(self):
        bd = BestDetector()

        # for empty history -- no best
        self.assertTrue(bd([]) == False)
        # we got the best if we have just 1
        self.assertTrue(bd([1]) == True)
        # we got the best if we have the last minimal
        self.assertTrue(bd([1, 0.9, 0.8]) == True)

        # test for alternative func
        bd = BestDetector(func=max)
        self.assertTrue(bd([0.8, 0.9, 1.0]) == True)
        self.assertTrue(bd([0.8, 0.9, 1.0]+[0.9]*9) == False)
        self.assertTrue(bd([0.8, 0.9, 1.0]+[0.9]*10) == False)

        # test to detect earliest and latest minimum
        bd = BestDetector(lastminimum=True)
        self.assertTrue(bd([3, 2, 1, 1, 1, 2, 1]) == True)
        bd = BestDetector()
        self.assertTrue(bd([3, 2, 1, 1, 1, 2, 1]) == False)


    def test_n_back_history_stop_crit(self):
        """Test stopping criterion"""
        stopcrit = NBackHistoryStopCrit()
        # for empty history -- no best but just go
        self.assertTrue(stopcrit([]) == False)
        # should not stop if we got 10 more after minimal
        self.assertTrue(stopcrit(
            [1, 0.9, 0.8]+[0.9]*(stopcrit.steps-1)) == False)
        # should stop if we got 10 more after minimal
        self.assertTrue(stopcrit(
            [1, 0.9, 0.8]+[0.9]*stopcrit.steps) == True)

        # test for alternative func
        stopcrit = NBackHistoryStopCrit(BestDetector(func=max))
        self.assertTrue(stopcrit([0.8, 0.9, 1.0]+[0.9]*9) == False)
        self.assertTrue(stopcrit([0.8, 0.9, 1.0]+[0.9]*10) == True)

        # test to detect earliest and latest minimum
        stopcrit = NBackHistoryStopCrit(BestDetector(lastminimum=True))
        self.assertTrue(stopcrit([3, 2, 1, 1, 1, 2, 1]) == False)
        stopcrit = NBackHistoryStopCrit(steps=4)
        self.assertTrue(stopcrit([3, 2, 1, 1, 1, 2, 1]) == True)


    def test_fixed_error_threshold_stop_crit(self):
        """Test stopping criterion"""
        stopcrit = FixedErrorThresholdStopCrit(0.5)

        self.assertTrue(stopcrit([]) == False)
        self.assertTrue(stopcrit([0.8, 0.9, 0.5]) == False)
        self.assertTrue(stopcrit([0.8, 0.9, 0.4]) == True)
        # only last error has to be below to stop
        self.assertTrue(stopcrit([0.8, 0.4, 0.6]) == False)


    def test_n_steps_stop_crit(self):
        """Test stopping criterion"""
        stopcrit = NStepsStopCrit(2)

        self.assertTrue(stopcrit([]) == False)
        self.assertTrue(stopcrit([0.8, 0.9]) == True)
        self.assertTrue(stopcrit([0.8]) == False)


    def test_multi_stop_crit(self):
        """Test multiple stop criteria"""
        stopcrit = MultiStopCrit([FixedErrorThresholdStopCrit(0.5),
                                  NBackHistoryStopCrit(steps=4)])

        # default 'or' mode
        # nback triggers
        self.assertTrue(stopcrit([1, 0.9, 0.8]+[0.9]*4) == True)
        # threshold triggers
        self.assertTrue(stopcrit([1, 0.9, 0.2]) == True)

        # alternative 'and' mode
        stopcrit = MultiStopCrit([FixedErrorThresholdStopCrit(0.5),
                                  NBackHistoryStopCrit(steps=4)],
                                 mode = 'and')
        # nback triggers not
        self.assertTrue(stopcrit([1, 0.9, 0.8]+[0.9]*4) == False)
        # threshold triggers not
        self.assertTrue(stopcrit([1, 0.9, 0.2]) == False)
        # only both satisfy
        self.assertTrue(stopcrit([1, 0.9, 0.4]+[0.4]*4) == True)


    def test_feature_selector(self):
        """Test feature selector"""
        # remove 10% weekest
        selector = FractionTailSelector(0.1)
        data = np.array([3.5, 10, 7, 5, -0.4, 0, 0, 2, 10, 9])
        # == rank [4, 5, 6, 7, 0, 3, 2, 9, 1, 8]
        target10 = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9])
        target30 = np.array([0, 1, 2, 3, 7, 8, 9])

        self.assertRaises(UnknownStateError,
                              selector.ca.__getattribute__, 'ndiscarded')
        self.assertTrue((selector(data) == target10).all())
        selector.felements = 0.30      # discard 30%
        self.assertTrue(selector.felements == 0.3)
        self.assertTrue((selector(data) == target30).all())
        self.assertTrue(selector.ca.ndiscarded == 3) # se 3 were discarded

        selector = FixedNElementTailSelector(1)
        #                   0   1   2  3   4    5  6  7  8   9
        data = np.array([3.5, 10, 7, 5, -0.4, 0, 0, 2, 10, 9])
        self.assertTrue((selector(data) == target10).all())

        selector.nelements = 3
        self.assertTrue(selector.nelements == 3)
        self.assertTrue((selector(data) == target30).all())
        self.assertTrue(selector.ca.ndiscarded == 3)

        # test range selector
        # simple range 'above'
        self.assertTrue((RangeElementSelector(lower=0)(data) == \
                         np.array([0,1,2,3,7,8,9])).all())

        self.assertTrue((RangeElementSelector(lower=0,
                                              inclusive=True)(data) == \
                         np.array([0,1,2,3,5,6,7,8,9])).all())

        self.assertTrue((RangeElementSelector(lower=0, mode='discard',
                                              inclusive=True)(data) == \
                         np.array([4])).all())

        # simple range 'below'
        self.assertTrue((RangeElementSelector(upper=2)(data) == \
                         np.array([4,5,6])).all())

        self.assertTrue((RangeElementSelector(upper=2,
                                              inclusive=True)(data) == \
                         np.array([4,5,6,7])).all())

        self.assertTrue((RangeElementSelector(upper=2, mode='discard',
                                              inclusive=True)(data) == \
                         np.array([0,1,2,3,8,9])).all())


        # ranges
        self.assertTrue((RangeElementSelector(lower=2, upper=9)(data) == \
                         np.array([0,2,3])).all())

        self.assertTrue((RangeElementSelector(lower=2, upper=9,
                                              inclusive=True)(data) == \
                         np.array([0,2,3,7,9])).all())

        self.assertTrue((RangeElementSelector(upper=2, lower=9, mode='discard',
                                              inclusive=True)(data) ==
                         RangeElementSelector(lower=2, upper=9,
                                              inclusive=False)(data)).all())

        # non-0 elements -- should be equivalent to np.nonzero()[0]
        self.assertTrue((RangeElementSelector()(data) == \
                         np.nonzero(data)[0]).all())


    # XXX put GPR back in after it gets fixed up
    @sweepargs(clf=clfswh['has_sensitivity', '!meta', '!gpr'])
    def test_sensitivity_based_feature_selection(self, clf):

        # sensitivity analyser and transfer error quantifier use the SAME clf!
        sens_ana = clf.get_sensitivity_analyzer(postproc=maxofabs_sample())

        # of features to remove
        Nremove = 2

        # because the clf is already trained when computing the sensitivity
        # map, prevent retraining for transfer error calculation
        # Use absolute of the svm weights as sensitivity
        fe = SensitivityBasedFeatureSelection(sens_ana,
                feature_selector=FixedNElementTailSelector(2),
                enable_ca=["sensitivity", "selected_ids"])

        data = self.get_data()

        data_nfeatures = data.nfeatures

        fe.train(data)
        resds = fe(data)

        # fail if orig datasets are changed
        self.assertTrue(data.nfeatures == data_nfeatures)

        # silly check if nfeatures got a single one removed
        self.assertEqual(data.nfeatures, resds.nfeatures+Nremove,
            msg="We had to remove just a single feature")

        self.assertEqual(fe.ca.sensitivity.nfeatures, data_nfeatures,
            msg="Sensitivity have to have # of features equal to original")



    def test_feature_selection_pipeline(self):
        sens_ana = SillySensitivityAnalyzer()

        data = self.get_data()
        data_nfeatures = data.nfeatures

        # test silly one first ;-)
        self.assertEqual(sens_ana(data).samples[0,0], -int(data_nfeatures/2))

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
        feat_sel_pipeline = ChainMapper(feature_selections)

        feat_sel_pipeline.train(data)
        resds = feat_sel_pipeline(data)

        self.assertEqual(len(feat_sel_pipeline),
                             len(feature_selections),
                             msg="Test the property feature_selections")

        desired_nfeatures = int(np.ceil(data_nfeatures*0.75))
        self.assertEqual([fe._oshape[0] for fe in feat_sel_pipeline],
                             [desired_nfeatures, desired_nfeatures - 4])


    # TODO: should later on work for any clfs_with_sens
    @sweepargs(clf=clfswh['has_sensitivity', '!meta'][:1])
    @reseed_rng()
    def test_rfe(self, clf):

        # sensitivity analyser and transfer error quantifier use the SAME clf!
        sens_ana = clf.get_sensitivity_analyzer(postproc=maxofabs_sample())
        pmeasure = ProxyMeasure(clf, postproc=BinaryFxNode(mean_mismatch_error,
                                                           'targets'))
        cvmeasure = CrossValidation(clf, NFoldPartitioner(),
                                    errorfx=mean_mismatch_error,
                                    postproc=mean_sample())

        rfesvm_split = SplitClassifier(clf, OddEvenPartitioner())

        # explore few recipes
        for rfe, data in [
            # because the clf is already trained when computing the sensitivity
            # map, prevent retraining for transfer error calculation
            # Use absolute of the svm weights as sensitivity
            (RFE(sens_ana,
                pmeasure,
                Splitter('train'),
                fselector=FixedNElementTailSelector(1),
                train_pmeasure=False),
             self.get_data()),
            # use cross-validation within training to get error for the stopping point
            # but use full training data to derive sensitivity
            (RFE(sens_ana,
                 cvmeasure,
                 Repeater(2),            # give the same full dataset to sens_ana and cvmeasure
                 fselector=FractionTailSelector(
                     0.70,
                     mode='select', tail='upper'),
                train_pmeasure=True),
             normal_feature_dataset(perlabel=20, nchunks=5, nfeatures=200,
                                    nonbogus_features=[0, 1], snr=1.5)),
            # use cross-validation (via SplitClassifier) and get mean
            # of normed sensitivities across those splits
            (RFE(rfesvm_split.get_sensitivity_analyzer(
                    postproc=ChainMapper([ FxMapper('features', l2_normed),
                                           FxMapper('samples', np.abs),
                                           FxMapper('samples', np.mean)])),
                 ConfusionBasedError(rfesvm_split, confusion_state='stats'),
                 Repeater(2),             #  we will use the same full cv-training dataset
                 fselector=FractionTailSelector(
                     0.50,
                     mode='select', tail='upper'),
                 stopping_criterion=NBackHistoryStopCrit(BestDetector(), 10),
                 train_pmeasure=False,    # we just extract it from existing confusion
                 update_sensitivity=True),
             normal_feature_dataset(perlabel=28, nchunks=7, nfeatures=200,
                                    nonbogus_features=[0, 1], snr=1.5))
            ]:
            # prep data
            # data = datasets['uni2medium']
            data_nfeatures = data.nfeatures

            rfe.train(data)
            resds = rfe(data)

            # fail if orig datasets are changed
            self.assertTrue(data.nfeatures == data_nfeatures)

            # check that the features set with the least error is selected
            if len(rfe.ca.errors):
                e = np.array(rfe.ca.errors)
                if isinstance(rfe._fselector, FixedNElementTailSelector):
                    self.assertTrue(resds.nfeatures == data_nfeatures - e.argmin())
                else:
                    imin = np.argmin(e)
                    if 'does_feature_selection' in clf.__tags__:
                        # if clf is smart it might figure it out right away
                        assert_array_less( imin, len(e) )
                    else:
                        # in this case we can even check if we had actual
                        # going down/up trend... although -- why up???
                        self.assertTrue( 1 < imin < len(e) - 1 )
            else:
                self.assertTrue(resds.nfeatures == data_nfeatures)

            # silly check if nfeatures is in decreasing order
            nfeatures = np.array(rfe.ca.nfeatures).copy()
            nfeatures.sort()
            self.assertTrue( (nfeatures[::-1] == rfe.ca.nfeatures).all() )

            # check if history has elements for every step
            self.assertTrue(set(rfe.ca.history)
                            == set(range(len(np.array(rfe.ca.errors)))))

            # Last (the largest number) can be present multiple times even
            # if we remove 1 feature at a time -- just need to stop well
            # in advance when we have more than 1 feature left ;)
            self.assertTrue(rfe.ca.nfeatures[-1]
                            == len(np.where(rfe.ca.history
                                           ==max(rfe.ca.history))[0]))

            # XXX add a test where sensitivity analyser and transfer error do not
            # use the same classifier


    def test_james_problem(self):
        percent = 80
        dataset = datasets['uni2small']
        rfesvm_split = LinearCSVMC()
        fs = \
            RFE(rfesvm_split.get_sensitivity_analyzer(),
                ProxyMeasure(rfesvm_split,
                             postproc=BinaryFxNode(mean_mismatch_error,
                                                   'targets')),
                Splitter('train'),
                fselector=FractionTailSelector(
                    percent / 100.0,
                    mode='select', tail='upper'), update_sensitivity=True)

        clf = FeatureSelectionClassifier(
            LinearCSVMC(),
            # on features selected via RFE
            fs)
             # update sensitivity at each step (since we're not using the
             # same CLF as sensitivity analyzer)

        class StoreResults(object):
            def __init__(self):
                self.storage = []
            def __call__(self, data, node, result):
                self.storage.append((node.measure.mapper.ca.history,
                                     node.measure.mapper.ca.errors)),

        cv_storage = StoreResults()
        cv = CrossValidation(clf, NFoldPartitioner(), postproc=mean_sample(),
                             callback=cv_storage,
                             enable_ca=['confusion']) # TODO -- it is stats
        #cv = SplitClassifier(clf)
        try:
            error = cv(dataset).samples.squeeze()
        except Exception, e:
            self.fail('CrossValidation cannot handle classifier with RFE '
                      'feature selection. Got exception: %s' % (e,))

        assert(len(cv_storage.storage) == len(dataset.sa['chunks'].unique))
        assert(len(cv_storage.storage[0]) == 2)
        assert(len(cv_storage.storage[0][0]) == dataset.nfeatures)

        self.assertTrue(error < 0.2)


    def test_james_problem_multiclass(self):
        percent = 80
        dataset = datasets['uni4large']
        #dataset = dataset[:, dataset.a.nonbogus_features]

        rfesvm_split = LinearCSVMC()
        fs = \
            RFE(rfesvm_split.get_sensitivity_analyzer(
            postproc=ChainMapper([
                #FxMapper('features', l2_normed),
                #FxMapper('samples', np.mean),
                #FxMapper('samples', np.abs)
                FxMapper('features', lambda x: np.argsort(np.abs(x))),
                #maxofabs_sample()
                mean_sample()
                ])),
                ProxyMeasure(rfesvm_split,
                             postproc=BinaryFxNode(mean_mismatch_error,
                                                   'targets')),
                Splitter('train'),
                fselector=FractionTailSelector(
                    percent / 100.0,
                    mode='select', tail='upper'), update_sensitivity=True)

        clf = FeatureSelectionClassifier(
            LinearCSVMC(),
            # on features selected via RFE
            fs)
             # update sensitivity at each step (since we're not using the
             # same CLF as sensitivity analyzer)

        class StoreResults(object):
            def __init__(self):
                self.storage = []
            def __call__(self, data, node, result):
                self.storage.append((node.measure.mapper.ca.history,
                                     node.measure.mapper.ca.errors)),

        cv_storage = StoreResults()
        cv = CrossValidation(clf, NFoldPartitioner(), postproc=mean_sample(),
                             callback=cv_storage,
                             enable_ca=['stats'])
        #cv = SplitClassifier(clf)
        try:
            error = cv(dataset).samples.squeeze()
        except Exception, e:
            self.fail('CrossValidation cannot handle classifier with RFE '
                      'feature selection. Got exception: %s' % (e,))
        #print "ERROR: ", error
        #print cv.ca.stats
        assert(len(cv_storage.storage) == len(dataset.sa['chunks'].unique))
        assert(len(cv_storage.storage[0]) == 2)
        assert(len(cv_storage.storage[0][0]) == dataset.nfeatures)
        #print "non bogus features",  dataset.a.nonbogus_features
        #print cv_storage.storage

        self.assertTrue(error < 0.2)


    ##REF: Name was automagically refactored
    def __test_matthias_question(self):
        rfe_clf = LinearCSVMC(C=1)

        rfesvm_split = SplitClassifier(rfe_clf)
        clf = \
            FeatureSelectionClassifier(
            clf = LinearCSVMC(C=1),
            feature_selection = RFE(
                sensitivity_analyzer = rfesvm_split.get_sensitivity_analyzer(
                    combiner=first_axis_mean,
                    transformer=np.abs),
                transfer_error=ConfusionBasedError(
                    rfesvm_split,
                    confusion_state="confusion"),
                stopping_criterion=FixedErrorThresholdStopCrit(0.20),
                feature_selector=FractionTailSelector(
                    0.2, mode='discard', tail='lower'),
                update_sensitivity=True))

        no_permutations = 1000
        permutator = AttributePermutator('targets', count=no_permutations)
        cv = CrossValidation(clf, NFoldPartitioner(),
            null_dist=MCNullDist(permutator, tail='left'),
            enable_ca=['stats'])
        error = cv(datasets['uni2small'])
        self.assertTrue(error < 0.4)
        self.assertTrue(cv.ca.null_prob < 0.05)

    @reseed_rng()
    @labile(3, 1)
    # Let's test with clf sens analyzer AND OneWayAnova
    @sweepargs(fmeasure=(None,  # use clf's sensitivity analyzer
                         OneWayAnova(), # ad-hoc feature-wise measure
                         # targets_mutualinfo_kde(), # FxMeasure
                         targets_dcorrcoef(), # FxMeasure wrapper
               ))
    def test_SplitRFE(self, fmeasure):
        # just a smoke test ATM
        from mvpa2.clfs.svm import LinearCSVMC
        from mvpa2.clfs.meta import MappedClassifier
        from mvpa2.misc.data_generators import normal_feature_dataset
        #import mvpa2.featsel.rfe
        #reload(mvpa2.featsel.rfe)
        from mvpa2.featsel.rfe import RFE, SplitRFE
        from mvpa2.generators.partition import NFoldPartitioner
        from mvpa2.featsel.helpers import FractionTailSelector
        from mvpa2.testing import ok_, assert_equal

        clf = LinearCSVMC(C=1)
        dataset = normal_feature_dataset(perlabel=20, nlabels=2, nfeatures=11,
                                         snr=1., nonbogus_features=[1, 5])
        # flip one of the meaningful features around to see
        # if we are still getting proper selection
        dataset.samples[:, dataset.a.nonbogus_features[1]] *= -1
        # 3 partitions should be enough for testing
        partitioner = NFoldPartitioner(count=3)

        rfeclf = MappedClassifier(
            clf, SplitRFE(clf,
                          partitioner,
                          fselector=FractionTailSelector(
                              0.5, mode='discard', tail='lower'),
                          fmeasure=fmeasure,
                           # need to update only when using clf's sens anal
                          update_sensitivity=fmeasure is None))
        r0 = repr(rfeclf)

        ok_(rfeclf.mapper.nfeatures_min == 0)
        rfeclf.train(dataset)
        ok_(rfeclf.mapper.nfeatures_min > 0)
        predictions = rfeclf(dataset).samples

        # at least 1 of the nonbogus-features should be chosen
        ok_(len(set(dataset.a.nonbogus_features).intersection(
                rfeclf.mapper.slicearg)) > 0)

        # check repr to have all needed pieces
        r = repr(rfeclf)
        s = str(rfeclf)
        ok_(('partitioner=NFoldP' in r) or
            ('partitioner=mvpa2.generators.partition.NFoldPartitioner' in r))
        ok_('lrn=' in r)
        ok_(not 'slicearg=' in r)
        assert_equal(r, r0)

        if externals.exists('joblib'):
            rfeclf.mapper.nproc = -1
            # compare results against the one ran in parallel
            _slicearg = rfeclf.mapper.slicearg
            _predictions = predictions
            rfeclf.train(dataset)
            predictions = rfeclf(dataset).samples
            assert_array_equal(predictions, _predictions)
            assert_array_equal(_slicearg, rfeclf.mapper.slicearg)

        # Test that we can collect stats from cas within cross-validation
        sensitivities = []
        nested_errors = []
        nested_nfeatures = []
        def store_me(data, node, result):
            sens = node.measure.get_sensitivity_analyzer(force_train=False)(data)
            sensitivities.append(sens)
            nested_errors.append(node.measure.mapper.ca.nested_errors)
            nested_nfeatures.append(node.measure.mapper.ca.nested_nfeatures)
        cv = CrossValidation(rfeclf, NFoldPartitioner(count=1), callback=store_me,
                             enable_ca=['stats'])
        _ = cv(dataset)
        # just to make sure we collected them
        assert_equal(len(sensitivities), 1)
        assert_equal(len(nested_errors), 1)
        assert_equal(len(nested_nfeatures), 1)

def suite():  # pragma: no cover
    return unittest.makeSuite(RFETests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

