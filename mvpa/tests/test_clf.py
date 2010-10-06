# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA basic Classifiers"""

import numpy as np

from mvpa.testing import *
from mvpa.testing import _ENFORCE_CA_ENABLED

from mvpa.testing.datasets import *
from mvpa.testing.clfs import *

from mvpa.support.copy import deepcopy
from mvpa.base import externals

from mvpa.datasets.base import dataset_wizard
from mvpa.generators.partition import NFoldPartitioner, OddEvenPartitioner
from mvpa.generators.permutation import AttributePermutator
from mvpa.generators.splitters import Splitter

from mvpa.misc.exceptions import UnknownStateError
from mvpa.misc.errorfx import mean_mismatch_error

from mvpa.base.learner import DegenerateInputError, FailedToTrainError, \
        FailedToPredictError
from mvpa.clfs.meta import CombinedClassifier, \
     BinaryClassifier, MulticlassClassifier, \
     SplitClassifier, MappedClassifier, FeatureSelectionClassifier, \
     TreeClassifier, RegressionAsClassifier
from mvpa.measures.base import TransferMeasure, ProxyMeasure, CrossValidation
from mvpa.mappers.flatten import mask_mapper
from mvpa.misc.attrmap import AttributeMap
from mvpa.mappers.fx import mean_sample, BinaryFxNode


# What exceptions to allow while testing degenerate cases.
# If it pukes -- it is ok -- user will notice that something
# is wrong
_degenerate_allowed_exceptions = [
    DegenerateInputError, FailedToTrainError, FailedToPredictError]


class ClassifiersTests(unittest.TestCase):

    def setUp(self):
        self.clf_sign = SameSignClassifier()
        self.clf_less1 = Less1Classifier()

        # simple binary dataset
        self.data_bin_1 = dataset_wizard(
            samples=[[0,0],[-10,-1],[1,0.1],[1,-1],[-1,1]],
            targets=[1, 1, 1, -1, -1], # labels
            chunks=[0, 1, 2,  2, 3])  # chunks

    def test_dummy(self):
        clf = SameSignClassifier(enable_ca=['training_confusion'])
        clf.train(self.data_bin_1)
        self.failUnlessRaises(UnknownStateError, clf.ca.__getattribute__,
                              "predictions")
        """Should have no predictions after training. Predictions
        state should be explicitely disabled"""

        if not _ENFORCE_CA_ENABLED:
            self.failUnlessRaises(UnknownStateError,
                clf.ca.__getattribute__, "trained_dataset")

        self.failUnlessEqual(clf.ca.training_confusion.percent_correct,
                             100,
                             msg="Dummy clf should train perfectly")
        self.failUnlessEqual(clf.predict(self.data_bin_1.samples),
                             list(self.data_bin_1.targets))

        self.failUnlessEqual(len(clf.ca.predictions),
            self.data_bin_1.nsamples,
            msg="Trained classifier stores predictions by default")

        clf = SameSignClassifier(enable_ca=['trained_dataset'])
        clf.train(self.data_bin_1)
        assert_array_equal(clf.ca.trained_dataset.samples,
                           self.data_bin_1.samples)
        assert_array_equal(clf.ca.trained_dataset.targets,
                           self.data_bin_1.targets)


    def test_boosted(self):
        # XXXXXXX
        # silly test if we get the same result with boosted as with a single one
        bclf = CombinedClassifier(clfs=[self.clf_sign.clone(),
                                        self.clf_sign.clone()])

        self.failUnlessEqual(list(bclf.predict(self.data_bin_1.samples)),
                             list(self.data_bin_1.targets),
                             msg="Boosted classifier should work")
        self.failUnlessEqual(bclf.predict(self.data_bin_1.samples),
                             self.clf_sign.predict(self.data_bin_1.samples),
                             msg="Boosted classifier should have the same as regular")


    def test_boosted_state_propagation(self):
        bclf = CombinedClassifier(clfs=[self.clf_sign.clone(),
                                        self.clf_sign.clone()],
                                  enable_ca=['training_confusion'])

        # check ca enabling propagation
        self.failUnlessEqual(self.clf_sign.ca.is_enabled('training_confusion'),
                             _ENFORCE_CA_ENABLED)
        self.failUnlessEqual(bclf.clfs[0].ca.is_enabled('training_confusion'), True)

        bclf2 = CombinedClassifier(clfs=[self.clf_sign.clone(),
                                         self.clf_sign.clone()],
                                  propagate_ca=False,
                                  enable_ca=['training_confusion'])

        self.failUnlessEqual(self.clf_sign.ca.is_enabled('training_confusion'),
                             _ENFORCE_CA_ENABLED)
        self.failUnlessEqual(bclf2.clfs[0].ca.is_enabled('training_confusion'),
                             _ENFORCE_CA_ENABLED)



    def test_binary_decorator(self):
        ds = dataset_wizard(samples=[ [0,0], [0,1], [1,100], [-1,0], [-1,-3], [ 0,-10] ],
                     targets=[ 'sp', 'sp', 'sp', 'dn', 'sn', 'dp'])
        testdata = [ [0,0], [10,10], [-10, -1], [0.1, -0.1], [-0.2, 0.2] ]
        # labels [s]ame/[d]ifferent (sign), and [p]ositive/[n]egative first element

        clf = SameSignClassifier()
        # lets create classifier to descriminate only between same/different,
        # which is a primary task of SameSignClassifier
        bclf1 = BinaryClassifier(clf=clf,
                                 poslabels=['sp', 'sn'],
                                 neglabels=['dp', 'dn'])

        orig_labels = ds.targets[:]
        bclf1.train(ds)

        self.failUnless(bclf1.predict(testdata) ==
                        [['sp', 'sn'], ['sp', 'sn'], ['sp', 'sn'],
                         ['dn', 'dp'], ['dn', 'dp']])

        self.failUnless((ds.targets == orig_labels).all(),
                        msg="BinaryClassifier should not alter labels")


    # TODO: XXX finally just make regression/clf separation cleaner
    @sweepargs(clf=clfswh[:])
    def test_classifier_generalization(self, clf):
        """Simple test if classifiers can generalize ok on simple data
        """
        te = CrossValidation(clf, NFoldPartitioner(), postproc=mean_sample())
        # check the default
        #self.failUnless(te.transerror.errorfx is mean_mismatch_error)

        nclasses = 2 * (1 + int('multiclass' in clf.__tags__))

        ds = datasets['uni%dmedium' % nclasses]
        try:
            cve = te(ds).samples.squeeze()
        except Exception, e:
            self.fail("Failed with %s" % e)

        if nclasses > 2 and \
               ('on 5%(' in clf.descr or 'regression_based' in clf.__tags__):
            # skip those since they are barely applicable/testable here
            return

        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless(cve < 0.25, # TODO: use multinom distribution
                            msg="Got transfer error %g on %s with %d labels"
                            % (cve, ds, len(ds.UT)))


    @sweepargs(lrn=clfswh['!meta']+regrswh['!meta'])
    def test_custom_targets(self, lrn):
        """Simple test if a learner could cope with custom sa not targets
        """

        # Since we are comparing performances of two learners, we need
        # to assure that if they depend on some random seed -- they
        # would use the same value.  Currently we have such stochastic
        # behavior in SMLR
        if 'seed' in lrn.params:
            from mvpa import _random_seed
            lrn = lrn.clone()              # clone the beast
            lrn.params.seed = _random_seed # reuse the same seed
        lrn_ = lrn.clone()

        te = CrossValidation(lrn, NFoldPartitioner())

        te_ = CrossValidation(lrn_, NFoldPartitioner(), space='custom')
        nclasses = 2 * (1 + int('multiclass' in lrn.__tags__))
        dsname = ('uni%dsmall' % nclasses,
                  'sin_modulated')[int(lrn.__is_regression__)]
        ds = datasets[dsname]
        ds_ = ds.copy()
        ds_.sa['custom'] = ds_.sa['targets']
        ds_.sa.pop('targets')
        self.failUnless('targets' in ds.sa,
                        msg="'targets' should remain in original ds")

        try:
            cve = te(ds)
            cve_ = te_(ds_)
        except Exception, e:
            self.fail("Failed with %r" % e)

        assert_array_almost_equal(cve, cve_)
        "We should have got very similar errors while operating on "
        "'targets' and on 'custom'. Got %r and %r." % (cve, cve_)

        # TODO: sg/libsvm segfaults
        #       GPR  -- non-linear sensitivities
        if ('has_sensitivity' in lrn.__tags__
            and not 'libsvm' in lrn.__tags__
            and not ('gpr' in lrn.__tags__
                     and 'non-linear' in lrn.__tags__)
            ):
            ## if str(lrn) == "SVM(svm_impl='EPSILON_SVR', kernel=LinearLSKernel())":
            ##     # TODO investigate segfault
            ##     import pydb
            ##     pydb.debugger()

            s = lrn.get_sensitivity_analyzer()(ds)
            s_ = lrn_.get_sensitivity_analyzer()(ds_)
            isreg = lrn.__is_regression__
            # ^ is XOR so we shouldn't get get those sa's in
            # regressions at all
            self.failUnless(('custom' in s_.sa) ^ isreg)
            self.failUnless(('targets' in s.sa) ^ isreg)
            self.failUnless(not 'targets' in s_.sa)
            self.failUnless(not 'custom' in s.sa)
            if not 'smlr' in lrn.__tags__ or \
               cfg.getboolean('tests', 'labile', default='yes'):
                assert_array_almost_equal(s.samples, s_.samples)


    @sweepargs(clf=clfswh[:] + regrswh[:])
    def test_summary(self, clf):
        """Basic testing of the clf summary
        """
        summary1 = clf.summary()
        self.failUnless('not yet trained' in summary1)
        # Need 2 different datasets for regressions/classifiers
        dsname = ('uni2small', 'sin_modulated')[int(clf.__is_regression__)]
        clf.train(datasets[dsname])
        summary = clf.summary()
        # It should get bigger ;)
        self.failUnless(len(summary) > len(summary1))
        self.failUnless(not 'not yet trained' in summary)


    @sweepargs(clf=clfswh[:] + regrswh[:])
    def test_degenerate_usage(self, clf):
        """Test how clf handles degenerate cases
        """
        # Whenever we have only 1 feature with only 0s in it
        ds1 = datasets['uni2small'][:, [0]]
        # XXX this very line breaks LARS in many other unittests --
        # very interesting effect. but screw it -- for now it will be
        # this way
        ds1.samples[:] = 0.0             # all 0s
        # For regression we need numbers
        if clf.__is_regression__:
            ds1.targets = AttributeMap().to_numeric(ds1.targets)
        #ds2 = datasets['uni2small'][[0], :]
        #ds2.samples[:] = 0.0             # all 0s

        clf.ca.change_temporarily(
            enable_ca=['estimates', 'training_confusion'])

        # Good pukes are good ;-)
        # TODO XXX add
        #  - ", ds2):" to test degenerate ds with 1 sample
        #  - ds1 but without 0s -- just 1 feature... feature selections
        #    might lead to 'surprises' due to magic in combiners etc
        for ds in (ds1, ):
            try:
                try:
                    clf.train(ds)                   # should not crash or stall
                except (ValueError), e:
                    self.fail("Failed to train on degenerate data. Error was %r" % e)
                except DegenerateInputError:
                    # so it realized that data is degenerate and puked
                    continue
                # could we still get those?
                _ = clf.summary()
                cm = clf.ca.training_confusion
                # If succeeded to train/predict (due to
                # training_confusion) without error -- results better be
                # at "chance"
                continue
                if 'ACC' in cm.stats:
                    self.failUnlessEqual(cm.stats['ACC'], 0.5)
                else:
                    self.failUnless(np.isnan(cm.stats['CCe']))
            except tuple(_degenerate_allowed_exceptions):
                pass
        clf.ca.reset_changed_temporarily()


    # TODO: validate for regressions as well!!!
    def test_split_classifier(self):
        ds = self.data_bin_1
        clf = SplitClassifier(clf=SameSignClassifier(),
                enable_ca=['confusion', 'training_confusion',
                               'feature_ids'])
        clf.train(ds)                   # train the beast
        error = clf.ca.confusion.error
        tr_error = clf.ca.training_confusion.error

        clf2 = clf.clone()
        cv = CrossValidation(clf2, NFoldPartitioner(), postproc=mean_sample(),
            enable_ca=['stats', 'training_stats'])
        cverror = cv(ds)
        cverror = cverror.samples.squeeze()
        tr_cverror = cv.ca.training_stats.error

        self.failUnlessEqual(error, cverror,
                msg="We should get the same error using split classifier as"
                    " using CrossValidation. Got %s and %s"
                    % (error, cverror))

        self.failUnlessEqual(tr_error, tr_cverror,
                msg="We should get the same training error using split classifier as"
                    " using CrossValidation. Got %s and %s"
                    % (tr_error, tr_cverror))

        self.failUnlessEqual(clf.ca.confusion.percent_correct,
                             100,
                             msg="Dummy clf should train perfectly")
        self.failUnlessEqual(len(clf.ca.confusion.sets),
                             len(ds.UC),
                             msg="Should have 1 confusion per each split")
        self.failUnlessEqual(len(clf.clfs), len(ds.UC),
                             msg="Should have number of classifiers equal # of epochs")
        self.failUnlessEqual(clf.predict(ds.samples), list(ds.targets),
                             msg="Should classify correctly")

        # feature_ids must be list of lists, and since it is not
        # feature-selecting classifier used - we expect all features
        # to be utilized
        #  NOT ANYMORE -- for BoostedClassifier we have now union of all
        #  used features across slave classifiers. That makes
        #  semantics clear. If you need to get deeper -- use upcoming
        #  harvesting facility ;-)
        # self.failUnlessEqual(len(clf.feature_ids), len(ds.uniquechunks))
        # self.failUnless(np.array([len(ids)==ds.nfeatures
        #                         for ids in clf.feature_ids]).all())

        # Just check if we get it at all ;-)
        summary = clf.summary()


    @sweepargs(clf_=clfswh['binary', '!meta'])
    def test_split_classifier_extended(self, clf_):
        clf2 = clf_.clone()
        ds = datasets['uni2medium']#self.data_bin_1
        clf = SplitClassifier(clf=clf_, #SameSignClassifier(),
                enable_ca=['confusion', 'feature_ids'])
        clf.train(ds)                   # train the beast
        error = clf.ca.confusion.error

        cv = CrossValidation(clf2, NFoldPartitioner(), postproc=mean_sample(),
            enable_ca=['stats', 'training_stats'])
        cverror = cv(ds).samples.squeeze()

        self.failUnless(abs(error-cverror)<0.01,
                msg="We should get the same error using split classifier as"
                    " using CrossValidation. Got %s and %s"
                    % (error, cverror))

        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless(error < 0.25,
                msg="clf should generalize more or less fine. "
                    "Got error %s" % error)
        self.failUnlessEqual(len(clf.ca.confusion.sets), len(ds.UC),
            msg="Should have 1 confusion per each split")
        self.failUnlessEqual(len(clf.clfs), len(ds.UC),
            msg="Should have number of classifiers equal # of epochs")
        #self.failUnlessEqual(clf.predict(ds.samples), list(ds.targets),
        #                     msg="Should classify correctly")



    def test_harvesting(self):
        """Basic testing of harvesting based on SplitClassifier
        """
        ds = self.data_bin_1
        clf = SplitClassifier(clf=SameSignClassifier(),
                enable_ca=['confusion', 'training_confusion'],
                harvest_attribs=['clf.ca.training_time'],
                descr="DESCR")
        clf.train(ds)                   # train the beast
        # Number of harvested items should be equal to number of chunks
        self.failUnlessEqual(
            len(clf.ca.harvested['clf.ca.training_time']), len(ds.UC))
        # if we can blame multiple inheritance and ClassWithCollections.__init__
        self.failUnlessEqual(clf.descr, "DESCR")


    def test_mapped_classifier(self):
        samples = np.array([ [ 0,  0, -1], [ 1, 0, 1],
                            [-1, -1,  1], [-1, 0, 1],
                            [ 1, -1,  1] ])
        for mask, res in (([1, 1, 0], [ 1, 1,  1, -1, -1]),
                          ([1, 0, 1], [-1, 1, -1, -1,  1]),
                          ([0, 1, 1], [-1, 1, -1,  1, -1])):
            clf = MappedClassifier(clf=self.clf_sign,
                                   mapper=mask_mapper(np.array(mask,
                                                              dtype=bool)))
            self.failUnlessEqual(clf.predict(samples), res)


    def test_feature_selection_classifier(self):
        from mvpa.featsel.base import \
             SensitivityBasedFeatureSelection
        from mvpa.featsel.helpers import \
             FixedNElementTailSelector

        # should give lowest weight to the feature with lowest index
        sens_ana = SillySensitivityAnalyzer()
        # should give lowest weight to the feature with highest index
        sens_ana_rev = SillySensitivityAnalyzer(mult=-1)

        # corresponding feature selections
        feat_sel = SensitivityBasedFeatureSelection(sens_ana,
            FixedNElementTailSelector(1, mode='discard'))

        feat_sel_rev = SensitivityBasedFeatureSelection(sens_ana_rev,
            FixedNElementTailSelector(1))

        samples = np.array([ [0, 0, -1], [1, 0, 1], [-1, -1, 1],
                            [-1, 0, 1], [1, -1, 1] ])

        testdata3 = dataset_wizard(samples=samples, targets=1)
        # dummy train data so proper mapper gets created
        traindata = dataset_wizard(samples=np.array([ [0, 0, -1], [1, 0, 1] ]),
                            targets=[1, 2])

        # targets
        res110 = [1, 1, 1, -1, -1]
        res011 = [-1, 1, -1, 1, -1]

        # first classifier -- 0th feature should be discarded
        clf011 = FeatureSelectionClassifier(self.clf_sign, feat_sel,
                    enable_ca=['feature_ids'])

        self.clf_sign.ca.change_temporarily(enable_ca=['estimates'])
        clf011.train(traindata)

        self.failUnlessEqual(clf011.predict(testdata3.samples), res011)
        # just silly test if we get values assigned in the 'ProxyClassifier'
        self.failUnless(len(clf011.ca.estimates) == len(res110),
                        msg="We need to pass values into ProxyClassifier")
        self.clf_sign.ca.reset_changed_temporarily()

        self.failUnlessEqual(clf011.mapper._oshape, (2,))
        "Feature selection classifier had to be trained on 2 features"

        # first classifier -- last feature should be discarded
        clf011 = FeatureSelectionClassifier(self.clf_sign, feat_sel_rev)
        clf011.train(traindata)
        self.failUnlessEqual(clf011.predict(testdata3.samples), res110)

    def test_feature_selection_classifier_with_regression(self):
        from mvpa.featsel.base import \
             SensitivityBasedFeatureSelection
        from mvpa.featsel.helpers import \
             FixedNElementTailSelector
        if sample_clf_reg is None:
            # none regression was found, so nothing to test
            return
        # should give lowest weight to the feature with lowest index
        sens_ana = SillySensitivityAnalyzer()

        # corresponding feature selections
        feat_sel = SensitivityBasedFeatureSelection(sens_ana,
            FixedNElementTailSelector(1, mode='discard'))

        # now test with regression-based classifier. The problem is
        # that it is determining predictions twice from values and
        # then setting the values from the results, which the second
        # time is set to predictions.  The final outcome is that the
        # values are actually predictions...
        dat = dataset_wizard(samples=np.random.randn(4, 10),
                      targets=[-1, -1, 1, 1])
        clf_reg = FeatureSelectionClassifier(sample_clf_reg, feat_sel)
        clf_reg.train(dat)
        _ = clf_reg.predict(dat.samples)
        self.failIf((np.array(clf_reg.ca.estimates)
                     - clf_reg.ca.predictions).sum()==0,
                    msg="Values were set to the predictions in %s." %
                    sample_clf_reg)


    def test_tree_classifier(self):
        """Basic tests for TreeClassifier
        """
        ds = datasets['uni4small']
        clfs = clfswh['binary']         # pool of classifiers
        # Lets permute so each time we try some different combination
        # of the classifiers
        clfs = [clfs[i] for i in np.random.permutation(len(clfs))]
        # Test conflicting definition
        tclf = TreeClassifier(clfs[0], {
            'L0+2' : (('L0', 'L2'), clfs[1]),
            'L2+3' : (('L2', 'L3'), clfs[2])})
        self.failUnlessRaises(ValueError, tclf.train, ds)
        """Should raise exception since label 2 is in both"""

        # Test insufficient definition
        tclf = TreeClassifier(clfs[0], {
            'L0+5' : (('L0', 'L5'), clfs[1]),
            'L2+3' : (('L2', 'L3'),       clfs[2])})
        self.failUnlessRaises(ValueError, tclf.train, ds)
        """Should raise exception since no group for L1"""

        # proper definition now
        tclf = TreeClassifier(clfs[0], {
            'L0+1' : (('L0', 'L1'), clfs[1]),
            'L2+3' : (('L2', 'L3'), clfs[2])})

        # Lets test train/test cycle using CVTE
        cv = CrossValidation(tclf, OddEvenPartitioner(), postproc=mean_sample(),
            enable_ca=['stats', 'training_stats'])
        cverror = cv(ds).samples.squeeze()
        try:
            rtclf = repr(tclf)
        except:
            self.fail(msg="Could not obtain repr for TreeClassifier")

        # Test accessibility of .clfs
        self.failUnless(tclf.clfs['L0+1'] is clfs[1])
        self.failUnless(tclf.clfs['L2+3'] is clfs[2])

        cvtrc = cv.ca.training_stats
        cvtc = cv.ca.stats
        if cfg.getboolean('tests', 'labile', default='yes'):
            # just a dummy check to make sure everything is working
            self.failUnless(cvtrc != cvtc)
            self.failUnless(cverror < 0.3)

        # TODO: whenever implemented
        tclf = TreeClassifier(clfs[0], {
            'L0' : (('L0',), clfs[1]),
            'L1+2+3' : (('L1', 'L2', 'L3'),    clfs[2])})
        # TEST ME


    @sweepargs(clf=clfswh[:])
    def test_values(self, clf):
        if isinstance(clf, MulticlassClassifier):
            # TODO: handle those values correctly
            return
        ds = datasets['uni2small']
        clf.ca.change_temporarily(enable_ca = ['estimates'])
        cv = CrossValidation(clf, OddEvenPartitioner(),
            enable_ca=['stats', 'training_stats'])
        _ = cv(ds)
        #print clf.descr, clf.values[0]
        # basic test either we get 1 set of values per each sample
        self.failUnlessEqual(len(clf.ca.estimates), ds.nsamples/2)

        clf.ca.reset_changed_temporarily()

    @sweepargs(clf=clfswh['linear', 'svm', 'libsvm', '!meta'])
    def test_multiclass_classifier(self, clf):
        oldC = None
        # XXX somewhat ugly way to force non-dataspecific C value.
        # Otherwise multiclass libsvm builtin and our MultiClass would differ
        # in results
        if clf.params.has_key('C') and clf.params.C<0:
            oldC = clf.params.C
            clf.params.C = 1.0                 # reset C to be 1

        svm, svm2 = clf, clf.clone()
        svm2.ca.enable(['training_confusion'])

        mclf = MulticlassClassifier(clf=svm,
                                   enable_ca=['training_confusion'])

        svm2.train(datasets['uni2small'])
        mclf.train(datasets['uni2small'])
        s1 = str(mclf.ca.training_confusion)
        s2 = str(svm2.ca.training_confusion)
        self.failUnlessEqual(s1, s2,
            msg="Multiclass clf should provide same results as built-in "
                "libsvm's %s. Got %s and %s" % (svm2, s1, s2))

        svm2.untrain()

        self.failUnless(svm2.trained == False,
            msg="Un-Trained SVM should be untrained")

        self.failUnless(np.array([x.trained for x in mclf.clfs]).all(),
            msg="Trained Boosted classifier should have all primary classifiers trained")
        self.failUnless(mclf.trained,
            msg="Trained Boosted classifier should be marked as trained")

        mclf.untrain()

        self.failUnless(not mclf.trained,
                        msg="UnTrained Boosted classifier should not be trained")
        self.failUnless(not np.array([x.trained for x in mclf.clfs]).any(),
            msg="UnTrained Boosted classifier should have no primary classifiers trained")

        if oldC is not None:
            clf.params.C = oldC

        # TODO: test combiners, e.g. MaximalVote and ca they store


    # XXX meta should also work but TODO
    @sweepargs(clf=clfswh['svm', '!meta'])
    def test_svms(self, clf):
        knows_probabilities = \
            'probabilities' in clf.ca.keys() and clf.params.probability
        enable_ca = ['estimates']
        if knows_probabilities:
            enable_ca += ['probabilities']

        clf.ca.change_temporarily(enable_ca = enable_ca)
        spl = Splitter('train', count=2)
        traindata, testdata = list(spl.generate(datasets['uni2small']))
        clf.train(traindata)
        predicts = clf.predict(testdata.samples)
        # values should be different from predictions for SVMs we have
        self.failUnless(np.any(predicts != clf.ca.estimates))

        if knows_probabilities and clf.ca.is_set('probabilities'):
            # XXX test more thoroughly what we are getting here ;-)
            self.failUnlessEqual( len(clf.ca.probabilities),
                                  len(testdata.samples)  )
        clf.ca.reset_changed_temporarily()


    @sweepargs(clf=clfswh['retrainable'])
    def test_retrainables(self, clf):
        # we need a copy since will tune its internals later on
        clf = clf.clone()
        clf.ca.change_temporarily(enable_ca = ['estimates'],
                                      # ensure that it does do predictions
                                      # while training
                                      disable_ca=['training_confusion'])
        clf_re = clf.clone()
        # TODO: .retrainable must have a callback to call smth like
        # _set_retrainable
        clf_re._set_retrainable(True)

        # need to have high snr so we don't 'cope' with problematic
        # datasets since otherwise unittests would fail.
        dsargs = {'perlabel':50, 'nlabels':2, 'nfeatures':5, 'nchunks':1,
                  'nonbogus_features':[2,4], 'snr': 5.0}

        ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # NB datasets will be changed by the end of testing, so if
        # are to change to use generic datasets - make sure to copy
        # them here
        ds = deepcopy(datasets['uni2large'])
        clf.untrain()
        clf_re.untrain()
        trerr = TransferMeasure(clf, Splitter('train'),
                                postproc=BinaryFxNode(mean_mismatch_error,
                                                      'targets'))
        trerr_re =  TransferMeasure(clf_re, Splitter('train'),
                                    disable_ca=['training_stats'],
                                    postproc=BinaryFxNode(mean_mismatch_error,
                                                          'targets'))

        # Just check for correctness of retraining
        err_1 = np.asscalar(trerr(ds))
        self.failUnless(err_1<0.3,
            msg="We should test here on easy dataset. Got error of %s" % err_1)
        values_1 = clf.ca.estimates[:]
        # some times retraining gets into deeper optimization ;-)
        eps = 0.05
        corrcoef_eps = 0.85        # just to get no failures... usually > 0.95


        def batch_test(retrain=True, retest=True, closer=True):
            err = np.asscalar(trerr(ds))
            err_re = np.asscalar(trerr_re(ds))
            corr = np.corrcoef(
                clf.ca.estimates, clf_re.ca.estimates)[0, 1]
            corr_old = np.corrcoef(values_1, clf_re.ca.estimates)[0, 1]
            if __debug__:
                debug('TEST', "Retraining stats: errors %g %g corr %g "
                      "with old error %g corr %g" %
                  (err, err_re, corr, err_1, corr_old))
            self.failUnless(clf_re.ca.retrained == retrain,
                            ("Must fully train",
                             "Must retrain instead of full training")[retrain])
            self.failUnless(clf_re.ca.repredicted == retest,
                            ("Must fully test",
                             "Must retest instead of full testing")[retest])
            self.failUnless(corr > corrcoef_eps,
              msg="Result must be close to the one without retraining."
                  " Got corrcoef=%s" % (corr))
            if closer:
                self.failUnless(
                    corr >= corr_old,
                    msg="Result must be closer to current without retraining"
                    " than to old one. Got corrcoef=%s" % (corr_old))

        # Check sequential retraining/retesting
        for i in xrange(3):
            flag = bool(i!=0)
            # ok - on 1st call we should train/test, then retrain/retest
            # and we can't compare for closinest to old result since
            # we are working on the same data/classifier
            batch_test(retrain=flag, retest=flag, closer=False)

        # should retrain nicely if we change a parameter
        if 'C' in clf.params:
            clf.params.C *= 0.1
            clf_re.params.C *= 0.1
            batch_test()
        elif 'sigma_noise' in clf.params:
            clf.params.sigma_noise *= 100
            clf_re.params.sigma_noise *= 100
            batch_test()
        else:
            raise RuntimeError, \
                  'Please implement testing while changing some of the ' \
                  'params for clf %s' % clf

        # should retrain nicely if we change kernel parameter
        if hasattr(clf, 'kernel_params') and len(clf.kernel_params):
            clf.kernel_params.gamma = 0.1
            clf_re.kernel_params.gamma = 0.1
            # retest is false since kernel got recomputed thus
            # can't expect to use the same kernel
            batch_test(retest=not('gamma' in clf.kernel_params))

        # should retrain nicely if we change labels
        permute = AttributePermutator('targets', assure=True)
        oldlabels = dstrain.targets[:]
        dstrain = permute(dstrain)
        self.failUnless((oldlabels != dstrain.targets).any(),
            msg="We should succeed at permutting -- now got the same targets")
        ds = vstack((dstrain, dstest))
        batch_test()

        # Change labels in testing
        oldlabels = dstest.targets[:]
        dstest = permute(dstest)
        self.failUnless((oldlabels != dstest.targets).any(),
            msg="We should succeed at permutting -- now got the same targets")
        ds = vstack((dstrain, dstest))
        batch_test()

        # should re-train if we change data
        # reuse trained SVM and its 'final' optimization point
        if not clf.__class__.__name__ in ['GPR']: # on GPR everything depends on the data ;-)
            oldsamples = dstrain.samples.copy()
            dstrain.samples[:] += dstrain.samples*0.05
            self.failUnless((oldsamples != dstrain.samples).any())
            ds = vstack((dstrain, dstest))
            batch_test(retest=False)
        clf.ca.reset_changed_temporarily()

        # test retrain()
        # TODO XXX  -- check validity
        clf_re.retrain(dstrain);
        self.failUnless(clf_re.ca.retrained)
        clf_re.retrain(dstrain, labels=True);
        self.failUnless(clf_re.ca.retrained)
        clf_re.retrain(dstrain, traindataset=True);
        self.failUnless(clf_re.ca.retrained)

        # test repredict()
        clf_re.repredict(dstest.samples);
        self.failUnless(clf_re.ca.repredicted)
        self.failUnlessRaises(RuntimeError, clf_re.repredict,
                              dstest.samples, labels=True)
        """for now retesting with anything changed makes no sense"""
        clf_re._set_retrainable(False)


    def test_generic_tests(self):
        """Test all classifiers for conformant behavior
        """
        for clf_, traindata in \
                [(clfswh['binary'], datasets['dumb2']),
                 (clfswh['multiclass'], datasets['dumb'])]:
            traindata_copy = deepcopy(traindata) # full copy of dataset
            for clf in clf_:
                clf.train(traindata)
                self.failUnless(
                   (traindata.samples == traindata_copy.samples).all(),
                   "Training of a classifier shouldn't change original dataset")

            # TODO: enforce uniform return from predict??
            #predicted = clf.predict(traindata.samples)
            #self.failUnless(isinstance(predicted, np.ndarray))

        # Just simple test that all of them are syntaxed correctly
        self.failUnless(str(clf) != "")
        self.failUnless(repr(clf) != "")

        # TODO: unify str and repr for all classifiers

    # XXX TODO: should work on smlr, knn, ridgereg, lars as well! but now
    #     they fail to train
    #    svmocas -- segfaults -- reported to mailing list
    #    GNB, LDA, QDA -- cannot train since 1 sample isn't sufficient
    #    to assess variance
    @sweepargs(clf=clfswh['!smlr', '!knn', '!gnb', '!lda', '!qda', '!lars', '!meta', '!ridge'])
    def test_correct_dimensions_order(self, clf):
        """To check if known/present Classifiers are working properly
        with samples being first dimension. Started to worry about
        possible problems while looking at sg where samples are 2nd
        dimension
        """
        # specially crafted dataset -- if dimensions are flipped over
        # the same storage, problem becomes unseparable. Like in this case
        # incorrect order of dimensions lead to equal samples [0, 1, 0]
        traindatas = [
            dataset_wizard(samples=np.array([ [0, 0, 1.0],
                                        [1, 0, 0] ]), targets=[-1, 1]),
            dataset_wizard(samples=np.array([ [0, 0.0],
                                      [1, 1] ]), targets=[-1, 1])]

        clf.ca.change_temporarily(enable_ca = ['training_confusion'])
        for traindata in traindatas:
            clf.train(traindata)
            self.failUnlessEqual(clf.ca.training_confusion.percent_correct, 100.0,
                "Classifier %s must have 100%% correct learning on %s. Has %f" %
                (`clf`, traindata.samples, clf.ca.training_confusion.percent_correct))

            # and we must be able to predict every original sample thus
            for i in xrange(traindata.nsamples):
                sample = traindata.samples[i,:]
                predicted = clf.predict([sample])
                self.failUnlessEqual([predicted], traindata.targets[i],
                    "We must be able to predict sample %s using " % sample +
                    "classifier %s" % `clf`)
        clf.ca.reset_changed_temporarily()


    @sweepargs(regr=regrswh[:])
    def test_regression_as_classifier(self, regr):
        """Basic tests of metaclass for using regressions as classifiers
        """
        for dsname in 'uni2small', 'uni4small':
            ds = datasets[dsname]

            clf = RegressionAsClassifier(regr, enable_ca=['distances'])
            cv = CrossValidation(clf, OddEvenPartitioner(),
                    postproc=mean_sample(),
                    enable_ca=['stats', 'training_stats'])

            error = cv(ds).samples.squeeze()

            nlabels = len(ds.uniquetargets)
            if nlabels == 2 \
               and cfg.getboolean('tests', 'labile', default='yes'):
                self.failUnless(error < 0.3)

            # Check if does not puke on repr and str
            self.failUnless(str(clf) != "")
            self.failUnless(repr(clf) != "")

            self.failUnlessEqual(clf.ca.distances.shape,
                                 (ds.nsamples / 2, nlabels))

            #print "Using %s " % regr, error
            # Just validate that everything is ok
            #self.failUnless(str(cv.ca.confusion) != "")



def suite():
    return unittest.makeSuite(ClassifiersTests)


if __name__ == '__main__':
    import runner
