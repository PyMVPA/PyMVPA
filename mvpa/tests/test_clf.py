# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA basic Classifiers"""

from mvpa.support.copy import deepcopy
from mvpa.base import externals

from mvpa.datasets import Dataset
from mvpa.mappers.mask import MaskMapper
from mvpa.datasets.splitters import NFoldSplitter, OddEvenSplitter

from mvpa.misc.exceptions import UnknownStateError

from mvpa.clfs.base import DegenerateInputError, FailedToTrainError
from mvpa.clfs.meta import CombinedClassifier, \
     BinaryClassifier, MulticlassClassifier, \
     SplitClassifier, MappedClassifier, FeatureSelectionClassifier, \
     TreeClassifier
from mvpa.clfs.transerror import TransferError
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError

from tests_warehouse import *
from tests_warehouse_clfs import *

# What exceptions to allow while testing degenerate cases.
# If it pukes -- it is ok -- user will notice that something
# is wrong
_degenerate_allowed_exceptions = [DegenerateInputError, FailedToTrainError]
if externals.exists('rpy'):
    import rpy
    _degenerate_allowed_exceptions += [rpy.RPyRException]


class ClassifiersTests(unittest.TestCase):

    def setUp(self):
        self.clf_sign = SameSignClassifier()
        self.clf_less1 = Less1Classifier()

        # simple binary dataset
        self.data_bin_1 = Dataset(
            samples=[[0,0],[-10,-1],[1,0.1],[1,-1],[-1,1]],
            labels=[1, 1, 1, -1, -1], # labels
            chunks=[0, 1, 2,  2, 3])  # chunks

    def testDummy(self):
        clf = SameSignClassifier(enable_states=['training_confusion'])
        clf.train(self.data_bin_1)
        self.failUnlessRaises(UnknownStateError, clf.states.__getattribute__,
                              "predictions")
        """Should have no predictions after training. Predictions
        state should be explicitely disabled"""

        if not _all_states_enabled:
            self.failUnlessRaises(UnknownStateError, clf.states.__getattribute__,
                                  "trained_dataset")

        self.failUnlessEqual(clf.training_confusion.percentCorrect,
                             100,
                             msg="Dummy clf should train perfectly")
        self.failUnlessEqual(clf.predict(self.data_bin_1.samples),
                             list(self.data_bin_1.labels))

        self.failUnlessEqual(len(clf.predictions), self.data_bin_1.nsamples,
            msg="Trained classifier stores predictions by default")

        clf = SameSignClassifier(enable_states=['trained_dataset'])
        clf.train(self.data_bin_1)
        self.failUnless((clf.trained_dataset.samples ==
                         self.data_bin_1.samples).all())
        self.failUnless((clf.trained_dataset.labels ==
                         self.data_bin_1.labels).all())


    def testBoosted(self):
        # XXXXXXX
        # silly test if we get the same result with boosted as with a single one
        bclf = CombinedClassifier(clfs=[self.clf_sign.clone(),
                                        self.clf_sign.clone()])

        self.failUnlessEqual(list(bclf.predict(self.data_bin_1.samples)),
                             list(self.data_bin_1.labels),
                             msg="Boosted classifier should work")
        self.failUnlessEqual(bclf.predict(self.data_bin_1.samples),
                             self.clf_sign.predict(self.data_bin_1.samples),
                             msg="Boosted classifier should have the same as regular")


    def testBoostedStatePropagation(self):
        bclf = CombinedClassifier(clfs=[self.clf_sign.clone(),
                                        self.clf_sign.clone()],
                                  enable_states=['feature_ids'])

        # check states enabling propagation
        self.failUnlessEqual(self.clf_sign.states.isEnabled('feature_ids'),
                             _all_states_enabled)
        self.failUnlessEqual(bclf.clfs[0].states.isEnabled('feature_ids'), True)

        bclf2 = CombinedClassifier(clfs=[self.clf_sign.clone(),
                                         self.clf_sign.clone()],
                                  propagate_states=False,
                                  enable_states=['feature_ids'])

        self.failUnlessEqual(self.clf_sign.states.isEnabled('feature_ids'),
                             _all_states_enabled)
        self.failUnlessEqual(bclf2.clfs[0].states.isEnabled('feature_ids'),
                             _all_states_enabled)



    def testBinaryDecorator(self):
        ds = Dataset(samples=[ [0,0], [0,1], [1,100], [-1,0], [-1,-3], [ 0,-10] ],
                     labels=[ 'sp', 'sp', 'sp', 'dn', 'sn', 'dp'])
        testdata = [ [0,0], [10,10], [-10, -1], [0.1, -0.1], [-0.2, 0.2] ]
        # labels [s]ame/[d]ifferent (sign), and [p]ositive/[n]egative first element

        clf = SameSignClassifier()
        # lets create classifier to descriminate only between same/different,
        # which is a primary task of SameSignClassifier
        bclf1 = BinaryClassifier(clf=clf,
                                 poslabels=['sp', 'sn'],
                                 neglabels=['dp', 'dn'])

        orig_labels = ds.labels[:]
        bclf1.train(ds)

        self.failUnless(bclf1.predict(testdata) ==
                        [['sp', 'sn'], ['sp', 'sn'], ['sp', 'sn'],
                         ['dn', 'dp'], ['dn', 'dp']])

        self.failUnless((ds.labels == orig_labels).all(),
                        msg="BinaryClassifier should not alter labels")


    @sweepargs(clf=clfswh['binary'])
    def testClassifierGeneralization(self, clf):
        """Simple test if classifiers can generalize ok on simple data
        """
        te = CrossValidatedTransferError(TransferError(clf), NFoldSplitter())
        cve = te(datasets['uni2medium'])
        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless(cve < 0.25,
                            msg="Got transfer error %g" % (cve))


    @sweepargs(clf=clfswh[:] + regrswh[:])
    def testSummary(self, clf):
        """Basic testing of the clf summary
        """
        summary1 = clf.summary()
        self.failUnless('not yet trained' in summary1)
        clf.train(datasets['uni2small'])
        summary = clf.summary()
        # It should get bigger ;)
        self.failUnless(len(summary) > len(summary1))
        self.failUnless(not 'not yet trained' in summary)


    @sweepargs(clf=clfswh[:] + regrswh[:])
    def testDegenerateUsage(self, clf):
        """Test how clf handles degenerate cases
        """
        # Whenever we have only 1 feature with only 0s in it
        ds1 = datasets['uni2small'][:, [0]]
        # XXX this very line breaks LARS in many other unittests --
        # very interesting effect. but screw it -- for now it will be
        # this way
        ds1.samples[:] = 0.0             # all 0s

        #ds2 = datasets['uni2small'][[0], :]
        #ds2.samples[:] = 0.0             # all 0s

        clf.states._changeTemporarily(
            enable_states=['values', 'training_confusion'])

        # Good pukes are good ;-)
        # TODO XXX add
        #  - ", ds2):" to test degenerate ds with 1 sample
        #  - ds1 but without 0s -- just 1 feature... feature selections
        #    might lead to 'surprises' due to magic in combiners etc
        for ds in (ds1, ):
            try:
                clf.train(ds)                   # should not crash or stall
                # could we still get those?
                summary = clf.summary()
                cm = clf.states.training_confusion
                # If succeeded to train/predict (due to
                # training_confusion) without error -- results better be
                # at "chance"
                continue
                if 'ACC' in cm.stats:
                    self.failUnlessEqual(cm.stats['ACC'], 0.5)
                else:
                    self.failUnless(N.isnan(cm.stats['CCe']))
            except tuple(_degenerate_allowed_exceptions):
                pass
        clf.states._resetEnabledTemporarily()


    # TODO: validate for regressions as well!!!
    def testSplitClassifier(self):
        ds = self.data_bin_1
        clf = SplitClassifier(clf=SameSignClassifier(),
                splitter=NFoldSplitter(1),
                enable_states=['confusion', 'training_confusion',
                               'feature_ids'])
        clf.train(ds)                   # train the beast
        error = clf.confusion.error
        tr_error = clf.training_confusion.error

        clf2 = clf.clone()
        cv = CrossValidatedTransferError(
            TransferError(clf2),
            NFoldSplitter(),
            enable_states=['confusion', 'training_confusion'])
        cverror = cv(ds)
        tr_cverror = cv.training_confusion.error

        self.failUnlessEqual(error, cverror,
                msg="We should get the same error using split classifier as"
                    " using CrossValidatedTransferError. Got %s and %s"
                    % (error, cverror))

        self.failUnlessEqual(tr_error, tr_cverror,
                msg="We should get the same training error using split classifier as"
                    " using CrossValidatedTransferError. Got %s and %s"
                    % (tr_error, tr_cverror))

        self.failUnlessEqual(clf.confusion.percentCorrect,
                             100,
                             msg="Dummy clf should train perfectly")
        self.failUnlessEqual(len(clf.confusion.sets),
                             len(ds.uniquechunks),
                             msg="Should have 1 confusion per each split")
        self.failUnlessEqual(len(clf.clfs), len(ds.uniquechunks),
                             msg="Should have number of classifiers equal # of epochs")
        self.failUnlessEqual(clf.predict(ds.samples), list(ds.labels),
                             msg="Should classify correctly")

        # feature_ids must be list of lists, and since it is not
        # feature-selecting classifier used - we expect all features
        # to be utilized
        #  NOT ANYMORE -- for BoostedClassifier we have now union of all
        #  used features across slave classifiers. That makes
        #  semantics clear. If you need to get deeper -- use upcoming
        #  harvesting facility ;-)
        # self.failUnlessEqual(len(clf.feature_ids), len(ds.uniquechunks))
        # self.failUnless(N.array([len(ids)==ds.nfeatures
        #                         for ids in clf.feature_ids]).all())

        # Just check if we get it at all ;-)
        summary = clf.summary()


    @sweepargs(clf_=clfswh['binary', '!meta'])
    def testSplitClassifierExtended(self, clf_):
        clf2 = clf_.clone()
        ds = datasets['uni2medium']#self.data_bin_1
        clf = SplitClassifier(clf=clf_, #SameSignClassifier(),
                splitter=NFoldSplitter(1),
                enable_states=['confusion', 'feature_ids'])
        clf.train(ds)                   # train the beast
        error = clf.confusion.error

        cv = CrossValidatedTransferError(
            TransferError(clf2),
            NFoldSplitter(),
            enable_states=['confusion', 'training_confusion'])
        cverror = cv(ds)

        self.failUnless(abs(error-cverror)<0.01,
                msg="We should get the same error using split classifier as"
                    " using CrossValidatedTransferError. Got %s and %s"
                    % (error, cverror))

        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless(error < 0.25,
                msg="clf should generalize more or less fine. "
                    "Got error %s" % error)
        self.failUnlessEqual(len(clf.confusion.sets), len(ds.uniquechunks),
            msg="Should have 1 confusion per each split")
        self.failUnlessEqual(len(clf.clfs), len(ds.uniquechunks),
            msg="Should have number of classifiers equal # of epochs")
        #self.failUnlessEqual(clf.predict(ds.samples), list(ds.labels),
        #                     msg="Should classify correctly")



    def testHarvesting(self):
        """Basic testing of harvesting based on SplitClassifier
        """
        ds = self.data_bin_1
        clf = SplitClassifier(clf=SameSignClassifier(),
                splitter=NFoldSplitter(1),
                enable_states=['confusion', 'training_confusion',
                               'feature_ids'],
                harvest_attribs=['clf.feature_ids',
                                 'clf.training_time'],
                descr="DESCR")
        clf.train(ds)                   # train the beast
        # Number of harvested items should be equal to number of chunks
        self.failUnlessEqual(len(clf.harvested['clf.feature_ids']),
                             len(ds.uniquechunks))
        # if we can blame multiple inheritance and ClassWithCollections.__init__
        self.failUnlessEqual(clf.descr, "DESCR")


    def testMappedClassifier(self):
        samples = N.array([ [0,0,-1], [1,0,1], [-1,-1, 1], [-1,0,1], [1, -1, 1] ])
        testdata3 = Dataset(samples=samples, labels=1)
        res110 = [1, 1, 1, -1, -1]
        res101 = [-1, 1, -1, -1, 1]
        res011 = [-1, 1, -1, 1, -1]

        clf110 = MappedClassifier(clf=self.clf_sign, mapper=MaskMapper(N.array([1,1,0])))
        clf101 = MappedClassifier(clf=self.clf_sign, mapper=MaskMapper(N.array([1,0,1])))
        clf011 = MappedClassifier(clf=self.clf_sign, mapper=MaskMapper(N.array([0,1,1])))

        self.failUnlessEqual(clf110.predict(samples), res110)
        self.failUnlessEqual(clf101.predict(samples), res101)
        self.failUnlessEqual(clf011.predict(samples), res011)


    def testFeatureSelectionClassifier(self):
        from test_rfe import SillySensitivityAnalyzer
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

        samples = N.array([ [0,0,-1], [1,0,1], [-1,-1, 1], [-1,0,1], [1, -1, 1] ])

        testdata3 = Dataset(samples=samples, labels=1)
        # dummy train data so proper mapper gets created
        traindata = Dataset(samples=N.array([ [0, 0,-1], [1,0,1] ]), labels=[1,2])

        # targets
        res110 = [1, 1, 1, -1, -1]
        res011 = [-1, 1, -1, 1, -1]

        # first classifier -- 0th feature should be discarded
        clf011 = FeatureSelectionClassifier(self.clf_sign, feat_sel,
                    enable_states=['feature_ids'])

        self.clf_sign.states._changeTemporarily(enable_states=['values'])
        clf011.train(traindata)

        self.failUnlessEqual(clf011.predict(testdata3.samples), res011)
        # just silly test if we get values assigned in the 'ProxyClassifier'
        self.failUnless(len(clf011.values) == len(res110),
                        msg="We need to pass values into ProxyClassifier")
        self.clf_sign.states._resetEnabledTemporarily()

        self.failUnlessEqual(len(clf011.feature_ids), 2)
        "Feature selection classifier had to be trained on 2 features"

        # first classifier -- last feature should be discarded
        clf011 = FeatureSelectionClassifier(self.clf_sign, feat_sel_rev)
        clf011.train(traindata)
        self.failUnlessEqual(clf011.predict(testdata3.samples), res110)

    def testFeatureSelectionClassifierWithRegression(self):
        from test_rfe import SillySensitivityAnalyzer
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
        dat = Dataset(samples=N.random.randn(4,10),labels=[-1,-1,1,1])
        clf_reg = FeatureSelectionClassifier(sample_clf_reg, feat_sel)
        clf_reg.train(dat)
        res = clf_reg.predict(dat.samples)
        self.failIf((N.array(clf_reg.values)-clf_reg.predictions).sum()==0,
                    msg="Values were set to the predictions in %s." %
                    sample_clf_reg)


    def testTreeClassifier(self):
        """Basic tests for TreeClassifier
        """
        ds = datasets['uni4small']
        # excluding PLR since that one can deal only with 0,1 labels ATM
        clfs = clfswh['binary', '!plr']         # pool of classifiers
        # Lets permute so each time we try some different combination
        # of the classifiers
        clfs = [clfs[i] for i in N.random.permutation(len(clfs))]
        # Test conflicting definition
        tclf = TreeClassifier(clfs[0], {
            'L0+2' : (('L0', 'L2'), clfs[1]),
            'L2+3' : ((2, 3),       clfs[2])})
        self.failUnlessRaises(ValueError, tclf.train, ds)
        """Should raise exception since label 2 is in both"""

        # Test insufficient definition
        tclf = TreeClassifier(clfs[0], {
            'L0+5' : (('L0', 'L5'), clfs[1]),
            'L2+3' : ((2, 3),       clfs[2])})
        self.failUnlessRaises(ValueError, tclf.train, ds)
        """Should raise exception since no group for L1"""

        # proper definition now
        tclf = TreeClassifier(clfs[0], {
            'L0+1' : (('L0', 1), clfs[1]),
            'L2+3' : ((2, 3),    clfs[2])})

        # Lets test train/test cycle using CVTE
        cv = CrossValidatedTransferError(
            TransferError(tclf),
            OddEvenSplitter(),
            enable_states=['confusion', 'training_confusion'])
        cverror = cv(ds)
        try:
            rtclf = repr(tclf)
        except:
            self.fail(msg="Could not obtain repr for TreeClassifier")

        # Test accessibility of .clfs
        self.failUnless(tclf.clfs['L0+1'] is clfs[1])
        self.failUnless(tclf.clfs['L2+3'] is clfs[2])

        cvtrc = cv.training_confusion
        cvtc = cv.confusion
        if cfg.getboolean('tests', 'labile', default='yes'):
            # just a dummy check to make sure everything is working
            self.failUnless(cvtrc != cvtc)
            self.failUnless(cverror < 0.3,
                            msg="Got too high error = %s using %s"
                            % (cverror, tclf))

        # TODO: whenever implemented
        tclf = TreeClassifier(clfs[0], {
            'L0' : (('L0',), clfs[1]),
            'L1+2+3' : ((1, 2, 3),    clfs[2])})
        # TEST ME


    @sweepargs(clf=clfswh[:])
    def testValues(self, clf):
        if isinstance(clf, MulticlassClassifier):
            # TODO: handle those values correctly
            return
        ds = datasets['uni2small']
        clf.states._changeTemporarily(enable_states = ['values'])
        cv = CrossValidatedTransferError(
            TransferError(clf),
            OddEvenSplitter(),
            enable_states=['confusion', 'training_confusion'])
        cverror = cv(ds)
        #print clf.descr, clf.values[0]
        # basic test either we get 1 set of values per each sample
        self.failUnlessEqual(len(clf.values), ds.nsamples/2)

        clf.states._resetEnabledTemporarily()

    @sweepargs(clf=clfswh['linear', 'svm', 'libsvm', '!meta'])
    def testMulticlassClassifier(self, clf):
        oldC = None
        # XXX somewhat ugly way to force non-dataspecific C value.
        # Otherwise multiclass libsvm builtin and our MultiClass would differ
        # in results
        if clf.params.isKnown('C') and clf.C<0:
            oldC = clf.C
            clf.C = 1.0                 # reset C to be 1

        svm, svm2 = clf, clf.clone()
        svm2.states.enable(['training_confusion'])

        mclf = MulticlassClassifier(clf=svm,
                                   enable_states=['training_confusion'])

        svm2.train(datasets['uni2small_train'])
        mclf.train(datasets['uni2small_train'])
        s1 = str(mclf.training_confusion)
        s2 = str(svm2.training_confusion)
        self.failUnlessEqual(s1, s2,
            msg="Multiclass clf should provide same results as built-in "
                "libsvm's %s. Got %s and %s" % (svm2, s1, s2))

        svm2.untrain()

        self.failUnless(svm2.trained == False,
            msg="Un-Trained SVM should be untrained")

        self.failUnless(N.array([x.trained for x in mclf.clfs]).all(),
            msg="Trained Boosted classifier should have all primary classifiers trained")
        self.failUnless(mclf.trained,
            msg="Trained Boosted classifier should be marked as trained")

        mclf.untrain()

        self.failUnless(not mclf.trained,
                        msg="UnTrained Boosted classifier should not be trained")
        self.failUnless(not N.array([x.trained for x in mclf.clfs]).any(),
            msg="UnTrained Boosted classifier should have no primary classifiers trained")

        if oldC is not None:
            clf.C = oldC

    # XXX meta should also work but TODO
    @sweepargs(clf=clfswh['svm', '!meta'])
    def testSVMs(self, clf):
        knows_probabilities = 'probabilities' in clf.states.names and clf.params.probability
        enable_states = ['values']
        if knows_probabilities: enable_states += ['probabilities']

        clf.states._changeTemporarily(enable_states = enable_states)
        for traindata, testdata in [
            (datasets['uni2small_train'], datasets['uni2small_test']) ]:
            clf.train(traindata)
            predicts = clf.predict(testdata.samples)
            # values should be different from predictions for SVMs we have
            self.failUnless( (predicts != clf.values).any() )

            if knows_probabilities and clf.states.isSet('probabilities'):
                # XXX test more thoroughly what we are getting here ;-)
                self.failUnlessEqual( len(clf.probabilities), len(testdata.samples)  )
        clf.states._resetEnabledTemporarily()


    @sweepargs(clf=clfswh['retrainable'])
    def testRetrainables(self, clf):
        # we need a copy since will tune its internals later on
        clf = clf.clone()
        clf.states._changeTemporarily(enable_states = ['values'],
                                      # ensure that it does do predictions
                                      # while training
                                      disable_states=['training_confusion'])
        clf_re = clf.clone()
        # TODO: .retrainable must have a callback to call smth like
        # _setRetrainable
        clf_re._setRetrainable(True)

        # need to have high snr so we don't 'cope' with problematic
        # datasets since otherwise unittests would fail.
        dsargs = {'perlabel':50, 'nlabels':2, 'nfeatures':5, 'nchunks':1,
                  'nonbogus_features':[2,4], 'snr': 5.0}

        ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # NB datasets will be changed by the end of testing, so if
        # are to change to use generic datasets - make sure to copy
        # them here
        dstrain = deepcopy(datasets['uni2large_train'])
        dstest = deepcopy(datasets['uni2large_test'])

        clf.untrain()
        clf_re.untrain()
        trerr, trerr_re = TransferError(clf), \
                          TransferError(clf_re, disable_states=['training_confusion'])

        # Just check for correctness of retraining
        err_1 = trerr(dstest, dstrain)
        self.failUnless(err_1<0.3,
            msg="We should test here on easy dataset. Got error of %s" % err_1)
        values_1 = clf.values[:]
        # some times retraining gets into deeper optimization ;-)
        eps = 0.05
        corrcoef_eps = 0.85             # just to get no failures... usually > 0.95


        def batch_test(retrain=True, retest=True, closer=True):
            err = trerr(dstest, dstrain)
            err_re = trerr_re(dstest, dstrain)
            corr = N.corrcoef(clf.values, clf_re.values)[0,1]
            corr_old = N.corrcoef(values_1, clf_re.values)[0,1]
            if __debug__:
                debug('TEST', "Retraining stats: errors %g %g corr %g "
                      "with old error %g corr %g" %
                  (err, err_re, corr, err_1, corr_old))
            self.failUnless(clf_re.states.retrained == retrain,
                            ("Must fully train",
                             "Must retrain instead of full training")[retrain])
            self.failUnless(clf_re.states.repredicted == retest,
                            ("Must fully test",
                             "Must retest instead of full testing")[retest])
            self.failUnless(corr > corrcoef_eps,
              msg="Result must be close to the one without retraining."
                  " Got corrcoef=%s" % (corr))
            if closer:
                self.failUnless(corr >= corr_old,
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
        if 'C' in clf.params.names:
            clf.params.C *= 0.1
            clf_re.params.C *= 0.1
            batch_test()
        elif 'sigma_noise' in clf.params.names:
            clf.params.sigma_noise *= 100
            clf_re.params.sigma_noise *= 100
            batch_test()
        else:
            raise RuntimeError, \
                  'Please implement testing while changing some of the ' \
                  'params for clf %s' % clf

        # should retrain nicely if we change kernel parameter
        if hasattr(clf, 'kernel_params') and len(clf.kernel_params.names):
            clf.kernel_params.gamma = 0.1
            clf_re.kernel_params.gamma = 0.1
            # retest is false since kernel got recomputed thus
            # can't expect to use the same kernel
            batch_test(retest=not('gamma' in clf.kernel_params.names))

        # should retrain nicely if we change labels
        oldlabels = dstrain.labels[:]
        dstrain.permuteLabels(status=True, assure_permute=True)
        self.failUnless((oldlabels != dstrain.labels).any(),
            msg="We should succeed at permutting -- now got the same labels")
        batch_test()

        # Change labels in testing
        oldlabels = dstest.labels[:]
        dstest.permuteLabels(status=True, assure_permute=True)
        self.failUnless((oldlabels != dstest.labels).any(),
            msg="We should succeed at permutting -- now got the same labels")
        batch_test()

        # should re-train if we change data
        # reuse trained SVM and its 'final' optimization point
        if not clf.__class__.__name__ in ['GPR']: # on GPR everything depends on the data ;-)
            oldsamples = dstrain.samples.copy()
            dstrain.samples[:] += dstrain.samples*0.05
            self.failUnless((oldsamples != dstrain.samples).any())
            batch_test(retest=False)
        clf.states._resetEnabledTemporarily()

        # test retrain()
        # TODO XXX  -- check validity
        clf_re.retrain(dstrain); self.failUnless(clf_re.states.retrained)
        clf_re.retrain(dstrain, labels=True);  self.failUnless(clf_re.states.retrained)
        clf_re.retrain(dstrain, traindataset=True);  self.failUnless(clf_re.states.retrained)

        # test repredict()
        clf_re.repredict(dstest.samples);
        self.failUnless(clf_re.states.repredicted)
        self.failUnlessRaises(RuntimeError, clf_re.repredict,
                              dstest.samples, labels=True,
             msg="for now retesting with anything changed makes no sense")
        clf_re._setRetrainable(False)


    def testGenericTests(self):
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
            #self.failUnless(isinstance(predicted, N.ndarray))

        # Just simple test that all of them are syntaxed correctly
        self.failUnless(str(clf) != "")
        self.failUnless(repr(clf) != "")

        # TODO: unify str and repr for all classifiers

    # XXX TODO: should work on smlr, knn, ridgereg, lars as well! but now
    #     they fail to train
    #    GNB -- cannot train since 1 sample isn't sufficient to assess variance
    @sweepargs(clf=clfswh['!smlr', '!knn', '!gnb', '!lars', '!meta', '!ridge'])
    def testCorrectDimensionsOrder(self, clf):
        """To check if known/present Classifiers are working properly
        with samples being first dimension. Started to worry about
        possible problems while looking at sg where samples are 2nd
        dimension
        """
        # specially crafted dataset -- if dimensions are flipped over
        # the same storage, problem becomes unseparable. Like in this case
        # incorrect order of dimensions lead to equal samples [0, 1, 0]
        traindatas = [
            Dataset(samples=N.array([ [0, 0, 1.0],
                                        [1, 0, 0] ]), labels=[0, 1]),
            Dataset(samples=N.array([ [0, 0.0],
                                      [1, 1] ]), labels=[0, 1])]

        clf.states._changeTemporarily(enable_states = ['training_confusion'])
        for traindata in traindatas:
            clf.train(traindata)
            self.failUnlessEqual(clf.training_confusion.percentCorrect, 100.0,
                "Classifier %s must have 100%% correct learning on %s. Has %f" %
                (`clf`, traindata.samples, clf.training_confusion.percentCorrect))

            # and we must be able to predict every original sample thus
            for i in xrange(traindata.nsamples):
                sample = traindata.samples[i,:]
                predicted = clf.predict([sample])
                self.failUnlessEqual([predicted], traindata.labels[i],
                    "We must be able to predict sample %s using " % sample +
                    "classifier %s" % `clf`)
        clf.states._resetEnabledTemporarily()

def suite():
    return unittest.makeSuite(ClassifiersTests)


if __name__ == '__main__':
    import runner
