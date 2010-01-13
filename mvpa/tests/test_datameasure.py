# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SplittingSensitivityAnalyzer"""

from mvpa.base import externals
from mvpa.featsel.base import FeatureSelectionPipeline, \
     SensitivityBasedFeatureSelection, CombinedFeatureSelection
from mvpa.featsel.helpers import FixedNElementTailSelector, \
                                 FractionTailSelector, RangeElementSelector

from mvpa.featsel.rfe import RFE

from mvpa.clfs.meta import SplitClassifier, MulticlassClassifier, \
     FeatureSelectionClassifier
from mvpa.clfs.smlr import SMLR, SMLRWeights
from mvpa.mappers.fx import sumofabs_sample, absolute_features, FxMapper, \
     maxofabs_sample
from mvpa.datasets.splitters import NFoldSplitter, NoneSplitter

from mvpa.misc.transformers import Absolute, \
     SecondAxisSumOfAbs, DistPValue

from mvpa.measures.base import SplitFeaturewiseDatasetMeasure
from mvpa.measures.anova import OneWayAnova, CompoundOneWayAnova
from mvpa.measures.irelief import IterativeRelief, IterativeReliefOnline, \
     IterativeRelief_Devel, IterativeReliefOnline_Devel

from tests_warehouse import *
from tests_warehouse_clfs import *

from nose.tools import assert_equal
from numpy.testing import assert_array_equal

_MEASURES_2_SWEEP = [ OneWayAnova(),
                      CompoundOneWayAnova(mapper=sumofabs_sample()),
                      IterativeRelief(), IterativeReliefOnline(),
                      IterativeRelief_Devel(), IterativeReliefOnline_Devel()
                      ]
if externals.exists('scipy'):
    from mvpa.measures.corrcoef import CorrCoef
    _MEASURES_2_SWEEP += [ CorrCoef(),
                           # that one is good when small... handle later
                           #CorrCoef(pvalue=True)
                           ]

class SensitivityAnalysersTests(unittest.TestCase):

    def setUp(self):
        self.dataset = datasets['uni2large']


    @sweepargs(dsm=_MEASURES_2_SWEEP)
    def testBasic(self, dsm):
        data = datasets['dumbinv']
        datass = data.samples.copy()

        # compute scores
        f = dsm(data)
        # check if nothing evil is done to dataset
        self.failUnless(N.all(data.samples == datass))
        self.failUnless(f.shape == (1, data.nfeatures))
        self.failUnless(abs(f.samples[0, 1]) <= 1e-12, # some small value
            msg="Failed test with value %g instead of != 0.0" % f.samples[0, 1])
        self.failUnless(f[0] > 0.1)     # some reasonably large value

        # we should not have NaNs
        self.failUnless(not N.any(N.isnan(f)))



    # NOTE: lars with stepwise used to segfault if all states are enabled
    @sweepargs(clfds=
               [(c, datasets['uni2large'])
                for c in clfswh['has_sensitivity', 'binary']] +
               [(c, datasets['uni4large'])
                for c in clfswh['has_sensitivity', 'multiclass']])
    def testAnalyzerWithSplitClassifier(self, clfds):
        """Test analyzers in split classifier
        """
        clf, ds = clfds             # unroll the tuple
        # We need to skip some LARSes here
        _sclf = str(clf)
        if 'LARS(' in _sclf and "type='stepwise'" in _sclf:
            return

        # To don't waste too much time testing lets limit to 3 splits
        nsplits = 3
        splitter = NFoldSplitter(count=nsplits)
        mclf = SplitClassifier(clf=clf,
                               splitter=splitter,
                               enable_states=['training_confusion',
                                              'confusion'])
        sana = mclf.getSensitivityAnalyzer(mapper=absolute_features(),
                                           enable_states=["sensitivities"])

        nlabels = len(ds.uniquelabels)
        # Can't rely on splitcfg since count-limit is done in __call__
        assert(nsplits == len(list(splitter(ds))))
        sens = sana(ds)

        # It should return either ...
        #  nlabels * nsplits
        req_nsamples = [ nlabels * nsplits ]
        if nlabels == 2:
            # A single sensitivity in case of binary
            req_nsamples += [ nsplits ]
        else:
            # and for pairs in case of multiclass
            req_nsamples += [ (nlabels * (nlabels-1) / 2) * nsplits ]
            # Also for regression_based -- they can do multiclass
            # but only 1 sensitivity is provided
            if 'regression_based' in clf.__tags__:
                req_nsamples += [ nsplits ]

        # # of features should correspond
        self.failUnlessEqual(sens.shape[1], ds.nfeatures)
        # # of samples/sensitivities should also be reasonable
        self.failUnless(sens.shape[0] in req_nsamples)

        # Check if labels are present
        self.failUnless('splits' in sens.sa)
        self.failUnless('labels' in sens.sa)

        assert_array_equal(sens.sa['labels'].unique, ds.sa['labels'].unique)

        errors = [x.percentCorrect
                    for x in sana.clf.states.confusion.matrices]

        # lets go through all sensitivities and see if we selected the right
        # features
        #if 'meta' in clf.__tags__ and len(sens.samples[0].nonzero()[0])<2:
        if '5%' in clf.descr \
               or (nlabels > 2 and 'regression_based' in clf.__tags__):
            # Some meta classifiers (5% of ANOVA) are too harsh ;-)
            # if we get less than 2 features with on-zero sensitivities we
            # cannot really test
            # Also -- regression based classifiers performance for multiclass
            # is expected to suck in general
            return

        if cfg.getboolean('tests', 'labile', default='yes'):
            for conf_matrix in [sana.clf.states.training_confusion] \
                              + sana.clf.states.confusion.matrices:
                self.failUnless(
                    conf_matrix.percentCorrect>=70,
                    msg="We must have trained on each one more or " \
                    "less correctly. Got %f%% correct on %d labels" %
                    (conf_matrix.percentCorrect,
                     nlabels))


        # Since  now we have per split and possibly per label -- lets just find
        # mean per each feature per split
        sensm = FxMapper('samples', lambda x: N.abs(x).sum(),
                         uattrs=['splits'])(sens)
        sensgm = maxofabs_sample()(sensm)    # global max of abs of means

        assert_equal(sensgm.shape[0], 1)
        selected = FixedNElementTailSelector(
            len(ds.a.bogus_features))(sensgm.samples[0])

        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnlessEqual(
                set(selected), set(ds.a.nonbogus_features),
                msg="At the end we should have selected the right features. "
                "Chose %s whenever nonbogus are %s"
                % (selected, ds.a.nonbogus_features))

        # Now test each one per label -- for a given sensitivity we cannot guarantee
        # TODO


    @sweepargs(clf=clfswh['has_sensitivity'])
    def testMappedClassifierSensitivityAnalyzer(self, clf):
        """Test sensitivity of the mapped classifier
        """
        # Assuming many defaults it is as simple as
        mclf = FeatureSelectionClassifier(
            clf,
            SensitivityBasedFeatureSelection(
                OneWayAnova(),
                FractionTailSelector(0.5, mode='select', tail='upper')),
            enable_states=['training_confusion'])

        sana = mclf.getSensitivityAnalyzer(mapper=sumofabs_sample(),
                                           enable_states=["sensitivities"])
        # and lets look at all sensitivities

        dataset = datasets['uni2medium']
        # and we get sensitivity analyzer which works on splits
        sens = sana(dataset)
        self.failUnlessEqual(sens.shape, (1, dataset.nfeatures))



    @sweepargs(svm=clfswh['linear', 'svm'])
    def testLinearSVMWeights(self, svm):
        # assumming many defaults it is as simple as
        sana = svm.getSensitivityAnalyzer(enable_states=["sensitivities"] )
        # and lets look at all sensitivities
        sens = sana(self.dataset)
        # for now we can do only linear SVM, so lets check if we raise
        # a concern
        svmnl = clfswh['non-linear', 'svm'][0]
        self.failUnlessRaises(NotImplementedError,
                              svmnl.getSensitivityAnalyzer)


    # XXX doesn't work easily with meta since it would need
    #     to be explicitely passed to the slave classifier's
    #     getSengetSensitivityAnalyzer
    # Note: only libsvm interface supports split_weights
    @sweepargs(svm=clfswh['linear', 'svm', 'libsvm', '!sg', '!meta'])
    def testLinearSVMWeightsPerClass(self, svm):
        # assumming many defaults it is as simple as
        kwargs = dict(enable_states=["sensitivities"])
        sana_split = svm.getSensitivityAnalyzer(
            split_weights=True, **kwargs)
        sana_full = svm.getSensitivityAnalyzer(
            force_training=False, **kwargs)

        # and lets look at all sensitivities
        ds2 = datasets['uni4large'].copy()
        ds2.zscore(baselinelabels = ['L2', 'L3'])
        ds2 = ds2[N.logical_or(ds2.sa.labels == 'L0', ds2.sa.labels == 'L1')]

        senssplit = sana_split(ds2)
        sensfull = sana_full(ds2)

        self.failUnlessEqual(senssplit.shape, (2, ds2.nfeatures))
        self.failUnlessEqual(sensfull.shape,  (1, ds2.nfeatures))

        # just to verify that we split properly and if we reconstruct
        # manually we obtain the same
        dmap = (-1 * senssplit.samples[1]  + senssplit.samples[0]) \
               - sensfull.samples
        self.failUnless((N.abs(dmap) <= 1e-10).all())
        #print "____"
        #print senssplit
        #print SMLR().getSensitivityAnalyzer(combiner=None)(ds2)

        # for now we can do split weights for binary tasks only, so
        # lets check if we raise a concern
        # we temporarily shutdown warning, since it is going to complain
        # otherwise, but we do it on purpose here
        handlers = warning.handlers
        warning.handlers = []
        self.failUnlessRaises(NotImplementedError,
                              sana_split, datasets['uni3medium'])
        # reenable the warnings
        warning.handlers = handlers


    def testSplitFeaturewiseDatasetMeasure(self):
        ds = datasets['uni3small']
        sana = SplitFeaturewiseDatasetMeasure(
            analyzer=SMLR(
              fit_all_weights=True).getSensitivityAnalyzer(),
            splitter=NFoldSplitter(),
            )

        sens = sana(ds)
        # a sensitivity for each chunk and each label combination
        assert_equal(sens.shape,
                     (len(ds.sa['chunks'].unique) * len(ds.sa['labels'].unique),
                      ds.nfeatures))

        # Lets try more complex example with 'boosting'
        ds = datasets['uni3medium']
        ds.init_origids('samples')
        sana = SplitFeaturewiseDatasetMeasure(
            analyzer=SMLR(
              fit_all_weights=True).getSensitivityAnalyzer(),
            splitter=NoneSplitter(nperlabel=0.25, mode='first',
                                  nrunspersplit=2),
            enable_states=['splits', 'sensitivities'])
        sens = sana(ds)

        assert_equal(sens.shape, (2 * len(ds.sa['labels'].unique),
                                  ds.nfeatures))
        splits = sana.states.splits
        self.failUnlessEqual(len(splits), 2)
        self.failUnless(N.all([s[0].nsamples == ds.nsamples/4 for s in splits]))
        # should have used different samples
        self.failUnless(N.any([splits[0][0].sa.origids != splits[1][0].sa.origids]))
        # and should have got different sensitivities
        self.failUnless(N.any(sens[0] != sens[1]))


        if not externals.exists('scipy'):
            return
        # Let's disable this one for now until we are sure about the destiny of
        # DistPValue -- read the docstring of it!
        # Most evil example
        #ds = datasets['uni2medium']
        #plain_sana = SVM().getSensitivityAnalyzer(
        #       transformer=DistPValue())
        #boosted_sana = SplitFeaturewiseDatasetMeasure(
        #    analyzer=SVM().getSensitivityAnalyzer(
        #       transformer=DistPValue(fpp=0.05)),
        #    splitter=NoneSplitter(nperlabel=0.8, mode='first', nrunspersplit=2),
        #    enable_states=['splits', 'sensitivities'])
        ## lets create feature selector
        #fsel = RangeElementSelector(upper=0.05, lower=0.95, inclusive=True)

        #sanas = dict(plain=plain_sana, boosted=boosted_sana)
        #for k,sana in sanas.iteritems():
        #    clf = FeatureSelectionClassifier(SVM(),
        #                SensitivityBasedFeatureSelection(sana, fsel),
        #                descr='SVM on p=0.01(both tails) using %s' % k)
        #    ce = CrossValidatedTransferError(TransferError(clf),
        #                                     NFoldSplitter())
        #    error = ce(ds)

        #sens = boosted_sana(ds)
        #sens_plain = plain_sana(ds)

        ## TODO: make a really unittest out of it -- not just runtime
        ##       bugs catcher

    # TODO -- unittests for sensitivity analyzers which use combiners
    # (linsvmweights for multi-class SVMs and smlrweights for SMLR)


    @sweepargs(basic_clf=clfswh['has_sensitivity'])
    def __testFSPipelineWithAnalyzerWithSplitClassifier(self, basic_clf):
        #basic_clf = LinearNuSVMC()
        multi_clf = MulticlassClassifier(clf=basic_clf)
        #svm_weigths = LinearSVMWeights(svm)

        # Proper RFE: aggregate sensitivities across multiple splits,
        # but also due to multi class those need to be aggregated
        # somehow. Transfer error here should be 'leave-1-out' error
        # of split classifier itself
        sclf = SplitClassifier(clf=basic_clf)
        rfe = RFE(sensitivity_analyzer=
                    sclf.getSensitivityAnalyzer(
                        enable_states=["sensitivities"]),
                  transfer_error=trans_error,
                  feature_selector=FeatureSelectionPipeline(
                      [FractionTailSelector(0.5),
                       FixedNElementTailSelector(1)]),
                  train_clf=True)

        # and we get sensitivity analyzer which works on splits and uses
        # sensitivity
        selected_features = rfe(self.dataset)

    def testUnionFeatureSelection(self):
        # two methods: 5% highes F-scores, non-zero SMLR weights
        fss = [SensitivityBasedFeatureSelection(
                    OneWayAnova(),
                    FractionTailSelector(0.05, mode='select', tail='upper')),
               SensitivityBasedFeatureSelection(
                    SMLRWeights(SMLR(lm=1, implementation="C"),
                                mapper=sumofabs_sample()),
                    RangeElementSelector(mode='select'))]

        fs = CombinedFeatureSelection(fss, combiner='union',
                                      enable_states=['selected_ids',
                                                     'selections_ids'])

        od, otd = fs(self.dataset)

        self.failUnless(fs.combiner == 'union')
        self.failUnless(len(fs.states.selections_ids))
        self.failUnless(len(fs.states.selections_ids) <= self.dataset.nfeatures)
        # should store one set per methods
        self.failUnless(len(fs.states.selections_ids) == len(fss))
        # no individual can be larger than union
        for s in fs.states.selections_ids:
            self.failUnless(len(s) <= len(fs.states.selected_ids))
        # check output dataset
        self.failUnless(od.nfeatures == len(fs.states.selected_ids))
        for i, id in enumerate(fs.states.selected_ids):
            self.failUnless((od.samples[:,i]
                             == self.dataset.samples[:,id]).all())

        # again for intersection
        fs = CombinedFeatureSelection(fss, combiner='intersection',
                                      enable_states=['selected_ids',
                                                     'selections_ids'])
        # simply run it for now -- can't think of additional tests
        od, otd = fs(self.dataset)



def suite():
    return unittest.makeSuite(SensitivityAnalysersTests)


if __name__ == '__main__':
    import runner

