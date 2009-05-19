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
from mvpa.clfs.transerror import TransferError
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.featsel.helpers import FixedNElementTailSelector, \
                                 FractionTailSelector, RangeElementSelector

from mvpa.featsel.rfe import RFE

from mvpa.clfs.meta import SplitClassifier, MulticlassClassifier, \
     FeatureSelectionClassifier
from mvpa.clfs.smlr import SMLR, SMLRWeights
from mvpa.misc.transformers import Absolute
from mvpa.datasets.splitters import NFoldSplitter, NoneSplitter

from mvpa.misc.transformers import Absolute, FirstAxisMean, \
     SecondAxisSumOfAbs, DistPValue

from mvpa.measures.base import SplitFeaturewiseDatasetMeasure
from mvpa.measures.anova import OneWayAnova, CompoundOneWayAnova
from mvpa.measures.irelief import IterativeRelief, IterativeReliefOnline, \
     IterativeRelief_Devel, IterativeReliefOnline_Devel

from tests_warehouse import *
from tests_warehouse_clfs import *

_MEASURES_2_SWEEP = [ OneWayAnova(),
                      CompoundOneWayAnova(combiner=SecondAxisSumOfAbs),
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
        self.failUnless(f.shape == (4,))
        self.failUnless(abs(f[1]) <= 1e-12, # some small value
            msg="Failed test with value %g instead of != 0.0" % f[1])
        self.failUnless(f[0] > 0.1)     # some reasonably large value

        # we should not have NaNs
        self.failUnless(not N.any(N.isnan(f)))


    # XXX meta should work too but doesn't
    @sweepargs(clf=clfswh['has_sensitivity'])
    def testAnalyzerWithSplitClassifier(self, clf):
        """Test analyzers in split classifier
        """
        # assumming many defaults it is as simple as
        mclf = SplitClassifier(clf=clf,
                               enable_states=['training_confusion',
                                              'confusion'])
        sana = mclf.getSensitivityAnalyzer(transformer=Absolute,
                                           enable_states=["sensitivities"])

        # Test access to transformers and combiners
        self.failUnless(sana.transformer is Absolute)
        self.failUnless(sana.combiner is FirstAxisMean)
        # and lets look at all sensitivities

        # and we get sensitivity analyzer which works on splits
        map_ = sana(self.dataset)
        self.failUnlessEqual(len(map_), self.dataset.nfeatures)

        if cfg.getboolean('tests', 'labile', default='yes'):
            for conf_matrix in [sana.clf.training_confusion] \
                              + sana.clf.confusion.matrices:
                self.failUnless(
                    conf_matrix.percentCorrect>75,
                    msg="We must have trained on each one more or " \
                    "less correctly. Got %f%% correct on %d labels" %
                    (conf_matrix.percentCorrect,
                     len(self.dataset.uniquelabels)))

        errors = [x.percentCorrect
                    for x in sana.clf.confusion.matrices]

        # XXX
        # That is too much to ask if the dataset is easy - thus
        # disabled for now
        #self.failUnless(N.min(errors) != N.max(errors),
        #                msg="Splits should have slightly but different " \
        #                    "generalization")

        # lets go through all sensitivities and see if we selected the right
        # features
        # XXX yoh: disabled checking of each map separately since in
        #     BoostedClassifierSensitivityAnalyzer and
        #     ProxyClassifierSensitivityAnalyzer
        #     we don't have yet way to provide transformers thus internal call
        #     to getSensitivityAnalyzer in _call of them is not parametrized
        if 'meta' in clf._clf_internals and len(map_.nonzero()[0])<2:
            # Some meta classifiers (5% of ANOVA) are too harsh ;-)
            return
        for map__ in [map_]: # + sana.combined_analyzer.sensitivities:
            selected = FixedNElementTailSelector(
                self.dataset.nfeatures -
                len(self.dataset.nonbogus_features))(map__)
            if cfg.getboolean('tests', 'labile', default='yes'):
                self.failUnlessEqual(
                    list(selected),
                    list(self.dataset.nonbogus_features),
                    msg="At the end we should have selected the right features")


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

        sana = mclf.getSensitivityAnalyzer(transformer=Absolute,
                                           enable_states=["sensitivities"])
        # and lets look at all sensitivities

        dataset = datasets['uni2medium']
        # and we get sensitivity analyzer which works on splits
        map_ = sana(dataset)
        self.failUnlessEqual(len(map_), dataset.nfeatures)



    @sweepargs(svm=clfswh['linear', 'svm'])
    def testLinearSVMWeights(self, svm):
        # assumming many defaults it is as simple as
        sana = svm.getSensitivityAnalyzer(enable_states=["sensitivities"] )

        # and lets look at all sensitivities
        map_ = sana(self.dataset)
        # for now we can do only linear SVM, so lets check if we raise
        # a concern
        svmnl = clfswh['non-linear', 'svm'][0]
        self.failUnlessRaises(NotImplementedError,
                              svmnl.getSensitivityAnalyzer)


    @sweepargs(svm=clfswh['linear', 'svm'])
    def testLinearSVMWeights(self, svm):
        # assumming many defaults it is as simple as
        sana = svm.getSensitivityAnalyzer(enable_states=["sensitivities"] )

        # and lets look at all sensitivities
        map_ = sana(self.dataset)
        # for now we can do only linear SVM, so lets check if we raise
        # a concern
        svmnl = clfswh['non-linear', 'svm'][0]
        self.failUnlessRaises(NotImplementedError,
                              svmnl.getSensitivityAnalyzer)

    # XXX doesn't work easily with meta since it would need
    #     to be explicitely passed to the slave classifier's
    #     getSengetSensitivityAnalyzer
    @sweepargs(svm=clfswh['linear', 'svm', 'libsvm', '!sg', '!meta'])
    def testLinearSVMWeightsPerClass(self, svm):
        # assumming many defaults it is as simple as
        kwargs = dict(combiner=None, transformer=None,
                      enable_states=["sensitivities"])
        sana_split = svm.getSensitivityAnalyzer(
            split_weights=True, **kwargs)
        sana_full = svm.getSensitivityAnalyzer(
            force_training=False, **kwargs)

        # and lets look at all sensitivities
        ds2 = datasets['uni4large'].copy()
        ds2.zscore(baselinelabels = [2, 3])
        ds2 = ds2['labels', [0,1]]

        map_split = sana_split(ds2)
        map_full = sana_full(ds2)

        self.failUnlessEqual(map_split.shape, (ds2.nfeatures, 2))
        self.failUnlessEqual(map_full.shape,  (ds2.nfeatures, ))

        # just to verify that we split properly and if we reconstruct
        # manually we obtain the same
        dmap = (-1*map_split[:, 1]  + map_split[:, 0]) - map_full
        self.failUnless((N.abs(dmap) <= 1e-10).all())
        #print "____"
        #print map_split
        #print SMLR().getSensitivityAnalyzer(combiner=None)(ds2)

        # for now we can do split weights for binary tasks only, so
        # lets check if we raise a concern
        self.failUnlessRaises(NotImplementedError,
                              sana_split, datasets['uni3medium'])


    def testSplitFeaturewiseDatasetMeasure(self):
        ds = datasets['uni3small']
        sana = SplitFeaturewiseDatasetMeasure(
            analyzer=SMLR(
              fit_all_weights=True).getSensitivityAnalyzer(combiner=None),
            splitter=NFoldSplitter(),
            combiner=None)

        sens = sana(ds)

        self.failUnless(sens.shape == (
            len(ds.uniquechunks), ds.nfeatures, len(ds.uniquelabels)))


        # Lets try more complex example with 'boosting'
        ds = datasets['uni3medium']
        sana = SplitFeaturewiseDatasetMeasure(
            analyzer=SMLR(
              fit_all_weights=True).getSensitivityAnalyzer(combiner=None),
            splitter=NoneSplitter(nperlabel=0.25, mode='first',
                                  nrunspersplit=2),
            combiner=None,
            enable_states=['splits', 'sensitivities'])
        sens = sana(ds)

        self.failUnless(sens.shape == (2, ds.nfeatures, 3))
        splits = sana.splits
        self.failUnlessEqual(len(splits), 2)
        self.failUnless(N.all([s[0].nsamples == ds.nsamples/4 for s in splits]))
        # should have used different samples
        self.failUnless(N.any([splits[0][0].origids != splits[1][0].origids]))
        # and should have got different sensitivities
        self.failUnless(N.any(sens[0] != sens[1]))


        if not externals.exists('scipy'):
            return
        # Most evil example
        ds = datasets['uni2medium']
        plain_sana = SVM().getSensitivityAnalyzer(
               combiner=None, transformer=DistPValue())
        boosted_sana = SplitFeaturewiseDatasetMeasure(
            analyzer=SVM().getSensitivityAnalyzer(
               combiner=None, transformer=DistPValue(fpp=0.05)),
            splitter=NoneSplitter(nperlabel=0.8, mode='first', nrunspersplit=2),
            combiner=FirstAxisMean,
            enable_states=['splits', 'sensitivities'])
        # lets create feature selector
        fsel = RangeElementSelector(upper=0.05, lower=0.95, inclusive=True)

        sanas = dict(plain=plain_sana, boosted=boosted_sana)
        for k,sana in sanas.iteritems():
            clf = FeatureSelectionClassifier(SVM(),
                        SensitivityBasedFeatureSelection(sana, fsel),
                        descr='SVM on p=0.01(both tails) using %s' % k)
            ce = CrossValidatedTransferError(TransferError(clf),
                                             NFoldSplitter())
            error = ce(ds)

        sens = boosted_sana(ds)
        sens_plain = plain_sana(ds)

        # TODO: make a really unittest out of it -- not just runtime
        #       bugs catcher

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
                    SMLRWeights(SMLR(lm=1, implementation="C")),
                    RangeElementSelector(mode='select'))]

        fs = CombinedFeatureSelection(fss, combiner='union',
                                      enable_states=['selected_ids',
                                                     'selections_ids'])

        od, otd = fs(self.dataset)

        self.failUnless(fs.combiner == 'union')
        self.failUnless(len(fs.selections_ids))
        self.failUnless(len(fs.selections_ids) <= self.dataset.nfeatures)
        # should store one set per methods
        self.failUnless(len(fs.selections_ids) == len(fss))
        # no individual can be larger than union
        for s in fs.selections_ids:
            self.failUnless(len(s) <= len(fs.selected_ids))
        # check output dataset
        self.failUnless(od.nfeatures == len(fs.selected_ids))
        for i, id in enumerate(fs.selected_ids):
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

