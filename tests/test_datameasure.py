#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SplittingSensitivityAnalyzer"""

from mvpa.base import externals
from mvpa.featsel.base import FeatureSelectionPipeline
from mvpa.featsel.helpers import FixedNElementTailSelector, \
                                 FractionTailSelector
from mvpa.featsel.rfe import RFE

from mvpa.clfs.base import SplitClassifier, MulticlassClassifier
from mvpa.misc.transformers import Absolute
from mvpa.datasets.splitter import NFoldSplitter

from mvpa.misc.transformers import Absolute

from mvpa.measures.anova import OneWayAnova
from mvpa.measures.irelief import IterativeRelief, IterativeReliefOnline

from tests_warehouse import *
from tests_warehouse_clfs import *

_MEASURES_2_SWEEP = [ OneWayAnova, IterativeRelief, IterativeReliefOnline ]
if externals.exists('scipy'):
    from mvpa.measures.corrcoef import CorrCoef
    _MEASURES_2_SWEEP += [ CorrCoef ]

class SensitivityAnalysersTests(unittest.TestCase):

    def setUp(self):
        self.dataset = datasets['uni2large']


    @sweepargs(saclass=_MEASURES_2_SWEEP)
    def testBasic(self, saclass):
        data = datasets['dumb']
        dsm = saclass()

        # compute scores
        f = dsm(data)

        self.failUnless(f.shape == (2,))
        self.failUnless(abs(f[1]) <= 1e-12, # some small value
            msg="Failed test with value %g instead of != 0.0" % f[1])
        self.failUnless(f[0] > 0.1)     # some reasonably large value


    # XXX meta should work too but doesn't
    @sweepargs(clf=clfs['has_sensitivity', '!meta'])
    def testAnalyzerWithSplitClassifier(self, clf):

        # assumming many defaults it is as simple as
        mclf = SplitClassifier(clf=clf,
                               enable_states=['training_confusion',
                                              'confusion'])
        sana = mclf.getSensitivityAnalyzer(transformer=Absolute, enable_states=["sensitivities"])
        # and lets look at all sensitivities

        # and we get sensitivity analyzer which works on splits
        map_ = sana(self.dataset)
        self.failUnlessEqual(len(map_), self.dataset.nfeatures)

        for conf_matrix in [sana.clf.training_confusion] \
                          + sana.clf.confusion.matrices:
            self.failUnless(conf_matrix.percentCorrect>75,
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
        #     BoostedClassifierSensitivityAnalyzer and ProxyClassifierSensitivityAnalyzer
        #     we don't have yet way to provide transformers thus internal call to
        #     getSensitivityAnalyzer in _call of them is not parametrized
        for map__ in [map_]: # + sana.combined_analyzer.sensitivities:
            selected = FixedNElementTailSelector(
                self.dataset.nfeatures -
                len(self.dataset.nonbogus_features))(map__)
            self.failUnlessEqual(
                list(selected),
                list(self.dataset.nonbogus_features),
                msg="At the end we should have selected the right features")


    @sweepargs(svm=clfs['linear', 'svm'])
    def testLinearSVMWeights(self, svm):
        # assumming many defaults it is as simple as
        sana = svm.getSensitivityAnalyzer(enable_states=["sensitivities"] )

        # and lets look at all sensitivities
        map_ = sana(self.dataset)
        # for now we can do only linear SVM, so lets check if we raise
        # a concern
        svmnl = clfs['non-linear', 'svm'][0]
        self.failUnlessRaises(NotImplementedError,
                              svmnl.getSensitivityAnalyzer)


    # TODO -- unittests for sensitivity analyzers which use combiners
    # (linsvmweights for multi-class SVMs and smlrweights for SMLR)


    @sweepargs(basic_clf=clfs['has_sensitivity'])
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


def suite():
    return unittest.makeSuite(SensitivityAnalysersTests)


if __name__ == '__main__':
    import runner

