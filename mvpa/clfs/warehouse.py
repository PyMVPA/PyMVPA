# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Collection of classifiers to ease the exploration.
"""

__docformat__ = 'restructuredtext'

from sets import Set
import operator

# Define sets of classifiers
from mvpa.clfs.meta import FeatureSelectionClassifier, SplitClassifier, \
     MulticlassClassifier, RegressionAsClassifier
from mvpa.clfs.smlr import SMLR
from mvpa.clfs.knn import kNN
from mvpa.clfs.gnb import GNB
from mvpa.kernels.np import LinearKernel, SquaredExponentialKernel, \
     GeneralizedLinearKernel

# Helpers
from mvpa.base import externals, cfg
from mvpa.measures.anova import OneWayAnova
from mvpa.mappers.fx import absolute_features, maxofabs_sample
from mvpa.clfs.smlr import SMLRWeights
from mvpa.featsel.helpers import FractionTailSelector, \
    FixedNElementTailSelector, RangeElementSelector

from mvpa.featsel.base import SensitivityBasedFeatureSelection

# Kernels
from mvpa.kernels.libsvm import LinearLSKernel, RbfLSKernel, \
     PolyLSKernel, SigmoidLSKernel

_KNOWN_INTERNALS = [ 'knn', 'binary', 'svm', 'linear',
        'smlr', 'does_feature_selection', 'has_sensitivity',
        'multiclass', 'non-linear', 'kernel-based', 'lars',
        'regression', 'regression_based',
        'libsvm', 'sg', 'meta', 'retrainable', 'gpr',
        'notrain2predict', 'ridge', 'blr', 'gnpp', 'enet', 'glmnet',
        'gnb', 'rpy2', 'swig', 'skl', 'lda' ]

class Warehouse(object):
    """Class to keep known instantiated classifiers

    Should provide easy ways to select classifiers of needed kind:
    clfswh['linear', 'svm'] should return all linear SVMs
    clfswh['linear', 'multiclass'] should return all linear classifiers
    capable of doing multiclass classification
    """

    def __init__(self, known_tags=None, matches=None):
        """Initialize warehouse

        Parameters
        ----------
        known_tags : list of str
          List of known tags
        matches : dict
          Optional dictionary of additional matches. E.g. since any
          regression can be used as a binary classifier,
          matches={'binary':['regression']}, would allow to provide
          regressions also if 'binary' was requested
          """
        self._known_tags = Set(known_tags)
        self.__items = []
        self.__keys = Set()
        if matches is None:
            matches = {}
        self.__matches = matches

    def __getitem__(self, *args):
        if isinstance(args[0], tuple):
            args = args[0]

        # so we explicitely handle [:]
        if args == (slice(None),):
            args = []

        # lets remove optional modifier '!'
        dargs = Set([str(x).lstrip('!') for x in args]).difference(
            self._known_tags)

        if len(dargs)>0:
            raise ValueError, "Unknown internals %s requested. Known are %s" % \
                  (list(dargs), list(self._known_tags))

        # dummy implementation for now
        result = []
        # check every known item
        for item in self.__items:
            good = True
            # by default each one counts
            for arg in args:
                # check for rejection first
                if arg.startswith('!'):
                    if (arg[1:] in item.__tags__):
                        good = False
                        break
                    else:
                        continue
                # check for inclusion
                found = False
                for arg in [arg] + self.__matches.get(arg, []):
                    if (arg in item.__tags__):
                        found = True
                        break
                good = found
                if not good:
                    break
            if good:
                result.append(item)
        return result

    def __iadd__(self, item):
        if operator.isSequenceType(item):
            for item_ in item:
                self.__iadd__(item_)
        else:
            if not hasattr(item, '__tags__'):
                raise ValueError, "Cannot register %s " % item + \
                      "which has no __tags__ defined"
            if len(item.__tags__) == 0:
                raise ValueError, "Cannot register %s " % item + \
                      "which has empty __tags__"
            clf_internals = Set(item.__tags__)
            if clf_internals.issubset(self._known_tags):
                self.__items.append(item)
                self.__keys |= clf_internals
            else:
                raise ValueError, 'Unknown clf internal(s) %s' % \
                      clf_internals.difference(self._known_tags)
        return self

    @property
    def internals(self):
        """Known internal tags of the classifiers
        """
        return self.__keys

    def listing(self):
        """Listing (description + internals) of registered items
        """
        return [(x.descr, x.__tags__) for x in self.__items]

    @property
    def items(self):
        """Registered items
        """
        return self.__items

clfswh = Warehouse(known_tags=_KNOWN_INTERNALS) # classifiers
regrswh = Warehouse(known_tags=_KNOWN_INTERNALS) # regressions

# NB:
#  - Nu-classifiers are turned off since for haxby DS default nu
#    is an 'infisible' one
#  - Python's SMLR is turned off for the duration of development
#    since it is slow and results should be the same as of C version
#
clfswh += [ SMLR(lm=0.1, implementation="C", descr="SMLR(lm=0.1)"),
          SMLR(lm=1.0, implementation="C", descr="SMLR(lm=1.0)"),
          #SMLR(lm=10.0, implementation="C", descr="SMLR(lm=10.0)"),
          #SMLR(lm=100.0, implementation="C", descr="SMLR(lm=100.0)"),
          #SMLR(implementation="Python", descr="SMLR(Python)")
          ]

clfswh += \
     [ MulticlassClassifier(clfswh['smlr'][0],
                            descr='Pairs+maxvote multiclass on ' + \
                            clfswh['smlr'][0].descr) ]

if externals.exists('libsvm'):
    from mvpa.clfs import libsvmc as libsvm
    clfswh._known_tags.union_update(libsvm.SVM._KNOWN_IMPLEMENTATIONS.keys())
    clfswh += [libsvm.SVM(descr="libsvm.LinSVM(C=def)", probability=1),
             libsvm.SVM(
                 C=-10.0, descr="libsvm.LinSVM(C=10*def)", probability=1),
             libsvm.SVM(
                 C=1.0, descr="libsvm.LinSVM(C=1)", probability=1),
             libsvm.SVM(svm_impl='NU_SVC',
                        descr="libsvm.LinNuSVM(nu=def)", probability=1)
             ]
    clfswh += [libsvm.SVM(kernel=RbfLSKernel(), descr="libsvm.RbfSVM()"),
             libsvm.SVM(kernel=RbfLSKernel(), svm_impl='NU_SVC',
                        descr="libsvm.RbfNuSVM(nu=def)"),
             libsvm.SVM(kernel=PolyLSKernel(),
                        descr='libsvm.PolySVM()', probability=1),
             #libsvm.svm.SVM(kernel=SigmoidLSKernel(),
             #               svm_impl='C_SVC',
             #               descr='libsvm.SigmoidSVM()'),
             ]

    # regressions
    regrswh._known_tags.union_update(['EPSILON_SVR', 'NU_SVR'])
    regrswh += [libsvm.SVM(svm_impl='EPSILON_SVR', descr='libsvm epsilon-SVR'),
                libsvm.SVM(svm_impl='NU_SVR', descr='libsvm nu-SVR')]

if externals.exists('shogun'):
    from mvpa.clfs import sg
    
    from mvpa.kernels.sg import LinearSGKernel, PolySGKernel, RbfSGKernel
    clfswh._known_tags.union_update(sg.SVM._KNOWN_IMPLEMENTATIONS)

    # some classifiers are not yet ready to be used out-of-the-box in
    # PyMVPA, thus we don't populate warehouse with their instances
    bad_classifiers = [
        'mpd',  # was segfault, now non-training on testcases, and XOR.
                # and was described as "for educational purposes", thus
                # shouldn't be used for real data ;-)
        # Should be a drop-in replacement for lightsvm
        'gpbt', # fails to train for testAnalyzerWithSplitClassifier
                # also 'retraining' doesn't work -- fails to generalize
        'gmnp', # would fail with 'assertion Cache_Size > 2'
                # if shogun < 0.6.3, also refuses to train
        'svrlight', # fails to 'generalize' as a binary classifier
                    # after 'binning'
        'krr', # fails to generalize
        'libsvr'                        # XXXregr removing regressions as classifiers
        ]
    if not externals.exists('sg_fixedcachesize'):
        # would fail with 'assertion Cache_Size > 2' if shogun < 0.6.3
        bad_classifiers.append('gnpp')

    for impl in sg.SVM._KNOWN_IMPLEMENTATIONS:
        # Uncomment the ones to disable
        if impl in bad_classifiers:
            continue
        clfswh += [
            sg.SVM(
                descr="sg.LinSVM(C=def)/%s" % impl, svm_impl=impl),
            sg.SVM(
                C=-10.0, descr="sg.LinSVM(C=10*def)/%s" % impl, svm_impl=impl),
            sg.SVM(
                C=1.0, descr="sg.LinSVM(C=1)/%s" % impl, svm_impl=impl),
            ]
        clfswh += [
            sg.SVM(kernel=RbfSGKernel(),
                   descr="sg.RbfSVM()/%s" % impl, svm_impl=impl),
#            sg.SVM(kernel=RbfSGKernel(),
#                   descr="sg.RbfSVM(gamma=0.1)/%s"
#                    % impl, svm_impl=impl, gamma=0.1),
#           sg.SVM(descr="sg.SigmoidSVM()/%s"
#                   % impl, svm_impl=impl, kernel=SigmoidSGKernel(),),
            ]

    _optional_regressions = []
    if externals.exists('shogun.krr') and externals.versions['shogun'] >= '0.9':
        _optional_regressions += ['krr']
    for impl in ['libsvr'] + _optional_regressions:# \
        # XXX svrlight sucks in SG -- dont' have time to figure it out
        #+ ([], ['svrlight'])['svrlight' in sg.SVM._KNOWN_IMPLEMENTATIONS]:
        regrswh._known_tags.union_update([impl])
        regrswh += [ sg.SVM(svm_impl=impl, descr='sg.LinSVMR()/%s' % impl),
                   #sg.SVM(svm_impl=impl, kernel_type='RBF',
                   #       descr='sg.RBFSVMR()/%s' % impl),
                   ]

if len(clfswh['svm', 'linear']) > 0:
    # if any SVM implementation is known, import default ones
    from mvpa.clfs.svm import *

# lars from R via RPy
if externals.exists('lars'):
    import mvpa.clfs.lars as lars
    from mvpa.clfs.lars import LARS
    for model in lars.known_models:
        # XXX create proper repository of classifiers!
        lars_clf = RegressionAsClassifier(LARS(descr="LARS(%s)" % model, model_type=model),
                                          descr='LARS(model_type=%r) classifier' % model)
        clfswh += lars_clf

        # is a regression, too
        lars_regr = LARS(descr="_LARS(%s)" % model,
                         model_type=model)
        regrswh += lars_regr
        # clfswh += MulticlassClassifier(lars,
        #             descr='Multiclass %s' % lars.descr)

## Still fails unittests battery although overhauled otherwise.
## # enet from R via RPy2
## if externals.exists('elasticnet'):
##     from mvpa.clfs.enet import ENET
##     clfswh += RegressionAsClassifier(ENET(),
##                                      descr="RegressionAsClassifier(ENET())")
##     regrswh += ENET(descr="ENET()")

# glmnet from R via RPy
if externals.exists('glmnet'):
    from mvpa.clfs.glmnet import GLMNET_C, GLMNET_R
    clfswh += GLMNET_C(descr="GLMNET_C()")
    regrswh += GLMNET_R(descr="GLMNET_R()")

if externals.exists('skl'):
    from scikits.learn.lda import LDA as sklLDA
    from mvpa.clfs.skl.base import SKLLearnerAdapter
    clfswh += SKLLearnerAdapter(sklLDA(), tags=['lda', 'linear', 'multiclass', 'binary'],
                                enable_ca=['estimates'],
                                descr='scikits.learn.LDA()_adapter')

# kNN
clfswh += kNN(k=5, descr="kNN(k=5)")
clfswh += kNN(k=5, voting='majority', descr="kNN(k=5, voting='majority')")

clfswh += \
    FeatureSelectionClassifier(
        kNN(),
        SensitivityBasedFeatureSelection(
           SMLRWeights(SMLR(lm=1.0, implementation="C"),
                       postproc=maxofabs_sample()),
           RangeElementSelector(mode='select')),
        descr="kNN on SMLR(lm=1) non-0")

clfswh += \
    FeatureSelectionClassifier(
        kNN(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FractionTailSelector(0.05, mode='select', tail='upper')),
        descr="kNN on 5%(ANOVA)")

clfswh += \
    FeatureSelectionClassifier(
        kNN(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FixedNElementTailSelector(50, mode='select', tail='upper')),
        descr="kNN on 50(ANOVA)")


# GNB
clfswh += GNB(descr="GNB()")
clfswh += GNB(common_variance=True, descr="GNB(common_variance=True)")
clfswh += GNB(prior='uniform', descr="GNB(prior='uniform')")
clfswh += \
    FeatureSelectionClassifier(
        GNB(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FractionTailSelector(0.05, mode='select', tail='upper')),
        descr="GNB on 5%(ANOVA)")


# GPR
if externals.exists('scipy'):
    from mvpa.clfs.gpr import GPR

    regrswh += GPR(kernel=LinearKernel(), descr="GPR(kernel='linear')")
    regrswh += GPR(kernel=SquaredExponentialKernel(),
                   descr="GPR(kernel='sqexp')")

    # Add wrapped GPR as a classifier
    gprcb = RegressionAsClassifier(
        GPR(kernel=GeneralizedLinearKernel()), descr="GPRC(kernel='linear')")
    # lets remove multiclass label from it
    gprcb.__tags__.pop(gprcb.__tags__.index('multiclass'))
    clfswh += gprcb

    # and create a proper multiclass one
    clfswh += MulticlassClassifier(
        RegressionAsClassifier(
            GPR(kernel=GeneralizedLinearKernel())),
        descr="GPRCM(kernel='linear')")

# BLR
from mvpa.clfs.blr import BLR
clfswh += RegressionAsClassifier(BLR(descr="BLR()"),
                                 descr="BLR Classifier")


# SVM stuff

if len(clfswh['linear', 'svm']) > 0:

    linearSVMC = clfswh['linear', 'svm',
                             cfg.get('svm', 'backend', default='libsvm').lower()
                             ][0]

    # "Interesting" classifiers
    clfswh += \
         FeatureSelectionClassifier(
             linearSVMC.clone(),
             SensitivityBasedFeatureSelection(
                SMLRWeights(SMLR(lm=0.1, implementation="C"),
                            postproc=maxofabs_sample()),
                RangeElementSelector(mode='select')),
             descr="LinSVM on SMLR(lm=0.1) non-0")


    clfswh += \
        FeatureSelectionClassifier(
            linearSVMC.clone(),
            SensitivityBasedFeatureSelection(
                SMLRWeights(SMLR(lm=1.0, implementation="C"),
                            postproc=maxofabs_sample()),
                RangeElementSelector(mode='select')),
            descr="LinSVM on SMLR(lm=1) non-0")


    # "Interesting" classifiers
    clfswh += \
        FeatureSelectionClassifier(
            RbfCSVMC(),
            SensitivityBasedFeatureSelection(
               SMLRWeights(SMLR(lm=1.0, implementation="C"),
                           postproc=maxofabs_sample()),
               RangeElementSelector(mode='select')),
            descr="RbfSVM on SMLR(lm=1) non-0")

    clfswh += \
        FeatureSelectionClassifier(
            linearSVMC.clone(),
            SensitivityBasedFeatureSelection(
               OneWayAnova(),
               FractionTailSelector(0.05, mode='select', tail='upper')),
            descr="LinSVM on 5%(ANOVA)")

    clfswh += \
        FeatureSelectionClassifier(
            linearSVMC.clone(),
            SensitivityBasedFeatureSelection(
               OneWayAnova(),
               FixedNElementTailSelector(50, mode='select', tail='upper')),
            descr="LinSVM on 50(ANOVA)")

    clfswh += \
        FeatureSelectionClassifier(
            linearSVMC.clone(),
            SensitivityBasedFeatureSelection(
               linearSVMC.get_sensitivity_analyzer(postproc=maxofabs_sample()),
               FractionTailSelector(0.05, mode='select', tail='upper')),
            descr="LinSVM on 5%(SVM)")

    clfswh += \
        FeatureSelectionClassifier(
            linearSVMC.clone(),
            SensitivityBasedFeatureSelection(
               linearSVMC.get_sensitivity_analyzer(postproc=maxofabs_sample()),
               FixedNElementTailSelector(50, mode='select', tail='upper')),
            descr="LinSVM on 50(SVM)")


    ### Imports which are specific to RFEs
    # from mvpa.datasets.splitters import OddEvenSplitter
    # from mvpa.clfs.transerror import TransferError
    # from mvpa.featsel.rfe import RFE
    # from mvpa.featsel.helpers import FixedErrorThresholdStopCrit
    # from mvpa.clfs.transerror import ConfusionBasedError

    # SVM with unbiased RFE -- transfer-error to another splits, or in
    # other terms leave-1-out error on the same dataset
    # Has to be bound outside of the RFE definition since both analyzer and
    # error should use the same instance.
    rfesvm_split = SplitClassifier(linearSVMC)#clfswh['LinearSVMC'][0])

    # "Almost" classical RFE. If this works it would differ only that
    # our transfer_error is based on internal splitting and classifier used
    # within RFE is a split classifier and its sensitivities per split will get
    # averaged
    #

    #clfswh += \
    #  FeatureSelectionClassifier(
    #    clf = LinearCSVMC(), #clfswh['LinearSVMC'][0],         # we train LinearSVM
    #    feature_selection = RFE(             # on features selected via RFE
    #        # based on sensitivity of a clf which does splitting internally
    #        sensitivity_analyzer=rfesvm_split.get_sensitivity_analyzer(),
    #        transfer_error=ConfusionBasedError(
    #           rfesvm_split,
    #           confusion_state="confusion"),
    #           # and whose internal error we use
    #        feature_selector=FractionTailSelector(
    #                           0.2, mode='discard', tail='lower'),
    #                           # remove 20% of features at each step
    #        update_sensitivity=True),
    #        # update sensitivity at each step
    #    descr='LinSVM+RFE(splits_avg)' )
    #
    #clfswh += \
    #  FeatureSelectionClassifier(
    #    clf = LinearCSVMC(),                 # we train LinearSVM
    #    feature_selection = RFE(             # on features selected via RFE
    #        # based on sensitivity of a clf which does splitting internally
    #        sensitivity_analyzer=rfesvm_split.get_sensitivity_analyzer(),
    #        transfer_error=ConfusionBasedError(
    #           rfesvm_split,
    #           confusion_state="confusion"),
    #           # and whose internal error we use
    #        feature_selector=FractionTailSelector(
    #                           0.2, mode='discard', tail='lower'),
    #                           # remove 20% of features at each step
    #        update_sensitivity=False),
    #        # update sensitivity at each step
    #    descr='LinSVM+RFE(splits_avg,static)' )

    rfesvm = LinearCSVMC()

    # This classifier will do RFE while taking transfer error to testing
    # set of that split. Resultant classifier is voted classifier on top
    # of all splits, let see what that would do ;-)
    #clfswh += \
    #  SplitClassifier(                      # which does splitting internally
    #   FeatureSelectionClassifier(
    #    clf = LinearCSVMC(),
    #    feature_selection = RFE(             # on features selected via RFE
    #        sensitivity_analyzer=\
    #            rfesvm.get_sensitivity_analyzer(postproc=absolute_features()),
    #        transfer_error=TransferError(rfesvm),
    #        stopping_criterion=FixedErrorThresholdStopCrit(0.05),
    #        feature_selector=FractionTailSelector(
    #                           0.2, mode='discard', tail='lower'),
    #                           # remove 20% of features at each step
    #        update_sensitivity=True)),
    #        # update sensitivity at each step
    #    descr='LinSVM+RFE(N-Fold)')
    #
    #
    #clfswh += \
    #  SplitClassifier(                      # which does splitting internally
    #   FeatureSelectionClassifier(
    #    clf = LinearCSVMC(),
    #    feature_selection = RFE(             # on features selected via RFE
    #        sensitivity_analyzer=\
    #            rfesvm.get_sensitivity_analyzer(postproc=absolute_features()),
    #        transfer_error=TransferError(rfesvm),
    #        stopping_criterion=FixedErrorThresholdStopCrit(0.05),
    #        feature_selector=FractionTailSelector(
    #                           0.2, mode='discard', tail='lower'),
    #                           # remove 20% of features at each step
    #        update_sensitivity=True)),
    #        # update sensitivity at each step
    #   splitter = OddEvenSplitter(),
    #   descr='LinSVM+RFE(OddEven)')

