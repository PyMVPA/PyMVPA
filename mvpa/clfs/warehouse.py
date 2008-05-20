#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
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

# Data
from mvpa.datasets.splitter import OddEvenSplitter

# Define sets of classifiers
from mvpa.clfs.base import FeatureSelectionClassifier, SplitClassifier, \
                                 MulticlassClassifier
from mvpa.clfs.smlr import SMLR
from mvpa.clfs.knn import kNN
from mvpa.clfs.gpr import GPR

# Helpers
from mvpa.clfs.transerror import TransferError
from mvpa.base import externals
from mvpa.measures.anova import OneWayAnova
from mvpa.misc.transformers import Absolute
from mvpa.featsel.rfe import RFE
from mvpa.clfs.smlr import SMLRWeights
from mvpa.featsel.helpers import FractionTailSelector, \
    FixedNElementTailSelector, RangeElementSelector, \
    FixedErrorThresholdStopCrit
from mvpa.clfs.transerror import ConfusionBasedError
from mvpa.featsel.base import SensitivityBasedFeatureSelection

class Warehouse(object):
    """Class to keep known instantiated classifiers

    Should provide easy ways to select classifiers of needed kind:
    clfs['linear', 'svm'] should return all linear SVMs
    clfs['linear', 'multiclass'] should return all linear classifiers
     capable of doing multiclass classification
     """

    _KNOWN_INTERNALS = Set(['knn', 'binary', 'svm', 'linear', 'smlr', 'does_feature_selection',
                            'has_sensitivity', 'multiclass', 'non-linear', 'kernel-based', 'lars',
                            'regression', 'libsvm', 'sg', 'meta', 'retrainable', 'gpr'])

    def __init__(self):
        self.__items = []
        self.__keys = Set()

    def __getitem__(self, *args):
        if isinstance(args[0], tuple):
            args = args[0]

        # so we explicitely handle [:]
        if args == (slice(None),):
            args = []

        # lets remove optional modifier '!'
        dargs = Set([x.lstrip('!') for x in args]).difference(self._KNOWN_INTERNALS)

        if len(dargs)>0:
            raise ValueError, "Unknown internals %s requested. Known are %s" % \
                  (list(dargs), list(self._KNOWN_INTERNALS))

        # dummy implementation for now
        result = []
        for item in self.__items:
            good = True
            for arg in args:
                if (arg.startswith('!') and \
                    (arg[1:] in item._clf_internals)) or \
                    (not arg.startswith('!') and \
                     (not (arg in item._clf_internals))):
                    good = False
                    break
            if good:
                result.append(item)
        return result

    def __iadd__(self, item):
        if operator.isSequenceType(item):
            for item_ in item:
                self.__iadd__(item_)
        else:
            if not hasattr(item, '_clf_internals'):
                raise ValueError, "Cannot register %s " % item + \
                      "which has no _clf_internals defined"
            if len(item._clf_internals) == 0:
                raise ValueError, "Cannot register %s " % item + \
                      "which has empty _clf_internals"
            clf_internals = Set(item._clf_internals)
            if clf_internals.issubset(self._KNOWN_INTERNALS):
                self.__items.append(item)
                self.__keys |= clf_internals
            else:
                raise ValueError, 'Unknown clf internal(s) %s' % \
                      clf_internals.difference(self._KNOWN_INTERNALS)
        return self

    @property
    def internals(self):
        return self.__keys

    def listing(self):
        return [(x.descr, x._clf_internals) for x in self.__items]

    @property
    def items(self):
        return self.__items

clfs = Warehouse()

# NB:
#  - Nu-classifiers are turned off since for haxby DS default nu
#    is an 'infisible' one
#  - Python's SMLR is turned off for the duration of development
#    since it is slow and results should be the same as of C version
#
clfs += [ SMLR(lm=0.1, implementation="C", descr="SMLR(lm=0.1)"),
          SMLR(lm=1.0, implementation="C", descr="SMLR(lm=1.0)"),
          SMLR(lm=10.0, implementation="C", descr="SMLR(lm=10.0)"),
          SMLR(lm=100.0, implementation="C", descr="SMLR(lm=100.0)"),
          #                         SMLR(implementation="Python", descr="SMLR(Python)")
          ]

clfs += \
     [ MulticlassClassifier(clfs['smlr'][0],
                            descr='Pairs+maxvote multiclass on ' + \
                            clfs['smlr'][0].descr) ]

if externals.exists('libsvm'):
    from mvpa.clfs import libsvm
    clfs += [libsvm.svm.LinearCSVMC(descr="libsvm.LinSVM(C=def)", probability=1),
             libsvm.svm.LinearCSVMC(
                 C=-10.0, descr="libsvm.LinSVM(C=10*def)", probability=1),
             libsvm.svm.LinearCSVMC(
                 C=1.0, descr="libsvm.LinSVM(C=1)", probability=1),
             libsvm.svm.LinearNuSVMC(descr="libsvm.LinNuSVM(nu=def)", probability=1)
             ]
    clfs += [libsvm.svm.RbfCSVMC(descr="libsvm.RbfSVM()"),
             libsvm.svm.SVMBase(kernel_type='poly',
                                svm_type=libsvm.svmc.C_SVC,
                                descr='libsvm.PolySVM()', probability=1),
             #libsvm.svm.SVMBase(kernel_type='sigmoid',
             #                   svm_type=libsvm.svmc.C_SVC,
             #                   descr='libsvm.SigmoidSVM()'),
             libsvm.svm.RbfNuSVMC(descr="libsvm.RbfNuSVM(nu=def)")
             ]

if externals.exists('shogun'):
    from mvpa.clfs import sg
    for impl in sg.svm.known_svm_impl:
        clfs += [
            sg.svm.LinearCSVMC(
                descr="sg.LinSVM(C=def)/%s" % impl, svm_impl=impl),
            sg.svm.LinearCSVMC(
                C=-10.0, descr="sg.LinSVM(C=10*def)/%s" % impl, svm_impl=impl),
            sg.svm.LinearCSVMC(
                C=1.0, descr="sg.LinSVM(C=1)/%s" % impl, svm_impl=impl),
            ]
        clfs += [
            sg.svm.RbfCSVMC(descr="sg.RbfSVM()/%s" % impl, svm_impl=impl),
#            sg.svm.RbfCSVMC(descr="sg.RbfSVM(gamma=0.1)/%s" % impl, svm_impl=impl, gamma=0.1),
#           sg.svm.SVM_SG_Modular(descr="sg.SigmoidSVM()/%s" % impl, svm_impl=impl, kernel_type="sigmoid"),
            ]



if len(clfs['svm', 'linear']) > 0:
    # if any SVM implementation is known, import default ones
    from mvpa.clfs.svm import *

# lars from R via RPy
if externals.exists('lars'):
    import mvpa.clfs.lars as lars
    from mvpa.clfs.lars import LARS
    for model in lars.known_models:
        # XXX create proper repository of classifiers!
        lars = LARS(descr="LARS(%s)" % model, model_type=model)
        clfs += lars
        # clfs += MulticlassClassifier(lars, descr='Multiclass %s' % lars.descr)

# kNN
clfs += kNN(k=5, descr="kNN(k=5)")

# GPR
clfs += GPR(descr="GPR()")

# "Interesting" classifiers
clfs += \
     FeatureSelectionClassifier(
         LinearCSVMC(),
         SensitivityBasedFeatureSelection(
            SMLRWeights(SMLR(lm=1.0, implementation="C")),
            RangeElementSelector(mode='select')),
         descr="LinSVM on SMLR(lm=1) non-0")


# "Interesting" classifiers
clfs += \
    FeatureSelectionClassifier(
        LinearCSVMC(),
        SensitivityBasedFeatureSelection(
            SMLRWeights(SMLR(lm=1.0, implementation="C")),
            RangeElementSelector(mode='select')),
        descr="LinSVM on SMLR(lm=1) non-0")


# "Interesting" classifiers
clfs += \
    FeatureSelectionClassifier(
        RbfCSVMC(),
        SensitivityBasedFeatureSelection(
           SMLRWeights(SMLR(lm=1.0, implementation="C")),
           RangeElementSelector(mode='select')),
        descr="RbfSVM on SMLR(lm=1) non-0")

clfs += \
    FeatureSelectionClassifier(
        kNN(),
        SensitivityBasedFeatureSelection(
           SMLRWeights(SMLR(lm=1.0, implementation="C")),
           RangeElementSelector(mode='select')),
        descr="kNN on SMLR(lm=1) non-0")

clfs += \
    FeatureSelectionClassifier(
        kNN(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FractionTailSelector(0.05, mode='select', tail='upper')),
        descr="kNN on 5%(ANOVA)")

clfs += \
    FeatureSelectionClassifier(
        kNN(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FixedNElementTailSelector(50, mode='select', tail='upper')),
        descr="kNN on 50(ANOVA)")

clfs += \
    FeatureSelectionClassifier(
        LinearCSVMC(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FractionTailSelector(0.05, mode='select', tail='upper')),
        descr="LinSVM on 5%(ANOVA)")

clfs += \
    FeatureSelectionClassifier(
        LinearCSVMC(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FixedNElementTailSelector(50, mode='select', tail='upper')),
        descr="LinSVM on 50(ANOVA)")

sample_linear_svm = clfs['linear', 'svm'][0]

clfs += \
    FeatureSelectionClassifier(
        sample_linear_svm,
        SensitivityBasedFeatureSelection(
           sample_linear_svm.getSensitivityAnalyzer(transformer=Absolute),
           FractionTailSelector(0.05, mode='select', tail='upper')),
        descr="LinSVM on 5%(SVM)")

clfs += \
    FeatureSelectionClassifier(
        sample_linear_svm,
        SensitivityBasedFeatureSelection(
           sample_linear_svm.getSensitivityAnalyzer(transformer=Absolute),
           FixedNElementTailSelector(50, mode='select', tail='upper')),
        descr="LinSVM on 50(SVM)")


# SVM with unbiased RFE -- transfer-error to another splits, or in
# other terms leave-1-out error on the same dataset
# Has to be bound outside of the RFE definition since both analyzer and
# error should use the same instance.
rfesvm_split = SplitClassifier(LinearCSVMC())#clfs['LinearSVMC'][0])

# "Almost" classical RFE. If this works it would differ only that
# our transfer_error is based on internal splitting and classifier used
# within RFE is a split classifier and its sensitivities per split will get
# averaged
#

#clfs += \
#  FeatureSelectionClassifier(
#    clf = LinearCSVMC(), #clfs['LinearSVMC'][0],         # we train LinearSVM
#    feature_selection = RFE(             # on features selected via RFE
#        # based on sensitivity of a clf which does splitting internally
#        sensitivity_analyzer=rfesvm_split.getSensitivityAnalyzer(),
#        transfer_error=ConfusionBasedError(
#           rfesvm_split,
#           confusion_state="training_confusions"),
#           # and whose internal error we use
#        feature_selector=FractionTailSelector(
#                           0.2, mode='discard', tail='lower'),
#                           # remove 20% of features at each step
#        update_sensitivity=True),
#        # update sensitivity at each step
#    descr='LinSVM+RFE(splits_avg)' )
#
#clfs += \
#  FeatureSelectionClassifier(
#    clf = LinearCSVMC(), #clfs['LinearSVMC'][0],         # we train LinearSVM
#    feature_selection = RFE(             # on features selected via RFE
#        # based on sensitivity of a clf which does splitting internally
#        sensitivity_analyzer=rfesvm_split.getSensitivityAnalyzer(),
#        transfer_error=ConfusionBasedError(
#           rfesvm_split,
#           confusion_state="training_confusions"),
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
#clfs += \
#  SplitClassifier(                      # which does splitting internally
#   FeatureSelectionClassifier(
#    clf = LinearCSVMC(),
#    feature_selection = RFE(             # on features selected via RFE
#        sensitivity_analyzer=\
#            rfesvm.getSensitivityAnalyzer(transformer=Absolute),
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
#clfs += \
#  SplitClassifier(                      # which does splitting internally
#   FeatureSelectionClassifier(
#    clf = LinearCSVMC(),
#    feature_selection = RFE(             # on features selected via RFE
#        sensitivity_analyzer=\
#            rfesvm.getSensitivityAnalyzer(transformer=Absolute),
#        transfer_error=TransferError(rfesvm),
#        stopping_criterion=FixedErrorThresholdStopCrit(0.05),
#        feature_selector=FractionTailSelector(
#                           0.2, mode='discard', tail='lower'),
#                           # remove 20% of features at each step
#        update_sensitivity=True)),
#        # update sensitivity at each step
#   splitter = OddEvenSplitter(),
#   descr='LinSVM+RFE(OddEven)')
