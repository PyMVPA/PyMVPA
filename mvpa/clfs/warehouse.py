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

# Data
from mvpa.datasets.splitter import OddEvenSplitter

# Define sets of classifiers
from mvpa.clfs.base import FeatureSelectionClassifier, SplitClassifier, \
                                 MulticlassClassifier
from mvpa.clfs.smlr import SMLR
from mvpa.clfs.knn import kNN

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


# NB:
#  - Nu-classifiers are turned off since for haxby DS default nu
#    is an 'infisible' one
#  - Python's SMLR is turned off for the duration of development
#    since it is slow and results should be the same as of C version
#
clfs = {
      'SMLR' : [ SMLR(lm=0.1, implementation="C", descr="SMLR(lm=0.1)"),
                 SMLR(lm=1.0, implementation="C", descr="SMLR(lm=1.0)"),
                 SMLR(lm=10.0, implementation="C", descr="SMLR(lm=10.0)"),
                 SMLR(lm=100.0, implementation="C", descr="SMLR(lm=100.0)"),
#                         SMLR(implementation="Python", descr="SMLR(Python)")
                 ]
      }

clfs['LinearSVMC'] = []
clfs['NonLinearSVMC'] = []

if externals.exists('libsvm'):
    from mvpa.clfs import libsvm
    clfs['LinearSVMC'] += [libsvm.svm.LinearCSVMC(descr="libsvm.LinSVM(C=def)"),
                           libsvm.svm.LinearCSVMC(
                                C=-10.0, descr="libsvm.LinSVM(C=10*def)"),
                           libsvm.svm.LinearCSVMC(
                                C=1.0, descr="libsvm.LinSVM(C=1)"),
                           # libsvm.svm.LinearNuSVMC(descr="Linear nu-SVM (default)")
                           ]
    clfs['NonLinearSVMC'] += [libsvm.svm.RbfCSVMC(descr="libsvm.RbfSVM()"),
                              # libsvm.svm.RbfNuSVMC(descr="Rbf nu-SVM (default)")
                              ]

if externals.exists('shogun'):
    from mvpa.clfs import sg
    for impl in sg.svm.known_svm_impl:
        clfs['LinearSVMC'] += [
            sg.svm.LinearCSVMC(
                descr="sg.LinSVM(C=def)/%s" % impl, svm_impl=impl),
            sg.svm.LinearCSVMC(
                C=-10.0, descr="sg.LinSVM(C=10*def)/%s" % impl, svm_impl=impl),
            sg.svm.LinearCSVMC(
                C=1.0, descr="sg.LinSVM(C=1)/%s" % impl, svm_impl=impl),
            ]
        clfs['NonLinearSVMC'] += [
            sg.svm.RbfCSVMC(descr="sg.RbfSVM()/%s" % impl, svm_impl=impl),
            ]

if len(clfs['LinearSVMC']) > 0:
    # if any SVM implementation is known, import default ones
    from mvpa.clfs.svm import *

# lars from R via RPy
if externals.exists('lars'):
    import mvpa.clfs.lars as lars
    from mvpa.clfs.lars import LARS
    clfs['LARS'] = []
    for model in lars.known_models:
        clfs['LARS'] += [LARS(descr="LARS(%s)" % model, model_type=model)]

clfs['LinReg'] = clfs['SMLR'] #+ [ RidgeReg(descr="RidgeReg(default)") ]
clfs['LinearC'] = clfs['LinearSVMC'] + clfs['LinReg'] + clfs['LARS']
clfs['NonLinearC'] = clfs['NonLinearSVMC'] + [ kNN(descr="kNN()") ]
clfs['clfs_with_sens'] =  clfs['LinearSVMC'] + clfs['SMLR'] #+ clfs['LARS']


# "Interesting" classifiers
clfs['SMLR(lm=10)->LinearSVM']  = [
    FeatureSelectionClassifier(
        LinearCSVMC(),
        SensitivityBasedFeatureSelection(
           SMLRWeights(SMLR(lm=10.0, implementation="C")),
           RangeElementSelector(mode='select')),
        descr="LinSVM on SMLR(lm=10) non-0")
    ]

# "Interesting" classifiers
clfs['SMLR(lm=1)->LinearSVM']  = [
    FeatureSelectionClassifier(
        LinearCSVMC(),
        SensitivityBasedFeatureSelection(
           SMLRWeights(SMLR(lm=1.0, implementation="C")),
           RangeElementSelector(mode='select')),
        descr="LinSVM on SMLR(lm=1) non-0")
    ]

# "Interesting" classifiers
clfs['SMLR(lm=10)->RbfSVM']  = [
    FeatureSelectionClassifier(
        RbfCSVMC(),
        SensitivityBasedFeatureSelection(
           SMLRWeights(SMLR(lm=10.0, implementation="C")),
           RangeElementSelector(mode='select')),
        descr="RbfSVM on SMLR(lm=10) non-0")
    ]

clfs['SMLR(lm=10)->kNN']  = [
    FeatureSelectionClassifier(
        kNN(),
        SensitivityBasedFeatureSelection(
           SMLRWeights(SMLR(lm=10.0, implementation="C")),
           RangeElementSelector(mode='select')),
        descr="kNN on SMLR(lm=10) non-0")
    ]


clfs['Anova5%->kNN']  = [
    FeatureSelectionClassifier(
        kNN(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FractionTailSelector(0.05, mode='select', tail='upper')),
        descr="kNN on 5%(ANOVA)")
    ]


clfs['Anova50->kNN']  = [
    FeatureSelectionClassifier(
        kNN(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FixedNElementTailSelector(50, mode='select', tail='upper')),
        descr="kNN on 50(ANOVA)")
    ]


clfs['Anova5%->LinearSVM']  = [
    FeatureSelectionClassifier(
        LinearCSVMC(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FractionTailSelector(0.05, mode='select', tail='upper')),
        descr="LinSVM on 5%(ANOVA)")
    ]


clfs['Anova50->LinearSVM']  = [
    FeatureSelectionClassifier(
        LinearCSVMC(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FixedNElementTailSelector(50, mode='select', tail='upper')),
        descr="LinSVM on 50(ANOVA)")
    ]


clfs['LinearSVM5%->LinearSVM']  = [
    FeatureSelectionClassifier(
        clfs['LinearSVMC'][0],
        SensitivityBasedFeatureSelection(
           clfs['LinearSVMC'][0].getSensitivityAnalyzer(transformer=Absolute),
           FractionTailSelector(0.05, mode='select', tail='upper')),
        descr="LinSVM on 5%(SVM)")
    ]

clfs['LinearSVM50->LinearSVM']  = [
    FeatureSelectionClassifier(
        clfs['LinearSVMC'][0],
        SensitivityBasedFeatureSelection(
           clfs['LinearSVMC'][0].getSensitivityAnalyzer(transformer=Absolute),
           FixedNElementTailSelector(50, mode='select', tail='upper')),
        descr="LinSVM on 50(SVM)")
    ]


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

# TODO: wrap head around on how to implement classical RFE (unbiased,
#  ie with independent generalization) within out framework without
#  much of changing
clfs['SVM+RFE/splits_avg'] = [
  FeatureSelectionClassifier(
    clf = LinearCSVMC(), #clfs['LinearSVMC'][0],         # we train LinearSVM
    feature_selection = RFE(             # on features selected via RFE
        # based on sensitivity of a clf which does splitting internally
        sensitivity_analyzer=rfesvm_split.getSensitivityAnalyzer(),
        transfer_error=ConfusionBasedError(
           rfesvm_split,
           confusion_state="training_confusions"),
           # and whose internal error we use
        feature_selector=FractionTailSelector(
                           0.2, mode='discard', tail='lower'),
                           # remove 20% of features at each step
        update_sensitivity=True),
        # update sensitivity at each step
    descr='LinSVM+RFE(splits_avg)' )
  ]

clfs['SVM+RFE/splits_avg(static)'] = [
  FeatureSelectionClassifier(
    clf = LinearCSVMC(), #clfs['LinearSVMC'][0],         # we train LinearSVM
    feature_selection = RFE(             # on features selected via RFE
        # based on sensitivity of a clf which does splitting internally
        sensitivity_analyzer=rfesvm_split.getSensitivityAnalyzer(),
        transfer_error=ConfusionBasedError(
           rfesvm_split,
           confusion_state="training_confusions"),
           # and whose internal error we use
        feature_selector=FractionTailSelector(
                           0.2, mode='discard', tail='lower'),
                           # remove 20% of features at each step
        update_sensitivity=False),
        # update sensitivity at each step
    descr='LinSVM+RFE(splits_avg,static)' )
  ]


rfesvm = LinearCSVMC()

# This classifier will do RFE while taking transfer error to testing
# set of that split. Resultant classifier is voted classifier on top
# of all splits, let see what that would do ;-)
clfs['SVM+RFE'] = [
  SplitClassifier(                      # which does splitting internally
   FeatureSelectionClassifier(
    clf = LinearCSVMC(),
    feature_selection = RFE(             # on features selected via RFE
        sensitivity_analyzer=\
            rfesvm.getSensitivityAnalyzer(transformer=Absolute),
        transfer_error=TransferError(rfesvm),
        stopping_criterion=FixedErrorThresholdStopCrit(0.05),
        feature_selector=FractionTailSelector(
                           0.2, mode='discard', tail='lower'),
                           # remove 20% of features at each step
        update_sensitivity=True)),
        # update sensitivity at each step
    descr='LinSVM+RFE(N-Fold)')
  ]

clfs['SVM+RFE/oe'] = [
  SplitClassifier(                      # which does splitting internally
   FeatureSelectionClassifier(
    clf = LinearCSVMC(),
    feature_selection = RFE(             # on features selected via RFE
        sensitivity_analyzer=\
            rfesvm.getSensitivityAnalyzer(transformer=Absolute),
        transfer_error=TransferError(rfesvm),
        stopping_criterion=FixedErrorThresholdStopCrit(0.05),
        feature_selector=FractionTailSelector(
                           0.2, mode='discard', tail='lower'),
                           # remove 20% of features at each step
        update_sensitivity=True)),
        # update sensitivity at each step
   splitter = OddEvenSplitter(),
   descr='LinSVM+RFE(OddEven)')
  ]


# RFE where each pair-wise classifier is trained with RFE, so we can get
# different feature sets for different pairs of categories (labels)
clfs['SVM/Multiclass+RFE/splits_avg'] = \
    [ MulticlassClassifier(clfs['SVM+RFE/splits_avg'][0],
      descr='SVM/Multiclass+RFE/splits_avg') ]

# Run on all here defined classifiers
clfs['all'] = clfs['LinearC'] + clfs['NonLinearC'] + \
              clfs['Anova5%->kNN'] + clfs['Anova50->kNN'] + \
              clfs['SMLR(lm=10)->kNN'] + \
              clfs['LinearSVM5%->LinearSVM'] + clfs['Anova5%->LinearSVM'] + \
              clfs['LinearSVM50->LinearSVM'] + clfs['Anova50->LinearSVM'] + \
              clfs['SMLR(lm=1)->LinearSVM'] + clfs['SMLR(lm=10)->LinearSVM'] + \
              clfs['SMLR(lm=10)->RbfSVM']
#              clfs['SVM+RFE'] + clfs['SVM+RFE/oe'] + \
#              clfs['SVM+RFE/splits_avg(static)'] + \
#              clfs['SVM+RFE/splits_avg']


# since some classifiers make sense only for multiclass
clfs['all_multi'] = clfs['all'] + \
                    [ MulticlassClassifier(
                        clfs['SMLR'][0],
                        descr='Pairs+maxvote multiclass on ' + \
                        clfs['SMLR'][0].descr) ]

# TODO:  This one yet to be fixed: deepcopy might fail if
#        sensitivity analyzer is classifierbased and classifier wasn't
#        untrained, which can happen, thus for now it is disabled
# + clfs['SVM/Multiclass+RFE/splits_avg']
