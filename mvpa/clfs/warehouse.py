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
from mvpa.datasets.splitter import *

# Define sets of classifiers
from mvpa.clfs.classifier import *
from mvpa.clfs.svm import *
from mvpa.clfs.smlr import SMLR
from mvpa.clfs.ridge import *
from mvpa.clfs.knn import *

# Algorithms
from mvpa.algorithms.featsel import *
from mvpa.algorithms.datameasure import *
from mvpa.algorithms.anova import *
from mvpa.algorithms.rfe import *
from mvpa.algorithms.linsvmweights import *
from mvpa.algorithms.smlrweights import *
from mvpa.algorithms.cvtranserror import *

# Helpers
from mvpa.clfs.transerror import *


# NB:
#  - Nu-classifiers are turned off since for haxby DS default nu
#    is an 'infisible' one
#  - Python's SMLR is turned off for the duration of development
#    since it is slow and results should be the same as of C version
#
clfs={'LinearSVMC' : [LinearCSVMC(descr="LinSVM(C=def)"),
                      LinearCSVMC(C=-10.0, descr="LinSVM(C=10*def)"),
                      LinearCSVMC(C=1.0, descr="LinSVM(C=1)"),
#                      LinearNuSVMC(descr="Linear nu-SVM (default)")
                      ],
      'NonLinearSVMC' : [RbfCSVMC(descr="RbfSVM()"),
#                         RbfNuSVMC(descr="Rbf nu-SVM (default)")
                         ],
      'SMLR' : [ SMLR(lm=0.1, implementation="C", descr="SMLR(lm=0.1)"),
                 SMLR(lm=1.0, implementation="C", descr="SMLR(lm=1.0)"),
                 SMLR(lm=10.0, implementation="C", descr="SMLR(lm=10.0)"),
#                         SMLR(implementation="Python", descr="SMLR(Python)")
                 ]
      }

clfs['LinReg'] = clfs['SMLR'] #+ [ RidgeReg(descr="RidgeReg(default)") ]
clfs['LinearC'] = clfs['LinearSVMC'] + clfs['LinReg']
clfs['NonLinearC'] = clfs['NonLinearSVMC'] + [ kNN(descr="kNN()") ]
clfs['clfs_with_sens'] =  clfs['LinearSVMC'] + clfs['SMLR']

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
clfs['SMLR->RbfSVM']  = [
    FeatureSelectionClassifier(
        RbfCSVMC(),
        SensitivityBasedFeatureSelection(
           SMLRWeights(SMLR(lm=10.0, implementation="C")),
           RangeElementSelector(mode='select')),
        descr="RbfSVM on SMLR(lm=10) non-0")
    ]

clfs['Anova5%->LinearSVM']  = [
    FeatureSelectionClassifier(
        LinearCSVMC(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FractionTailSelector(0.05, mode='select', tail='upper')),
        descr="LinSVM on 5%(ANOVA)")
    ]

clfs['LinearSVM5%->LinearSVM']  = [
    FeatureSelectionClassifier(
        clfs['LinearSVMC'][0],
        SensitivityBasedFeatureSelection(
           LinearSVMWeights(clfs['LinearSVMC'][0],
                            transformer=Absolute),
           FractionTailSelector(0.05, mode='select', tail='upper')),
        descr="LinSVM on 5%(SVM)")
    ]


# SVM with unbiased RFE -- transfer-error to another splits, or in
# other terms leave-1-out error on the same dataset
# Has to be bound outside of the RFE definition since both analyzer and
# error should use the same instance.
rfesvm_split = SplitClassifier(LinearCSVMC())#clfs['LinearSVMC'][0])

# "Almost" classical RFE. If this works it would differ only that
# our transfer_error is based on internal splitting and classifier used within RFE
# is a split classifier and its sensitivities per split will get averaged
#
#
# TODO: wrap head around on how to implement classical RFE (unbiased,
#  ie with independent generalization) within out framework without
#  much of changing
clfs['SVM+RFE/splits_avg'] = [
  FeatureSelectionClassifier(
    clf = LinearCSVMC(), #clfs['LinearSVMC'][0],         # we train LinearSVM
    feature_selection = RFE(             # on features selected via RFE
        sensitivity_analyzer=selectAnalyzer( # based on sensitivity of a clf
           clf=rfesvm_split), # which does splitting internally
        transfer_error=ConfusionBasedError(
           rfesvm_split,
           confusion_state="training_confusions"), # and whose internal error we use
        feature_selector=FractionTailSelector(
                           0.2, mode='discard', tail='lower'),   # remove 20% of features at each step
        update_sensitivity=True),                     # update sensitivity at each step
    descr='LinSVM+RFE(splits_avg)' )
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
        sensitivity_analyzer=LinearSVMWeights(clf=rfesvm,
                                              transformer=Absolute),
        transfer_error=TransferError(rfesvm),
        stopping_criterion=FixedErrorThresholdStopCrit(0.05),
        feature_selector=FractionTailSelector(
                           0.2, mode='discard', tail='lower'),   # remove 20% of features at each step
        update_sensitivity=True)),                     # update sensitivity at each step
    descr='LinSVM+RFE(N-Fold)')
  ]

clfs['SVM+RFE/oe'] = [
  SplitClassifier(                      # which does splitting internally
   FeatureSelectionClassifier(
    clf = LinearCSVMC(),
    feature_selection = RFE(             # on features selected via RFE
        sensitivity_analyzer=LinearSVMWeights(clf=rfesvm,
                                              transformer=Absolute),
        transfer_error=TransferError(rfesvm),
        feature_selector=FractionTailSelector(
                           0.2, mode='discard', tail='lower'),   # remove 20% of features at each step
        update_sensitivity=True)),                     # update sensitivity at each step
   splitter = OddEvenSplitter(),
   descr='LinSVM+RFE(OddEven)')
  ]


# RFE where each pair-wise classifier is trained with RFE, so we can get
# different feature sets for different pairs of categories (labels)
clfs['SVM/Multiclass+RFE/splits_avg'] = [ MulticlassClassifier(clfs['SVM+RFE/splits_avg'][0],
                                                    descr='SVM/Multiclass+RFE/splits_avg') ]

# Run on all here defined classifiers
clfs['all'] = clfs['LinearC'] + clfs['NonLinearC'] + \
              clfs['LinearSVM5%->LinearSVM'] + clfs['Anova5%->LinearSVM'] + \
              clfs['SMLR(lm=1)->LinearSVM'] + clfs['SMLR(lm=10)->LinearSVM'] + clfs['SMLR->RbfSVM'] + \
              clfs['SVM+RFE'] + clfs['SVM+RFE/oe']
#+ clfs['SVM+RFE/splits'] + \


# since some classifiers make sense only for multiclass
clfs['all_multi'] = clfs['all']
# TODO:  This one yet to be fixed: deepcopy might fail if
#        sensitivity analyzer is classifierbased and classifier wasn't
#        untrained, which can happen, thus for now it is disabled
# + clfs['SVM/Multiclass+RFE/splits_avg']
