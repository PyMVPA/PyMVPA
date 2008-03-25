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
clfs={'LinearSVMC' : [LinearCSVMC(descr="Linear C-SVM (default)"),
                      LinearCSVMC(C=1.0, descr="Linear C-SVM (C=1)"),
#                      LinearNuSVMC(descr="Linear nu-SVM (default)")
                      ],
      'NonLinearSVMC' : [RbfCSVMC(descr="Rbf C-SVM (default)"),
#                         RbfNuSVMC(descr="Rbf nu-SVM (default)")
                         ],
      'SMLR' : [ # SMLR(implementation="C", descr="SMLR(default)"),
                 SMLR(lm=1.0, implementation="C", descr="SMLR(lm=1.0)"),
                 SMLR(lm=10.0, implementation="C", descr="SMLR(lm=10.0)"),
#                         SMLR(implementation="Python", descr="SMLR(Python)")
                 ]
      }

clfs['LinReg'] = clfs['SMLR'] + [ RidgeReg(descr="RidgeReg(default)") ]
clfs['LinearC'] = clfs['LinearSVMC'] + clfs['LinReg']
clfs['NonLinearC'] = clfs['NonLinearSVMC'] + [ kNN(descr="kNN(default)") ]
clfs['clfs_with_sens'] =  clfs['LinearSVMC'] + clfs['SMLR']

# "Interesting" classifiers
clfs['SMLR->SVM']  = [
    FeatureSelectionClassifier(
        clfs['LinearSVMC'][0],
        SensitivityBasedFeatureSelection(
           SMLRWeights(clfs['SMLR'][0]),
           RangeElementSelector()),
        descr="SVM on SMLR(lm=1.0) non-0 features")
    ]

clfs['Anova25%->SVM']  = [
    FeatureSelectionClassifier(
        clfs['LinearSVMC'][0],
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FractionTailSelector(0.25, mode='select')),
        descr="SVM on 25% best(ANOVA) features")
    ]

clfs['SVM25%->SVM']  = [
    FeatureSelectionClassifier(
        clfs['LinearSVMC'][0],
        SensitivityBasedFeatureSelection(
           LinearSVMWeights(clfs['LinearSVMC'][0],
                            transformer=Absolute),
           FractionTailSelector(0.25, mode='select')),
        descr="SVM on 25% best(SVM) features")
    ]


# SVM with unbiased RFE -- transfer-error to another splits, or in
# other terms leave-1-out error on the same dataset
# Has to be bound outside of the RFE definition since both analyzer and
# error should use the same instance.
rfesvm = SplitClassifier(LinearCSVMC())#clfs['LinearSVMC'][0])

# "Almost" classical RFE. If this works it would differ only that
# our transfer_error is based on internal splitting and classifier used within RFE
# is a split classifier and its sensitivities per split will get averaged
#
#
# TODO: wrap head around on how to implement classical RFE (unbiased,
#  ie with independent generalization) within out framework without
#  much of changing
clfs['SVM+RFE'] = [
  FeatureSelectionClassifier(
    clf = LinearCSVMC(), #clfs['LinearSVMC'][0],         # we train LinearSVM
    feature_selection = RFE(             # on features selected via RFE
        sensitivity_analyzer=selectAnalyzer( # based on sensitivity of a clf
           clf=SplitClassifier(clf=rfesvm)), # which does splitting internally
        transfer_error=ConfusionBasedError(
           rfesvm,
           confusion_state="training_confusions"), # and whose internall error we use
        feature_selector=FractionTailSelector(0.2),   # remove 20% of features at each step
        update_sensitivity=True),                     # update sensitivity at each step
    descr='SVM+RFE/splits' )
  ]


# RFE where each pair-wise classifier is trained with RFE, so we can get
# different feature sets for different pairs of categories (labels)
clfs['SVM/Multiclass+RFE'] = [ MulticlassClassifier(clfs['SVM+RFE'][0],
                                                    descr='SVM/Multiclass+RFE') ]

# Run on all here defined classifiers
clfs['all'] = clfs['LinearC'] + clfs['NonLinearC'] + \
              clfs['SVM25%->SVM'] + clfs['Anova25%->SVM'] + clfs['SMLR->SVM'] + \
              clfs['SVM+RFE']

# since some classifiers make sense only for multiclass
clfs['all_multi'] = clfs['all'] + clfs['SVM/Multiclass+RFE']
