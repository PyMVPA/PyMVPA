#!/usr/bin/env python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Examples demonstrating varioius classifiers on different datasets"""

import os
from time import time
import numpy as N

from mvpa.datasets.dataset import Dataset
from mvpa.datasets.niftidataset import NiftiDataset
from mvpa.datasets.splitter import *
from mvpa.datasets.misc import zscore

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
from mvpa.misc.data_generators import *
from mvpa.misc.iohelpers import SampleAttributes
from mvpa.misc.signal import detrend

# Misc tools
#
# no MVPA warnings during whole testsuite
from mvpa.misc import warning
warning.handlers = []


# Define groups of classifiers.
#
# TODO: Should be moved somewhere in mvpa -- all those duplicate
#       list of classifiers within tests/tests_warehouse_clfs
#
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
        descr="SVM on SMLR(lm=10) non-0 features")
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

#clfs['all'] = clfs['SVM+RFE']
#clfs['all'] = clfs['SVM/Multiclass+RFE']

# fix seed or set to None for new each time
N.random.seed(44)


# Load Haxby dataset example
haxby1path = '../../data'
attrs = SampleAttributes(os.path.join(haxby1path, 'attributes.txt'))
haxby8 = NiftiDataset(samples=os.path.join(haxby1path, 'bold.nii.gz'),
                      labels=attrs.labels,
                      chunks=attrs.chunks,
                      mask=os.path.join(haxby1path, 'mask.nii.gz'),
                      dtype=N.float32)

# preprocess slightly
detrend(haxby8, perchunk=True, model='linear')
zscore(haxby8, perchunk=True, baselinelabels=[0], targetdtype='float32')
haxby8_no0 = haxby8.selectSamples(haxby8.labels != 0)

dummy2 = normalFeatureDataset(perlabel=30, nlabels=2,
                              nfeatures=400,
                              nchunks=6, nonbogus_features=[1, 2],
                              snr=5.0)


for (dataset, datasetdescr), clfs in \
    [
    ((dummy2, "Dummy 2-class univariate with 2 useful features"), clfs['all']),
    ((pureMultivariateSignal(8, 3), "Dummy XOR-pattern"), clfs['all_multi']),
    ((haxby8_no0, "Haxby 8-cat subject 1"), clfs['all_multi']),
    ]:

    print "%s: %s" % (datasetdescr, `dataset`)
    for clf in clfs:
        # Lets do splits/train/predict explicitely so we could track timing
        # otherwise could be just
        #cv = CrossValidatedTransferError(
        #         TransferError(clf),
        #         NFoldSplitter(),
        #         enable_states=['confusion'])
        #error = cv(dataset)
        #print cv.confusion

        # to report transfer error
        confusion = ConfusionMatrix()
        times = []
        for nfold, (training_ds, validation_ds) in \
                enumerate(NFoldSplitter()(dataset)):
            clf.train(training_ds)
            predictions = clf.predict(validation_ds.samples)
            confusion.add(validation_ds.labels, predictions)
            times.append([clf.training_time, clf.predicting_time])

        times = N.mean(times, axis=0)
        print "  %-30s: correct=%.1f%% train:%.2fsec predict:%.2fsec" % \
              (clf.descr, confusion.percentCorrect, times[0], times[1])

