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

from time import time
import numpy as N

from mvpa.datasets.dataset import Dataset
from mvpa.datasets.splitter import *

# Define sets of classifiers
from mvpa.clfs.classifier import *
from mvpa.clfs.svm import *
from mvpa.clfs.smlr import SMLR
from mvpa.clfs.ridge import *
from mvpa.clfs.knn import *

# Algorithms
from mvpa.algorithms.datameasure import *
from mvpa.algorithms.rfe import *
from mvpa.algorithms.linsvmweights import *

# Helpers
from mvpa.clfs.transerror import *
from mvpa.misc.data_generators import *

# Misc tools
#
# no MVPA warnings during whole testsuite
from mvpa.misc import warning
warning.handlers = []


# Define groups of classifiers. Should be moved somewhere in mvpa
clfs={'LinearSVMC' : [LinearCSVMC(descr="Linear C-SVM (default)"),
                      LinearNuSVMC(descr="Linear nu-SVM (default)")],
      'NonLinearSVMC' : [RbfCSVMC(descr="Rbf C-SVM (default)"),
                         RbfNuSVMC(descr="Rbf nu-SVM (default)")],
      'SMLR' : [ SMLR(implementation="C", descr="SMLR(default)"),
                 # SMLR(implementation="Python", descr="SMLR(Python)")
                 ]
      }

clfs['LinReg'] = clfs['SMLR'] + [ RidgeReg(descr="RidgeReg(default)") ]
clfs['LinearC'] = clfs['LinearSVMC'] + clfs['LinReg']
clfs['NonLinearC'] = clfs['NonLinearSVMC'] + [ kNN(descr="kNN(default)") ]
clfs['clfs_with_sens'] =  clfs['LinearSVMC'] + clfs['SMLR']

# "Interesting" classifiers

clfs['SMLR->SVM']  = [
    FeatureSelectionClassifier(
        clf=clfs['LinearSVMC'][0],
        SensitivityBasedFeatureSelection(
           SMLRWeights(clfs['SMLR'[0])),
           NonZero()))#TODO


# TODO: Fix a bug which wouldn't require me to explicitely untrain here
clfs['LinearSVMC'][0].untrain()

# SVM with unbiased RFE -- transfer-error to another splits, or in
# other terms leave-1-out error on the same dataset
# Has to be bound outside of the RFE definition since both analyzer and
# error should use the same instance.
rfesvm = SplitClassifier(clfs['LinearSVMC'][0])


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
    clf = clfs['LinearSVMC'][0],         # we train LinearSVM
    feature_selection = RFE(             # on features selected via RFE
        sensitivity_analyzer=selectAnalyzer( # based on sensitivity of a clf
           clf=SplitClassifier(clf=rfesvm)), # which does splitting internally
        transfer_error=ConfusionBasedError(
           rfesvm,
           confusion_state="training_confusions"), # and whose internall error we use
        update_sensitivity=True),                     # and we update sensitivity at each step
    descr='SVM+RFE/splits' )
  ]

# Run on all here defined classifiers
clfs['all'] = clfs['LinearC'] + clfs['NonLinearC'] + clfs['SVM+RFE']

# fix seed or set to None for new each time
N.random.seed(44)

for (dataset, datasetdescr), clfs in \
    [
    ( ( normalFeatureDataset(perlabel=10, nlabels=2,
                             nfeatures=1000,
                             nchunks=5, nonbogus_features=[1, 2],
                             snr=5.0), "Dummy 2-class univariate with 2 useful features"), clfs['all'] ),
    ( ( pureMultivariateSignal(4, 3), "Dummy XOR-pattern"), clfs['all'] )
    ]:

    print "%s: %s" % (datasetdescr, `dataset`)
    for clf in clfs:
        # Lets do splits/train/predict explicitely so we could track timing
        # otherwise could be just
        # error = CrossValidatedTransferError(TransferError(clf),
        #                                     NFoldSplitter())(dataset)
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
        print "  %-30s: correct=%.1f%% train:%.1fsec predict:%.1fsec" % \
              (clf.descr, confusion.percentCorrect, times[0], times[1])

