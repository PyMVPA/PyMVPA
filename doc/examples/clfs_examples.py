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

# Define sets of classifiers
from mvpa.clfs.classifier import *
from mvpa.clfs.svm import *
from mvpa.clfs.smlr import SMLR
from mvpa.clfs.ridge import *
from mvpa.clfs.knn import *

# Helpers
from mvpa.clfs.transerror import ConfusionMatrix
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
                 SMLR(implementation="Python", descr="SMLR(Python)")]
      }

clfs['LinReg'] = clfs['SMLR'] + [ RidgeReg(descr="RidgeReg(default)") ]
clfs['LinearC'] = clfs['LinearSVMC'] + clfs['LinReg']
clfs['NonLinearC'] = clfs['NonLinearSVMC'] + [ kNN(descr="kNN(default)") ]
clfs['all'] = clfs['LinearC'] + clfs['NonLinearC']
clfs['clfs_with_sens'] =  clfs['LinearSVMC'] + clfs['SMLR']

# Fix seed or set to None for new each time
N.random.seed(44)

for (dataset, datasetdescr), clfs in \
    [
    ( ( normalFeatureDataset(perlabel=10, nlabels=2,
                             nfeatures=1000,
                             nchunks=5, nonbogus_features=[1, 2],
                             snr=5.0), "Dummy 2-class univariate with 2 useful features"), clfs['all'] ),
    ( ( pureMultivariateSignal(20, 3), "Dummy XOR-pattern"), clfs['all'] )
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
            t0 = time()
            clf.train(training_ds)
            t1 = time()
            predictions = clf.predict(validation_ds.samples)
            t2 = time()
            confusion.add(validation_ds.labels, predictions)
            times.append([t1-t0, t2-t1])

        times = N.mean(times, axis=0)
        print "  %-30s: correct=%.1f%% train:%.1fsec predict:%.1fsec" % \
              (clf.descr, confusion.percentCorrect, times[0], times[1])

