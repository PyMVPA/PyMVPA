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
# DONE:
from mvpa.clfs.warehouse import clfs


#clfs['all'] = clfs['SVM+RFE']
#clfs['all'] = clfs['SVM/Multiclass+RFE']

if __name__ == "__main__":

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
                                  nfeatures=100,
                                  nchunks=6, nonbogus_features=[11, 10],
                                  snr=3.0)


    for (dataset, datasetdescr), clfs_ in \
        [
        ((dummy2, "Dummy 2-class univariate with 2 useful features out of 400"), clfs['all']),
        ((pureMultivariateSignal(8, 3), "Dummy XOR-pattern"), clfs['all_multi']),
        ((haxby8_no0, "Haxby 8-cat subject 1"), clfs['all_multi']),
        ]:

        print "%s: %s" % (datasetdescr, `dataset`)
        print " Classifier                                  %corr  #features\t train predict  full"
        for clf in clfs_:
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
            nf = []
            t0 = time()
            clf.states.enable('feature_ids')
            for nfold, (training_ds, validation_ds) in \
                    enumerate(NFoldSplitter()(dataset)):
                clf.train(training_ds)
                nf.append(len(clf.feature_ids))
                if nf[-1] == 0:
                    break
                predictions = clf.predict(validation_ds.samples)
                confusion.add(validation_ds.labels, predictions)
                times.append([clf.training_time, clf.predicting_time])
            print "  %-40s: "  % clf.descr,
            if nf[-1] == 0:
                print "no features were selected. skipped"
                continue
            tfull = time() - t0
            times = N.mean(times, axis=0)
            nf = N.mean(nf)
            print "%5.1f%%   %-4d\t %.2fs  %.2fs   %.2fs" % \
                  (confusion.percentCorrect, nf, times[0], times[1], tfull)


