#!/usr/bin/env python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Example demonstrating a SMLR classifier"""

import numpy as N

from mvpa.datasets.dataset import Dataset
from mvpa.clfs.smlr import SMLR
from mvpa.clfs.transerror import ConfusionMatrix

# features of sample data
nfeat = 10000
nsamp = 100
ntrain = 90
goodfeat = 10
offset = .5

# create the sample datasets
samp1 = N.random.randn(nsamp,nfeat)
samp1[:,:goodfeat] += offset

samp2 = N.random.randn(nsamp,nfeat)
samp2[:,:goodfeat] -= offset

traindat = N.vstack((samp1[:ntrain,:],samp2[:ntrain,:]))
testdat = N.vstack((samp1[ntrain:,:],samp2[ntrain:,:]))

# create the pymvpa training dataset from the labeled features
patternsPos = Dataset(samples=samp1[:ntrain,:], labels=1)
patternsNeg = Dataset(samples=samp2[:ntrain,:], labels=0)
trainpat = patternsPos + patternsNeg

# create patters for the testing dataset
patternsPos = Dataset(samples=samp1[ntrain:,:], labels=1)
patternsNeg = Dataset(samples=samp2[ntrain:,:], labels=0)
testpat = patternsPos + patternsNeg

# set up the SMLR classifier
clf = SMLR(lm=1.5)

# enable saving of the values used for the prediction
clf.states.enable('values')

# train with the known points
clf.train(trainpat)

# run the predictions on the test values
pre = clf.predict(testpat.samples)

# calculate the confusion matrix
testing_confusion = ConfusionMatrix(
    labels=trainpat.uniquelabels, targets=testpat.labels,
    predictions=pre)

print "Percent Correct: %g%%" % (testing_confusion.percentCorrect)
