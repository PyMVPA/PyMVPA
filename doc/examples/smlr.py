#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Compare SMLR to Linear SVM Classifier
=====================================

.. index:: SMLR, SVM

Runs both classifiers on the the same dataset and compare their performance.
This example also shows an example usage of confusion matrices and how two
classifers can be combined.
"""

from mvpa.suite import *

if __debug__:
    debug.active.append('SMLR_')

# features of sample data
print "Generating samples..."
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

# create the pymvpa training dataset from the labeled features
patternsPos = Dataset(samples=samp1[:ntrain,:], labels=1)
patternsNeg = Dataset(samples=samp2[:ntrain,:], labels=0)
trainpat = patternsPos + patternsNeg

# create patters for the testing dataset
patternsPos = Dataset(samples=samp1[ntrain:,:], labels=1)
patternsNeg = Dataset(samples=samp2[ntrain:,:], labels=0)
testpat = patternsPos + patternsNeg

# set up the SMLR classifier
print "Evaluating SMLR classifier..."
smlr = SMLR(fit_all_weights=True)

# enable saving of the values used for the prediction
smlr.states.enable('values')

# train with the known points
smlr.train(trainpat)

# run the predictions on the test values
pre = smlr.predict(testpat.samples)

# calculate the confusion matrix
smlr_confusion = ConfusionMatrix(
    labels=trainpat.uniquelabels, targets=testpat.labels,
    predictions=pre)

# now do the same for a linear SVM
print "Evaluating Linear SVM classifier..."
lsvm = LinearNuSVMC(probability=1)

# enable saving of the values used for the prediction
lsvm.states.enable('values')

# train with the known points
lsvm.train(trainpat)

# run the predictions on the test values
pre = lsvm.predict(testpat.samples)

# calculate the confusion matrix
lsvm_confusion = ConfusionMatrix(
    labels=trainpat.uniquelabels, targets=testpat.labels,
    predictions=pre)

# now train SVM with selected features
print "Evaluating Linear SVM classifier with SMLR's features..."

keepInd = (N.abs(smlr.weights).mean(axis=1)!=0)
newtrainpat = trainpat.selectFeatures(keepInd, sort=False)
newtestpat = testpat.selectFeatures(keepInd, sort=False)

# train with the known points
lsvm.train(newtrainpat)

# run the predictions on the test values
pre = lsvm.predict(newtestpat.samples)

# calculate the confusion matrix
lsvm_confusion_sparse = ConfusionMatrix(
    labels=newtrainpat.uniquelabels, targets=newtestpat.labels,
    predictions=pre)


print "SMLR Percent Correct:\t%g%% (Retained %d/%d features)" % \
    (smlr_confusion.percentCorrect,
     (smlr.weights!=0).sum(), N.prod(smlr.weights.shape))
print "linear-SVM Percent Correct:\t%g%%" % \
    (lsvm_confusion.percentCorrect)
print "linear-SVM Percent Correct (with %d features from SMLR):\t%g%%" % \
    (keepInd.sum(), lsvm_confusion_sparse.percentCorrect)
