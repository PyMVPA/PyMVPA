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
Nested Cross-Validation
=======================

.. index:: model selection, cross-validation

Often it is desired to explore multiple models (classifiers,
parameterizations) but it becomes an easy trap for introducing an
optimistic bias into generalization estimate.  The easiest but
computationally intensive solution to overcome such a bias is to carry
model selection by estimating the same (or different) performance
characteristic while operating only on training data.  If such
performance is a cross-validation, then it leads to the so called
"nested cross-validation" procedure.

This example will demonstrate on how to implement such nested
cross-validation while selecting the best performing classifier from
the warehouse of available within PyMVPA.
"""

from mvpa2.suite import *
# increase verbosity a bit for now
verbose.level = 3
# pre-seed RNG if you want to investigate the effects, thus
# needing reproducible results
#mvpa2.seed(3)

"""
For this simple example lets generate some fresh random data with 2
relevant features and low SNR.
"""

dataset = normal_feature_dataset(perlabel=24, nlabels=2, nchunks=3,
                                 nonbogus_features=[0, 1],
                                 nfeatures=100, snr=3.0)

"""
For the demonstration of model selection benefit, lets first compute
cross-validated error using simple and popular kNN.
"""

clf_sample = kNN()
cv_sample = CrossValidation(clf_sample, NFoldPartitioner())

verbose(1, "Estimating error using a sample classifier")
error_sample = np.mean(cv_sample(dataset))

"""
For the convenience lets define a helpful function which we will use
twice -- once within cross-validation, and once on the whole dataset
"""

def select_best_clf(dataset_, clfs):
    """Select best model according to CVTE

    Helper function which we will use twice -- once for proper nested
    cross-validation, and once to see how big an optimistic bias due
    to model selection could be if we simply provide an entire dataset.

    Parameters
    ----------
    dataset_ : Dataset
    clfs : list of Classifiers
      Which classifiers to explore

    Returns
    -------
    best_clf, best_error
    """
    best_error = None
    for clf in clfs:
        cv = CrossValidation(clf, NFoldPartitioner())
        # unfortunately we don't have ability to reassign clf atm
        # cv.transerror.clf = clf
        try:
            error = np.mean(cv(dataset_))
        except LearnerError, e:
            # skip the classifier if data was not appropriate and it
            # failed to learn/predict at all
            continue
        if best_error is None or error < best_error:
            best_clf = clf
            best_error = error
        verbose(4, "Classifier %s cv error=%.2f" % (clf.descr, error))
    verbose(3, "Selected the best out of %i classifiers %s with error %.2f"
            % (len(clfs), best_clf.descr, best_error))
    return best_clf, best_error

"""
First lets select a classifier within cross-validation, thus
eliminating model-selection bias
"""

best_clfs = {}
confusion = ConfusionMatrix()
verbose(1, "Estimating error using nested CV for model selection")
partitioner = NFoldPartitioner()
splitter = Splitter('partitions')
for isplit, partitions in enumerate(partitioner.generate(dataset)):
    verbose(2, "Processing split #%i" % isplit)
    dstrain, dstest = list(splitter.generate(partitions))
    best_clf, best_error = select_best_clf(dstrain, clfswh['!gnpp'])
    best_clfs[best_clf.descr] = best_clfs.get(best_clf.descr, 0) + 1
    # now that we have the best classifier, lets assess its transfer
    # to the testing dataset while training on entire training
    tm = TransferMeasure(best_clf, splitter,
                         postproc=BinaryFxNode(mean_mismatch_error,
                                               space='targets'),
                         enable_ca=['stats'])
    tm(partitions)
    confusion += tm.ca.stats

"""
And for comparison, lets assess what would be the best performance if
we simply explore all available classifiers, providing all the data at
once
"""


verbose(1, "Estimating error via fishing expedition (best clf on entire dataset)")
cheating_clf, cheating_error = select_best_clf(dataset, clfswh['!gnpp'])

print """Errors:
 sample classifier (kNN): %.2f
 model selection within cross-validation: %.2f
 model selection via fishing expedition: %.2f with %s
 """ % (error_sample, 1 - confusion.stats['ACC'],
        cheating_error, cheating_clf.descr)

print "# of times following classifiers were selected within " \
      "nested cross-validation:"
for c, count in sorted(best_clfs.items(), key=lambda x:x[1], reverse=True):
    print " %i times %s" % (count, c)

print "\nConfusion table for the nested cross-validation results:"
print confusion
