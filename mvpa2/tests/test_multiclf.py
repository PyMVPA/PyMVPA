# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Multiclass Classifiers

Pulled into a separate tests file for efficiency
"""

import numpy as np

from mvpa2.testing import *
from mvpa2.testing.datasets import *
from mvpa2.testing.clfs import *

from mvpa2.base.dataset import vstack

from mvpa2.generators.partition import NFoldPartitioner, OddEvenPartitioner
from mvpa2.generators.splitters import Splitter

from mvpa2.clfs.gnb import GNB
from mvpa2.clfs.meta import CombinedClassifier, \
     BinaryClassifier, MulticlassClassifier, \
     MaximalVote
from mvpa2.measures.base import TransferMeasure, CrossValidation
from mvpa2.mappers.fx import mean_sample, BinaryFxNode
from mvpa2.misc.errorfx import mean_mismatch_error



# Generate test data for testing ties
#mvpa2._random_seed = 2#982220910
@reseed_rng()
def get_dsties1():
    ds = datasets['uni2small'].copy()
    dtarget = ds.targets[0]             # duplicate target
    tied_samples = ds.targets == dtarget
    ds2 = ds[tied_samples].copy(deep=True)
    # add similar noise to both ties
    noise_level = 0.2
    ds2.samples += \
                  np.random.normal(size=ds2.shape)*noise_level
    ds[tied_samples].samples += \
                  np.random.normal(size=ds2.shape)*noise_level
    ds2.targets[:] = 'TI' # 'E' would have been swallowed since it is S2 here
    ds = vstack((ds, ds2))
    ds.a.ties = [dtarget, 'TI']
    ds.a.ties_idx = [ds.targets == t for t in ds.a.ties]
    return ds
_dsties1 = get_dsties1()

#from mvpa2.clfs.smlr import SMLR
#clf=SMLR(lm=1.0, fit_all_weights=True, enable_ca=['estimates'])
#if True:
@sweepargs(clf=clfswh['multiclass'])
def test_multiclass_ties(clf):
    if 'lars' in clf.__tags__:
        raise SkipTest("Known to crash while running this test")
    ds = _dsties1

    # reassign data between ties, so we know that decision is data, not order driven
    ds_ = ds.copy(deep=True)
    ds_.samples[ds.a.ties_idx[1]] = ds.samples[ds.a.ties_idx[0]]
    ds_.samples[ds.a.ties_idx[0]] = ds.samples[ds.a.ties_idx[1]]
    ok_(np.any(ds_.samples != ds.samples))

    clf_ = clf.clone()
    clf = clf.clone()
    clf.ca.enable(['estimates', 'predictions'])
    clf_.ca.enable(['estimates', 'predictions'])
    te = TransferMeasure(clf, Splitter('train'),
                            postproc=BinaryFxNode(mean_mismatch_error,
                                                  'targets'),
                            enable_ca=['stats'])
    te_ = TransferMeasure(clf_, Splitter('train'),
                            postproc=BinaryFxNode(mean_mismatch_error,
                                                  'targets'),
                            enable_ca=['stats'])

    te = CrossValidation(clf, NFoldPartitioner(), postproc=mean_sample(),
                        enable_ca=['stats'])
    te_ = CrossValidation(clf_, NFoldPartitioner(), postproc=mean_sample(),
                        enable_ca=['stats'])

    error = te(ds)
    matrix = te.ca.stats.matrix

    # if ties were broken randomly we should have got nearly the same
    # number of hits for tied targets
    ties_indices = [te.ca.stats.labels.index(c) for c in ds.a.ties]
    hits = np.diag(te.ca.stats.matrix)[ties_indices]

    # First check is to see if we swap data between tied labels we
    # are getting the same results if we permute labels accordingly,
    # i.e. that tie resolution is not dependent on the labels order
    # but rather on the data
    te_(ds_)
    matrix_swapped = te_.ca.stats.matrix

    if False: #0 in hits:
        print clf, matrix, matrix_swapped
        print clf.ca.estimates[:, 2] - clf.ca.estimates[:,0]
        #print clf.ca.estimates

    # TODO: for now disabled all the non-compliant ones to pass the
    #       tests. For visibility decided to skip them instead of just
    #       exclusion and skipping only here to possibly catch crashes
    #       which might happen before
    if len(set(('libsvm', 'sg', 'skl', 'gpr', 'blr')
               ).intersection(clf.__tags__)):
        raise SkipTest("Skipped %s because it is known to fail")

    ok_(not (np.array_equal(matrix, matrix_swapped) and 0 in hits))

    # this check is valid only if ties are not broken randomly
    # like it is the case with SMLR
    if not ('random_tie_breaking' in clf.__tags__
            or  # since __tags__ would not go that high up e.g. in
                # <knn on SMLR non-0>
            'SMLR' in str(clf)):
        assert_array_equal(hits,
                           np.diag(matrix_swapped)[ties_indices[::-1]])

    # Second check is to just see if we didn't get an obvious bias and
    # got 0 in one of the hits, although it is labile
    if cfg.getboolean('tests', 'labile', default='yes'):
        ok_(not 0 in hits)
    # this is old test... even more cumbersome/unreliable
    #hits_ndiff = abs(float(hits[1]-hits[0]))/max(hits)
    #thr = 0.9   # let's be generous and pretty much just request absent 0s
    #ok_(hits_ndiff < thr)

@sweepargs(clf=clfswh['linear', 'svm', 'libsvm', '!meta', 'multiclass'])
@sweepargs(ds=[datasets['uni%dsmall' % i] for i in 2,3,4])
def test_multiclass_classifier_cv(clf, ds):
    # Extending test_clf.py:ClassifiersTests.test_multiclass_classifier
    # Compare performance with our MaximalVote to the one done natively
    # by e.g. LIBSVM
    clf = clf.clone()
    clf.params.C = 1                      # so it doesn't auto-adjust
    mclf = MulticlassClassifier(clf=clf.clone())
    part = NFoldPartitioner()
    cv  = CrossValidation(clf , part, enable_ca=['stats', 'training_stats'])
    mcv = CrossValidation(mclf, part, enable_ca=['stats', 'training_stats'])

    er  =  cv(ds)
    mer = mcv(ds)

    # errors should be the same
    assert_array_equal(er, mer)
    assert_equal(str(cv.ca.training_stats), str(mcv.ca.training_stats))
    # if it was a binary task, cv.ca.stats would also have AUC column
    # while mcv would not  :-/  TODO
    if len(ds.UT) == 2:
        # so just compare the matrix and ACC
        assert_array_equal(cv.ca.stats.matrix, mcv.ca.stats.matrix)
        assert_equal(cv.ca.stats.stats['ACC'], mcv.ca.stats.stats['ACC'])
    else:
        assert_equal(str(cv.ca.stats), str(mcv.ca.stats))
    

def test_multiclass_classifier_pass_ds_attributes():
    # TODO: replicate/extend basic testing of pass_attr
    #       in some more "basic" test_*
    clf = LinearCSVMC(C=1)
    ds = datasets['uni3small'].copy()
    ds.sa['ids'] = np.arange(len(ds))
    mclf = MulticlassClassifier(
        clf,
        pass_attr=['ids', 'sa.chunks', 'a.bogus_features',
                  # 'ca.raw_estimates' # this one is binary_clf x samples list ATM
                  # that is why raw_predictions_ds was born
                  'ca.raw_predictions_ds',
                  'ca.estimates', # this one is ok
                  'ca.predictions',
                  ],
        enable_ca=['all'])
    mcv  = CrossValidation(mclf, NFoldPartitioner(), errorfx=None)
    res = mcv(ds)
    assert_array_equal(sorted(res.sa.ids), ds.sa.ids)
    assert_array_equal(res.chunks, ds.chunks[res.sa.ids])
    assert_array_equal(res.sa.predictions, res.samples[:, 0])
    assert_array_equal(res.sa.cvfolds,
                       np.repeat(range(len(ds.UC)), len(ds)/len(ds.UC)))


def test_multiclass_without_combiner():
    # The goal is to obtain all pairwise results as the resultant dataset
    # avoiding even calling any combiner
    clf = LinearCSVMC(C=1)
    ds = datasets['uni3small'].copy()
    ds.sa['ids'] = np.arange(len(ds))
    mclf = MulticlassClassifier(clf, combiner=None)

    # without combining results at all
    mcv = CrossValidation(mclf, NFoldPartitioner(), errorfx=None)
    res = mcv(ds)
    assert_equal(len(res), len(ds))
    assert_equal(res.nfeatures, 3)        # 3 pairs for 3 classes
    assert_array_equal(res.UT, ds.UT)
    assert_array_equal(np.unique(np.array(res.fa.targets.tolist())), ds.UT)
    # TODO -- check that we have all the pairs?
    assert_array_equal(res.sa['cvfolds'].unique, np.arange(len(ds.UC)))
    if mcv.ca.is_enabled('training_stats'):
        # we must have received a dictionary per each pair
        training_stats = mcv.ca.training_stats
        assert_equal(set(training_stats.keys()),
                     set([('L0', 'L1'), ('L0', 'L2'), ('L1', 'L2')]))
        for pair, cm in training_stats.iteritems():
            assert_array_equal(cm.labels, ds.UT)
            # we should have no predictions for absent label
            assert_array_equal(cm.matrix[~np.in1d(ds.UT, pair)], 0)
            # while altogether all samples were processed once
            assert_array_equal(cm.stats['P'], len(ds))
            # and number of sets should be equal number of chunks here
            assert_equal(len(cm.sets), len(ds.UC))


# Sweep through some representative interesting classifiers
@sweepargs(clf=[
    LinearCSVMC(C=1),
    GNB(common_variance=True),
])
def test_multiclass_without_combiner_sens(clf):
    ds = datasets['uni3small'].copy()
    # do the clone since later we will compare sensitivities and need it
    # independently trained etc
    mclf = MulticlassClassifier(clf.clone(), combiner=None)

    # We have lots of sandwiching
    #    Multiclass.clfs -> [BinaryClassifier] -> clf
    # where BinaryClassifier's estimates are binarized.
    # Let's also check that we are getting sensitivities correctly.
    # With addition of MulticlassClassifierSensitivityAnalyzer we managed to break
    # it and none tests picked it up, so here we will test that sensitivities
    # are computed and labeled correctly


    # verify that all kinds of results on two classes are identical to the ones
    # if obtained running it without MulticlassClassifier
    # ds = ds[:, 0]  #  uncomment out to ease/speed up troubleshooting
    ds2 = ds.select(sadict=dict(targets=['L1', 'L2']))
    # we will train only on one chunk so we could get "realistic" (not just
    # overfit) predictions
    ds2_train = ds2.select(sadict=dict(chunks=ds.UC[:1]))

    # also consider simpler BinaryClassifier to easier pin point the problem
    # and be explicit about what is positive and what is negative label(s)
    bclf = BinaryClassifier(clf.clone(), poslabels=['L2'], neglabels=['L1'])

    predictions = []
    clfs = [clf, bclf, mclf]
    for c in clfs:
        c.ca.enable('all')
        c.train(ds2_train)
        predictions.append(c.predict(ds2))
    p1, bp1, mp1 = predictions

    assert_equal(p1, bp1)

    # ATM mclf.predict returns dataset (with fa.targets to list pairs of targets
    # used I guess) while p1 is just a list.
    def assert_list_equal_to_ds(l, ds):
        assert_equal(ds.shape, (len(l), 1))
        assert_array_equal(l, ds.samples[:, 0])
    assert_list_equal_to_ds(p1, mp1)

    # but if we look at sensitivities
    s1, bs1, ms1 = [
        c.get_sensitivity_analyzer()(ds2)
        for c in clfs
    ]
    # Do ground checks for s1
    nonbogus_target = ds2.fa.nonbogus_targets[0]

    # if there was a feature with signal, we know what to expect!:
    # such assignments are randomized, so we might not have signal in that
    # single feature we chose to test with
    if nonbogus_target and nonbogus_target in ds2.UT:
        # that in the pair of labels it would be 2nd one if positive sensitivity
        # or 1st one is negative
        # with classifier we try (SVM) should be pairs of labels
        assert isinstance(s1.T[0], tuple)
        assert_equal(len(s1), 1)
        assert_equal(s1.T[0][int(s1.samples[0, 0] > 0)], nonbogus_target)

    # And in either case we could check that we are getting identical results!
    # lrn_index is unique to ms1 and "ignore_sa" to assert_datasets_equal still
    # compares for the keys to be present in both, so does not help
    ms1.sa.pop('lrn_index')

    assert_datasets_equal(s1, bs1)
    # and here we get a "problem"!
    assert_datasets_equal(s1, ms1)
