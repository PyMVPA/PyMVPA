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

## from mvpa2.generators.partition import NFoldPartitioner, OddEvenPartitioner
from mvpa2.generators.splitters import Splitter

from mvpa2.clfs.meta import CombinedClassifier, \
     BinaryClassifier, MulticlassClassifier, \
     MaximalVote
from mvpa2.measures.base import TransferMeasure ##, ProxyMeasure, CrossValidation
from mvpa2.mappers.fx import mean_sample, BinaryFxNode
from mvpa2.misc.errorfx import mean_mismatch_error



# Generate test data for testing ties
@reseed_rng()
def get_dsties1():
    ds = datasets['uni2small'].copy()
    dtarget = ds.targets[0]             # duplicate target
    tied_samples = ds.targets == dtarget
    ds2 = ds[tied_samples].copy(deep=True)
    # add similar noise to both ties
    noise_level = 0.01
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

@sweepargs(clf=clfswh['multiclass'])
def test_multiclass_ties(clf):
    ds = _dsties1

    # reassign data between ties, so we know that decision is data, not order driven
    ds_ = ds.copy(deep=True)
    ds_.samples[ds.a.ties_idx[1]] = ds.samples[ds.a.ties_idx[0]]
    ds_.samples[ds.a.ties_idx[0]] = ds.samples[ds.a.ties_idx[1]]
    ok_(np.any(ds_.samples != ds.samples))

    clf = clf.clone()
    clf.ca.enable('estimates')
    te = TransferMeasure(clf, Splitter('train'),
                            postproc=BinaryFxNode(mean_mismatch_error,
                                                  'targets'),
                            enable_ca=['stats'])
    #te = CrossValidation(clf, NFoldPartitioner(), postproc=mean_sample(),
    #                    enable_ca=['stats'])
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
    te(ds_)
    matrix_swapped = te.ca.stats.matrix
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
