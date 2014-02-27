# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Functions to deal with pair-wise multiclass classification

Result of the Yarik's brain-block forbidding proper formalization
within PyMVPA.  For now only used/tested only in a single test within
test_usecases.py
"""

import numpy as np

from mvpa2.datasets import Dataset

# Multi-class classification
def _get_unique_in_axis(a, axis_=-1):
    """Obtain pairs of present targets from the dataset

    Parameters
    ----------
    a : array
    axis_ : int
      Axis (!=0) to get unique values for each element of 
    """
    assert axis_ == -1, "others not implemented yet"
    return np.apply_along_axis(
        np.unique, 0,
        ## reshape slmap so we have only simple pairs in the columns
        np.reshape(a, (-1, a.shape[axis_]))).T


def get_pairwise_hits_misses(a, targets, pairs=None, select=None, axis=-1):
    """For all-pairs results extract results per each pair as hits and misses

    This function assumes that the last dimension is the one
    sweeping through the pairs, thus it could readily be applied
    to the results from searchlight

    Parameters
    ---------
    select : list, optional
      Deal only with those targets listed here, omitting the others
    axis : int, optional
      Contains predictions over which to gather hits/misses
    """
    results = []
    result_pairs = []
    a = np.asanyarray(a)
    assert axis == -1, "others not implemented yet"
    if pairs is None:
        pairs = _get_unique_in_axis(a)
    # This is a somewhat slow implementation for now. optimize later
    for i, p in enumerate(pairs):
        if select is not None and len(set(p).difference(select)):
            # skip those which are not among 'select'
            continue
        # select only those samples which have targets in the pair
        idx = np.in1d(targets, p)
        p_samples = a[idx, ..., i]
        p_targets = targets[idx]

        if p_samples.ndim > 1:
            p_targets = p_targets[:, None]
        hits_all = (p_samples == p_targets)
        hits = np.sum(hits_all, axis=0)
        misses = len(hits_all) - hits
        results.append((hits, misses))
        result_pairs.append(p)
    return result_pairs, results

# Probably it should become a mapper -- may be later and in a more
# generic way
def get_pairwise_accuracies(ds, stat='acc', pairs=None, select=None, space='targets'):
    """Extract pair-wise classification performances as a dataset

    Converts a dataset of classifications for all pairs of
    classifiers (e.g. obtained from raw_predictions_ds of a
    MulticlassClassifier) into a dataset of performances for each
    pair of stimuli categories.  I.e. only the pair-wise results
    where the target label matches one of the targets in the pair
    will contribute to the count.

    Parameters
    ----------
    pairs : 
      Pairs of targets corresponding to the last dimension in the
      provided dataset
    select : list, optional
      Deal only with those targets listed here, omitting the others
    """
    pairs, hits_misses = get_pairwise_hits_misses(
        ds.samples, ds.sa[space].value, pairs=pairs, select=select)
    hits_misses = np.array(hits_misses)
    if stat in ['acc']:
        stat_values = hits_misses[:, 0].astype(float)/np.sum(hits_misses, axis=1)
        stat_fa = [stat]
    elif stat == 'hits_misses':
        stat_values = hits_misses
        stat_fa = ['hits', 'misses']
    else:
        raise NotImplementedError("%s statistic not there yet" % stat)

    return Dataset(stat_values, sa={space: pairs}, fa={'stat': stat_fa})
