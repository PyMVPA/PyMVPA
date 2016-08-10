# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for Stelzer et al. cluster thresholding algorithm"""

from mvpa2.base import externals
from mvpa2.testing.tools import skip_if_no_external

# TODO a tiny bit also needs statsmodels
skip_if_no_external('statsmodels')
skip_if_no_external('scipy')

from collections import Counter
import numpy as np
import random

from mvpa2.testing import assert_array_equal, assert_raises, assert_equal, \
    assert_array_almost_equal, assert_almost_equal, assert_true, assert_false
import mvpa2.algorithms.group_clusterthr as gct
from mvpa2.datasets import Dataset, dataset_wizard
from nose.tools import assert_greater_equal, assert_greater
from mvpa2.testing.sweep import sweepargs

from scipy.ndimage import measurements
from scipy.stats import norm


def test_pval():
    def not_inplace_shuffle(x):
        x = list(x)
        random.shuffle(x)
        return x

    x = range(100000) * 20
    x = np.array(x)
    x = x.reshape(20, 100000)
    x = x.T
    x = np.apply_along_axis(not_inplace_shuffle, axis=0, arr=x)
    expected_result = [100000 - 100000 * 0.001] * 20

    thresholds = gct.get_thresholding_map(x, p=0.001)
    assert_array_equal(thresholds, expected_result)
    # works with datasets too
    dsthresholds = gct.get_thresholding_map(Dataset(x), p=0.001)
    assert_almost_equal(thresholds, dsthresholds)
    assert_raises(ValueError,
                  gct.get_thresholding_map, x, p=0.00000001)

    x = range(0, 100, 5)
    null_dist = np.repeat(1, 100).astype(float)[None]
    pvals = gct._transform_to_pvals(x, null_dist)
    desired_output = np.array([1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6,
                               0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15,
                               0.1, 0.05])
    assert_array_almost_equal(desired_output, pvals)


def test_cluster_count():
    skip_if_no_external('scipy', min_version='0.10')
    # we get a ZERO cluster count of one if there are no clusters at all
    # this is needed to keept track of the number of bootstrap samples that yield
    # no cluster at all (high treshold) in order to compute p-values when there is no
    # actual cluster size histogram
    assert_equal(gct._get_map_cluster_sizes([0, 0, 0, 0]), [0])
    # if there is at least one cluster: no ZERO count
    assert_equal(gct._get_map_cluster_sizes([0, 0, 1, 0]), [1])
    for i in range(2):  # rerun tests for bool type of test_M
        test_M = np.array([[1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                           [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                           [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                           [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                           [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                           [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0],
                           [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0]])
        expected_result = [5, 4, 3, 3, 2, 0, 2]  # 5 clusters of size 1,
                                                 # 4 clusters of size 2 ...

        test_ds = Dataset([test_M])
        if i == 1:
            test_M = test_M.astype(bool)

        test_M_3d = np.hstack((test_M.flatten(),
                               test_M.flatten())).reshape(2, 9, 16)
        test_ds_3d = Dataset([test_M_3d])
        # expected_result^2
        expected_result_3d = np.array([0, 5, 0, 4, 0, 3, 0,
                                       3, 0, 2, 0, 0, 0, 2])

        size = 10000  # how many times bigger than test_M_3d
        test_M_3d_big = np.hstack((test_M_3d.flatten(), np.zeros(144)))
        test_M_3d_big = np.hstack((test_M_3d_big for i in range(size))
                                  ).reshape(3 * size, 9, 16)
        test_ds_3d_big = Dataset([test_M_3d_big])
        expected_result_3d_big = expected_result_3d * size

        # check basic cluster size determination for plain arrays and datasets
        # with a single sample
        for t, e in ((test_M, expected_result),
                     (test_ds, expected_result),
                     (test_M_3d, expected_result_3d),
                     (test_ds_3d, expected_result_3d),
                     (test_M_3d_big, expected_result_3d_big),
                     (test_ds_3d_big, expected_result_3d_big)):
            assert_array_equal(np.bincount(gct._get_map_cluster_sizes(t))[1:],
                               e)
        # old
        M = np.vstack([test_M_3d.flatten()] * 10)
        # new
        ds = dataset_wizard([test_M_3d] * 10)
        assert_array_equal(M, ds)
        expected_result = Counter(np.hstack([gct._get_map_cluster_sizes(test_M_3d)] * 10))
        assert_array_equal(expected_result,
                           gct.get_cluster_sizes(ds))

        # test the same with some arbitrary per-feature threshold
        thr = 4
        labels, num = measurements.label(test_M_3d)
        area = measurements.sum(test_M_3d, labels,
                                index=np.arange(labels.max() + 1))
        cluster_sizes_map = area[labels]  # .astype(int)
        thresholded_cluster_sizes_map = cluster_sizes_map > thr
        # old
        M = np.vstack([cluster_sizes_map.flatten()] * 10)
        # new
        ds = dataset_wizard([cluster_sizes_map] * 10)
        assert_array_equal(M, ds)
        expected_result = Counter(np.hstack(
            [gct._get_map_cluster_sizes(thresholded_cluster_sizes_map)] * 10))
        th_map = np.ones(cluster_sizes_map.flatten().shape) * thr
        # threshold dataset by hand
        ds.samples = ds.samples > th_map
        assert_array_equal(expected_result,
                           gct.get_cluster_sizes(ds))


# run same test with parallel and serial execution
@sweepargs(n_proc=[1, 2])
def test_group_clusterthreshold_simple(n_proc):
    if n_proc > 1:
        skip_if_no_external('joblib')
    feature_thresh_prob = 0.005
    nsubj = 10
    # make a nice 1D blob and a speck
    blob = np.array([0, 0, .5, 3, 5, 3, 3, 0, 2, 0])
    blob = Dataset([blob])
    # and some nice random permutations
    nperms = 100 * nsubj
    perm_samples = np.random.randn(nperms, blob.nfeatures)
    perms = Dataset(perm_samples,
                    sa=dict(chunks=np.repeat(range(nsubj), len(perm_samples) / nsubj)),
                    fa=dict(fid=range(perm_samples.shape[1])))
    # the algorithm instance
    # scale number of bootstraps to match desired probability
    # plus a safety margin to minimize bad luck in sampling
    clthr = gct.GroupClusterThreshold(n_bootstrap=int(3. / feature_thresh_prob),
                                      feature_thresh_prob=feature_thresh_prob,
                                      fwe_rate=0.01, n_blocks=3, n_proc=n_proc)
    clthr.train(perms)
    # get the FE thresholds
    thr = clthr._thrmap
    # perms are normally distributed, hence the CDF should be close, std of the distribution
    # will scale 1/sqrt(nsubj)
    assert_true(np.abs(
        feature_thresh_prob - (1 - norm.cdf(thr.mean(),
                                            loc=0,
                                            scale=1. / np.sqrt(nsubj)))) < 0.01)

    clstr_sizes = clthr._null_cluster_sizes
    # getting anything but a lonely one feature cluster is very unlikely
    assert_true(max([c[0] for c in clstr_sizes.keys()]) <= 1)
    # threshold orig map
    res = clthr(blob)
    #
    # check output
    #
    # samples unchanged
    assert_array_equal(blob.samples, res.samples)
    # need to find the big cluster
    assert_true(len(res.a.clusterstats) > 0)
    assert_equal(len(res.a.clusterstats), res.fa.clusters_featurewise_thresh.max())
    # probs need to decrease with size, clusters are sorted by size (decreasing)
    assert_true(res.a.clusterstats['prob_raw'][0] <= res.a.clusterstats['prob_raw'][1])
    # corrected probs for every uncorrected cluster
    assert_true('prob_corrected' in res.a.clusterstats.dtype.names)
    # fwe correction always increases the p-values (if anything)
    assert_true(np.all(res.a.clusterstats['prob_raw'] <= res.a.clusterstats['prob_corrected']))
    # check expected cluster sizes, ordered large -> small
    assert_array_equal(res.a.clusterstats['size'], [4, 1])
    # check max position
    assert_array_equal(res.a.clusterlocations['max'], [[4], [8]])
    # center of mass: eyeballed
    assert_array_almost_equal(res.a.clusterlocations['center_of_mass'],
                              [[4.429], [8]],
                              3)
    # other simple stats
    #[0, 0, .5, 3, 5, 3, 3, 0, 2, 0]
    assert_array_equal(res.a.clusterstats['mean'], [3.5, 2])
    assert_array_equal(res.a.clusterstats['min'], [3, 2])
    assert_array_equal(res.a.clusterstats['max'], [5, 2])
    assert_array_equal(res.a.clusterstats['median'], [3, 2])
    assert_array_almost_equal(res.a.clusterstats['std'], [0.866, 0], 3)

    # fwe thresholding only ever removes clusters
    assert_true(np.all(np.abs(res.fa.clusters_featurewise_thresh - res.fa.clusters_fwe_thresh) >= 0))
    # FWE should kill the small one
    assert_greater(res.fa.clusters_featurewise_thresh.max(),
                   res.fa.clusters_fwe_thresh.max())

    # check that the cluster results aren't depending in the actual location of
    # the clusters
    shifted_blob = Dataset([[.5, 3, 5, 3, 3, 0, 0, 0, 2, 0]])
    shifted_res = clthr(shifted_blob)
    assert_array_equal(res.a.clusterstats, shifted_res.a.clusterstats)

    # check that it averages multi-sample datasets
    # also checks that scenarios work where all features are part of one big
    # cluster
    multisamp = Dataset(np.arange(30).reshape(3, 10) + 100)
    avgres = clthr(multisamp)
    assert_equal(len(avgres), 1)
    assert_array_equal(avgres.samples[0], np.mean(multisamp.samples, axis=0))

    # retrain, this time with data from only a single subject
    perms = Dataset(perm_samples,
                    sa=dict(chunks=np.repeat(1, len(perm_samples))),
                    fa=dict(fid=range(perms.shape[1])))
    clthr.train(perms)
    # same blob -- 1st this should work without issues
    sglres = clthr(blob)
    # NULL estimation does no averaging
    # -> more noise -> fewer clusters -> higher p
    assert_greater_equal(len(res.a.clusterstats), len(sglres.a.clusterstats))
    assert_greater_equal(np.round(sglres.a.clusterstats[0]['prob_raw'], 4),
                         np.round(res.a.clusterstats[0]['prob_raw'], 4))
    # no again for real scientists: no FWE correction
    superclthr = gct.GroupClusterThreshold(
        n_bootstrap=int(3. / feature_thresh_prob),
        feature_thresh_prob=feature_thresh_prob,
        multicomp_correction=None, n_blocks=3,
        n_proc=n_proc)
    superclthr.train(perms)
    superres = superclthr(blob)
    assert_true('prob_corrected' in res.a.clusterstats.dtype.names)
    assert_true('clusters_fwe_thresh' in res.fa)
    assert_false('prob_corrected' in superres.a.clusterstats.dtype.names)
    assert_false('clusters_fwe_thresh' in superres.fa)

    # check validity test
    assert_raises(ValueError, gct.GroupClusterThreshold,
                  n_bootstrap=10, feature_thresh_prob=.09, n_proc=n_proc)
    # check mapped datasets
    blob = np.array([[0, 0, .5, 3, 5, 3, 3, 0, 2, 0],
                     [0, 0,  0, 0, 0, 0, 0, 0, 0, 0]])
    blob = dataset_wizard([blob])
    # and some nice random permutations
    nperms = 100 * nsubj
    perm_samples = np.random.randn(*((nperms,) + blob.shape))
    perms = dataset_wizard(
        perm_samples, chunks=np.repeat(range(nsubj), len(perm_samples) / nsubj))
    clthr.train(perms)
    twodres = clthr(blob)
    # finds two clusters of the same size
    assert_array_equal(twodres.a.clusterstats['size'],
                       res.a.clusterstats['size'])

    # TODO continue with somewhat more real dataset

def test_repeat_cluster_vals():
    assert_array_equal(gct.repeat_cluster_vals({1: 2, 3: 1}), [1, 1, 3])
    assert_array_equal(gct.repeat_cluster_vals({1: 2, 3: 2, 2: 1}),
                       [1, 1, 2, 3, 3])

    assert_array_equal(gct.repeat_cluster_vals({1: 2, 3: 1}, {1: 0.2, 3: 0.5}),
                       [0.2, 0.2, 0.5])
    assert_array_equal(gct.repeat_cluster_vals({1: 2, 3: 2, 2: 1}, {1: 'a', 2: 'b', 3: 'c'}),
                       ['a', 'a', 'b', 'c', 'c'])
