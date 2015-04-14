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
from mvpa2.testing.tools import SkipTest

# TODO a tiny bit also needs statsmodels
if not externals.exists('scipy'):
    raise SkipTest

from collections import Counter
import numpy as np
import random

from mvpa2.testing import assert_array_equal, assert_raises, assert_equal, \
    assert_array_almost_equal, assert_almost_equal, assert_true
import mvpa2.algorithms.group_clusterthr as gct
from mvpa2.datasets import Dataset, dataset_wizard
from mvpa2.mappers.base import IdentityMapper

from scipy.ndimage import measurements

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
    expected_result = [100000-100000*0.001]*20

    thresholds = gct.get_thresholding_map(x, p=0.001)
    assert_array_equal(thresholds, expected_result)
    # works with datasets too
    dsthresholds = gct.get_thresholding_map(Dataset(x), p=0.001)
    assert_almost_equal(thresholds, dsthresholds)
    assert_raises(ValueError,
                  gct.get_thresholding_map, x, p=0.00000001)

    x = range(0,100,5)
    null_dist = range(100)
    pvals = gct.transform_to_pvals(x, null_dist)
    desired_output = np.array([1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6,
                      0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1,
                      0.05])
    assert_array_almost_equal(desired_output, pvals)

    x = range(100)
    random.shuffle(x)
    y = gct.get_pval(95, x)
    desired_output = 0.05
    assert_almost_equal(y, desired_output)


def test_cluster_count():
    # we get a ZERO cluster count of one if there are no clusters at all
    # this is needed to keept track of the number of bootstrap samples that yield
    # no cluster at all (high treshold) in order to compute p-values when there is no
    # actual cluster size histogram
    assert_equal(gct._get_map_cluster_sizes([0,0,0,0]), [0])
    # if there is at least one cluster: no ZERO count
    assert_equal(gct._get_map_cluster_sizes([0,0,1,0]), [1])
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
        M = np.vstack([test_M_3d.flatten()]*10)
        # new
        ds = dataset_wizard([test_M_3d] * 10)
        assert_array_equal(M, ds)
        expected_result = Counter(np.hstack([gct._get_map_cluster_sizes(test_M_3d)]*10))
        assert_array_equal(expected_result,
                           gct.get_cluster_sizes(ds))

        # test the same with some arbitrary per-feature threshold
        thr = 4
        labels, num = measurements.label(test_M_3d)
        area = measurements.sum(test_M_3d, labels,
                                index=np.arange(labels.max() + 1))
        cluster_sizes_map = area[labels]  #.astype(int)
        thresholded_cluster_sizes_map = cluster_sizes_map > thr
        # old
        M = np.vstack([cluster_sizes_map.flatten()]*10)
        # new
        ds = dataset_wizard([cluster_sizes_map] * 10)
        assert_array_equal(M, ds)
        expected_result = Counter(np.hstack([gct._get_map_cluster_sizes(
                                         thresholded_cluster_sizes_map)]*10))
        th_map = np.ones(cluster_sizes_map.flatten().shape) * thr
        # threshold dataset by hand
        ds.samples = ds.samples > th_map
        assert_array_equal(expected_result,
                           gct.get_cluster_sizes(ds))

        dumm_null_dist = range(10)
        assert_array_equal(gct.label_clusters(dumm_null_dist,
                                              test_M_3d,
                                               # not testing correction
                                              method="None",
                                               # not testing rejection
                                              alpha=1,
                                 return_type="binary_map"), test_M_3d)

        assert_array_equal(gct.label_clusters(dumm_null_dist,
                                              test_M_3d,
                                              method="None",
                                              alpha=1,
                                              return_type="cluster_sizes"),
                                              cluster_sizes_map)

        assert_raises(AssertionError, gct.label_clusters, dumm_null_dist,
                                         test_M_3d,
                                         method="None",
                                         alpha=1,
                                         return_type="UNKNOWN")

        clusters = gct._get_map_cluster_sizes(test_M_3d)
        x = np.hstack([dumm_null_dist, clusters])
        pvals = np.array(gct.transform_to_pvals(clusters, x))
        pvals = 1-pvals
        pval_map = cluster_sizes_map.copy()
        for clust_size, pval in set(zip(clusters, pvals)):
            pval_map[pval_map==clust_size] = pval
        labeled = gct.label_clusters(dumm_null_dist, test_M_3d,
                  method="None", return_type='p_vals')
        assert_array_equal(labeled, pval_map)

        thresholded_pval_map = pval_map.copy()
        thresholded_pval_map[pval_map < 0.5] = 0
        labeled = gct.label_clusters(dumm_null_dist, test_M_3d,
                  method="None", alpha=0.5, return_type='thresholded_p_vals')
        assert_array_equal(labeled, thresholded_pval_map)

        num_of_clusters = len(clusters)
        labeled = gct.label_clusters(dumm_null_dist, test_M_3d,
                  method="None", return_type='unique_clusters')

        # num_of_clusters +1 because there is also +1 cluster for 0 value
        assert_equal(num_of_clusters+1, len(np.unique(labeled)))

def test_acccluster_threshold():
    from scipy.stats import norm
    feprob = 0.005
    nsubj = 10
    # make a nice 1D blob and a speck
    blob = np.array([3,0,1,3,5,3,2,0,0,0])
    blob = Dataset([blob])
    # and some nice random permutations
    nperms = 100 * nsubj
    perms = np.random.randn(nperms, blob.nfeatures)
    perms = Dataset(perms,
                    sa=dict(chunks=np.repeat(range(nsubj), len(perms) / nsubj)),
                    fa=dict(fid=range(perms.shape[1])))
    # the algorithm instance
    # scale number of bootstraps to match desired probability
    clthr = gct.ACCClusterThreshold(n_bootstrap=int(1./feprob), feprob=feprob,
            fwe_rate=0.05, n_blocks=3)
    clthr.train(perms)
    # get the FE thresholds
    thr = clthr._thrmap
    # perms are normally distributed, hence the CDF should be close, std of the distribution
    # will scale 1/sqrt(nsubj)
    assert_true(np.abs(feprob - (1 - norm.cdf(thr.mean(), loc=0, scale=1./np.sqrt(nsubj)))) < 0.01)

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
    assert_true(len(res.a.cluster_probs_uncorrected) > 0)
    assert_equal(len(res.a.cluster_probs_uncorrected), res.fa.clusters_voxelwise_thresh.max())
    # probs need to decrease with size
    assert_true(res.a.cluster_probs_uncorrected[1] >= res.a.cluster_probs_uncorrected[2])
    # corrected probs for every uncorrected cluster
    assert_equal(len(res.a.cluster_probs_uncorrected), len(res.a.cluster_probs_fwe_corrected))
    # fwe correction always increases the p-values (if anything)
    assert_true(np.all(res.a.cluster_probs_uncorrected <= res.a.cluster_probs_fwe_corrected))
    # fwe thresholding only ever removes clusters
    assert_true(np.all(np.abs(res.fa.clusters_voxelwise_thresh - res.fa.clusters_fwe_thresh) >= 0))

    # TODO continue with somewhat more real dataset
