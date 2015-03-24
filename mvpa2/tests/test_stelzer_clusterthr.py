# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for Stelzer et al. cluster thresholding algorithm"""

import numpy as np
import random

from mvpa2.testing import assert_array_equal, assert_raises
from mvpa2.testing import assert_array_almost_equal, assert_almost_equal
import mvpa2.algorithms.stelzer_clusterthr as sct
from scipy.ndimage import measurements


def test_thresholding():
    M = np.array([[0, 1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5, 0],
                  [2, 3, 4, 5, 0, 1],
                  [3, 4, 5, 0, 1, 2],
                  [4, 5, 0, 1, 2, 3]])
    thresholding = [3, 2, 0, 1, 5, 7]
    expected_result = np.array([[0, 0, 1, 1, 0, 0],
                                [0, 0, 1, 1, 0, 0],
                                [0, 1, 1, 1, 0, 0],
                                [0, 1, 1, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0]])

    np.array_equal(sct.threshold(M, thresholding),
                   expected_result)


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

    thresholds = sct.get_thresholding_map(x, p=0.001)
    assert_array_equal(thresholds, expected_result)
    assert_raises(AssertionError,
                  sct.get_thresholding_map, x, p=0.00000001)

    x = range(0,100,5)
    null_dist = range(100)
    pvals = sct.transform_to_pvals(x, null_dist)
    desired_output = np.array([1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6,
                      0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1,
                      0.05])
    assert_array_almost_equal(desired_output, pvals)

    x = range(100)
    random.shuffle(x)
    y = sct.get_pval(95, x)
    desired_output = 0.05
    assert_almost_equal(y, desired_output)


def test_unmask():

    mask = np.zeros(100)
    to_mask = np.array(range(0, 100, 10))
    mask[to_mask] = 1
    arr = np.array(range(1, 11))

    unmasked = sct.unmask(arr, mask, (100,))
    desired_output = mask.copy()
    desired_output[mask == 1] = range(1, 11)

    assert_array_equal(unmasked, desired_output)
    assert_array_equal(sct.unmask(arr, mask, (20, 5)),
                       desired_output.reshape(20, 5))
    assert_raises(AssertionError, sct.unmask, arr.reshape(2,5),
                  mask, (20, 5))


def test_cluster_count():
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
        if i == 1:
            test_M = test_M.astype(bool)

        test_M_3d = np.hstack((test_M.flatten(),
                               test_M.flatten())).reshape(2, 9, 16)
        # expected_result^2
        expected_result_3d = np.array([0, 5, 0, 4, 0, 3, 0,
                                       3, 0, 2, 0, 0, 0, 2])

        size = 10000  # how many times bigger than test_M_3d
        test_M_3d_big = np.hstack((test_M_3d.flatten(), np.zeros(144)))
        test_M_3d_big = np.hstack((test_M_3d_big for i in range(size))
                               ).reshape(3 * size, 9, 16)
        expected_result_3d_big = expected_result_3d * size

#        # visualize clusters in test_M,
#        # usefull if numbers to colors synesthesia not imported
#        imshow(test_M, interpolation='nearest')
#        show()
#        labels, num = measurements.label(test_M)
#        area = measurements.sum(test_M, labels,
#                                 index=arange(labels.max() + 1))
#        areaImg = area[labels]
#        print areaImg.shape
#        imshow(areaImg, origin='lower', interpolation='nearest')
#        colorbar()
#        show()
#        area = area.astype(int)
#        print np.bincount(area)

        assert_array_equal(
            np.bincount(sct.get_map_cluster_sizes(test_M))[1:],
                        expected_result)
        assert_array_equal(
            np.bincount(sct.get_map_cluster_sizes(test_M_3d))[1:],
                        expected_result_3d)
        assert_array_equal(
            np.bincount(sct.get_map_cluster_sizes(test_M_3d_big))[1:],
                        expected_result_3d_big)

        M = []
        for i in range(10):
            M.append(test_M_3d.flatten())
        M = np.vstack(M)
        expected_result = np.hstack([sct.get_map_cluster_sizes(test_M_3d)]*10)
        mask = np.ones(len(test_M_3d.flatten()))
        shape = test_M_3d.shape
        assert_array_equal(expected_result,
                           sct.get_null_dist_clusters(M, mask,
                                                      shape,
                                                      thresholded=True))

        doesnt_matter = range(100)
        assert_array_equal(sct.label_clusters(doesnt_matter,
                                              test_M_3d,
                                               # not testing correction
                                              method="None",
                                               # not testing rejection
                                              alpha=1,
                                 return_type="binary_map"), test_M_3d)

        labels, num = measurements.label(test_M_3d)
        area = measurements.sum(test_M_3d, labels,
                                index=np.arange(labels.max() + 1))
        cluster_sizes = area[labels].astype(int)
        assert_array_equal(sct.label_clusters(doesnt_matter,
                                              test_M_3d,
                                              method="None",
                                              alpha=1,
                                              return_type="cluster_sizes"),
                                              cluster_sizes)

        assert_raises(AssertionError, sct.label_clusters, doesnt_matter,
                                         test_M_3d,
                                         method="None",
                                         alpha=1,
                                         return_type="UNKNOWN")


test_cluster_count()
