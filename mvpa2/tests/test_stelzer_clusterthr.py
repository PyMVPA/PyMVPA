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

from mvpa2.testing import assert_array_equal

import mvpa2.algorithms.stelzer_clusterthr as sct

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

    z = sct.get_thresholding_map(x)
    # TODO: turn into actual test
    print z


def test_cluster_count():
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

    test_M_3d = np.hstack((test_M.flatten(),  test_M.flatten())).reshape(2, 9, 16)
    # expected_result^2
    expected_result_3d = np.array([0, 5, 0, 4, 0, 3, 0, 3, 0, 2, 0, 0, 0, 2])

    size = 10000  # how many times bigger than test_M_3d
    test_M_3d_big = np.hstack((test_M_3d.flatten(), np.zeros(144)))
    test_M_3d_big = np.hstack((test_M_3d_big for i in range(size))
                           ).reshape(3 * size, 9, 16)
    expected_result_3d_big = expected_result_3d * size

    # visualize clusters in test_M,
    # usefull if numbers to colors synesthesia not imported
#    imshow(test_M, interpolation='nearest')
#    show()
#    labels, num = measurements.label(test_M)
#    area = measurements.sum(test_M, labels, index=arange(labels.max() + 1))
#    areaImg = area[labels]
#    print areaImg.shape
#    imshow(areaImg, origin='lower', interpolation='nearest')
#    colorbar()
#    show()
#    area = area.astype(int)
#    print np.bincount(area)

    np.array_equal(np.bincount(sct.get_map_cluster_sizes(test_M))[1:],
                         expected_result)
    np.array_equal(np.bincount(sct.get_map_cluster_sizes(test_M_3d))[1:],
                         expected_result_3d)
    np.array_equal(np.bincount(sct.get_map_cluster_sizes(test_M_3d_big))[1:],
                         expected_result_3d_big)


