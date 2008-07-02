#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   Copyright (c) 2008 Emanuele Olivetti <emanuele@relativita.com>
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Distance functions to be used in kernels and elsewhere
"""

__docformat__ = 'restructuredtext'

import numpy as N

if __debug__:
    from mvpa.base import debug, warning


def squared_euclidean_distance(data1, data2=None, weight=None):
    """Compute weighted euclidean distance matrix between two datasets.


    :Parameters:
      data1 : numpy.ndarray
          first dataset
      data2 : numpy.ndarray
          second dataset. If None, compute the euclidean distance between
          the first dataset versus itself.
          (Defaults to None)
      weight : numpy.ndarray
          vector of weights, each one associated to each dimension of the
          dataset (Defaults to None)
    """
    if __debug__:
        # check if both datasets are floating point
        if not N.issubdtype(data1.dtype, 'f') \
           or (data2 is not None and not N.issubdtype(data2.dtype, 'f')):
            warning('Computing euclidean distance on integer data ' \
                    'is not supported.')

    # removed for efficiency (see below)
    #if weight is None:
    #    weight = N.ones(data1.shape[1], 'd') # unitary weight

    # In the following you can find faster implementations of this
    # basic code:
    #
    # squared_euclidean_distance_matrix = \
    #           N.zeros((data1.shape[0], data2.shape[0]), 'd')
    # for i in range(size1):
    #     for j in range(size2):
    #         squared_euclidean_distance_matrix[i,j] = \
    #           ((data1[i,:]-data2[j,:])**2*weight).sum()
    #         pass
    #     pass

    # Fast computation of distance matrix in Python+NumPy,
    # adapted from Bill Baxter's post on [numpy-discussion].
    # Basically: (x-y)**2*w = x*w*x - 2*x*w*y + y*y*w

    # based on value of weight and data2 we might save on computation
    # and resources
    if weight is None:
        data1w = data1
        if data2 is None:
            data2, data2w = data1, data1w
        else:
            data2w = data2
    else:
        data1w = data1 * weight
        if data2 is None:
            data2, data2w = data1, data1w
        else:
            data2w = data2 * weight

    squared_euclidean_distance_matrix = \
        (data1w * data1).sum(1)[:, None] \
        -2 * N.dot(data1w, data2.T) \
        + (data2 * data2w).sum(1)

    # correction to some possible numerical instabilities:
    less0 = squared_euclidean_distance_matrix < 0
    if __debug__ and 'CHECK_STABILITY' in debug.active:
        less0num = N.sum(less0)
        if less0num > 0:
            norm0 = N.linalg.norm(squared_euclidean_distance_matrix[less0])
            totalnorm = N.linalg.norm(squared_euclidean_distance_matrix)
            if totalnorm !=0 and norm0 / totalnorm > 1e-8:
                warning("Found %d elements out of %d unstable (<0) in " \
                        "computation of squared_euclidean_distance_matrix. " \
                        "Their norm is %s when total norm is %s" % \
                        (less0num, N.sum(less0.shape), norm0, totalnorm))
    squared_euclidean_distance_matrix[less0] = 0
    return squared_euclidean_distance_matrix

