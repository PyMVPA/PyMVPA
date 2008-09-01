#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Distance functions to be used in kernels and elsewhere
"""

__docformat__ = 'restructuredtext'

import numpy as N
from scipy import weave
from scipy.weave import converters

if __debug__:
    from mvpa.base import debug, warning


def cartesianDistance(a, b):
    """Return Cartesian distance between a and b
    """
    return N.linalg.norm(a-b)


def absminDistance(a, b):
    """Returns dinstance max(\|a-b\|)
    XXX There must be better name!

    Useful to select a whole cube of a given "radius"
    """
    return max(abs(a-b))


def manhattenDistance(a, b):
    """Return Manhatten distance between a and b
    """
    return sum(abs(a-b))


def mahalanobisDistance(x, y=None, w=None):
    """Caclulcate Mahalanobis distance of the pairs of points.

    :Parameters:
      `x`
        first list of points. Rows are samples, columns are
        features.
      `y`
        second list of points (optional)
      `w` : N.ndarray
        optional inverse covariance matrix between the points. It is
        computed if not given

    Inverse covariance matrix can be calculated with the following

      w = N.linalg.solve(N.cov(x.T),N.identity(x.shape[1]))

    or

      w = N.linalg.inv(N.cov(x.T))
    """
    # see if pairwise between two matrices or just within a single matrix
    if y is None:
        # pairwise distances of single matrix
        # calculate the inverse correlation matrix if necessary
        if w is None:
            w = N.linalg.inv(N.cov(x.T))

        # get some shapes of the data
        mx, nx = x.shape
        #mw, nw = w.shape

        # allocate for the matrix to fill
        d = N.zeros((mx, mx), dtype=N.float32)
        for i in range(mx-1):
            # get the current row to compare
            xi = x[i, :]
            # replicate the row
            xi = xi[N.newaxis, :].repeat(mx-i-1, axis=0)
            # take the distance between all the matrices
            dc = x[i+1:mx, :] - xi
            # scale the distance by the correlation
            d[i+1:mx, i] = N.real(N.sum((N.inner(dc, w) * N.conj(dc)), 1))
            # fill the other direction of the matrix
            d[i, i+1:mx] = d[i+1:mx, i].T
    else:
        # is between two matrixes
        # calculate the inverse correlation matrix if necessary
        if w is None:
            # calculate over all points
            w = N.linalg.inv(N.cov(N.concatenate((x, y)).T))

        # get some shapes of the data
        mx, nx = x.shape
        my, ny = y.shape

        # allocate for the matrix to fill
        d = N.zeros((mx, my), dtype=N.float32)

        # loop over shorter of two dimensions
        if mx <= my:
            # loop over the x patterns
            for i in range(mx):
                # get the current row to compare
                xi = x[i, :]
                # replicate the row
                xi = xi[N.newaxis, :].repeat(my, axis=0)
                # take the distance between all the matrices
                dc = xi - y
                # scale the distance by the correlation
                d[i, :] = N.real(N.sum((N.inner(dc, w) * N.conj(dc)), 1))
        else:
            # loop over the y patterns
            for j in range(my):
                # get the current row to compare
                yj = y[j, :]
                # replicate the row
                yj = yj[N.newaxis, :].repeat(mx, axis=0)
                # take the distance between all the matrices
                dc = x - yj
                # scale the distance by the correlation
                d[:, j] = N.real(N.sum((N.inner(dc, w) * N.conj(dc)), 1))

    # return the dist
    return N.sqrt(d)


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


def pnorm_w(data1, data2=None, weight=None, p=2, python=False):
    """Weighted p-norm between two datasets.

    ||x - x'||_w = (\sum_{i=1...N} (w_i*|x_i - x'_i|)**p)**(1/p)
    """
    if p == 2 and python:
        return N.sqrt(squared_euclidean_distance(data1=data1, data2=data2, weight=weight**2))

    if weight == None:
        weight = numpy.ones(data.shape[1],'d')
        pass
    size1 = data1.shape[0]
    F1 = data1.shape[1]
    code = ""
    if data2 == None or id(data1)==id(data2):
        assert(F1==weight.size) # Assert correct dimensions
        F = F1
        d = N.zeros((size1,size1),'d')
        if p == 1.0:
            code = """
            int i,j,t;
            double tmp;
            for (i=0;i<size1-1;i++) {
                for (j=i+1;j<size1;j++) {
                    tmp = 0.0;
                    for(t=0;t<F;t++) {
                        tmp = tmp+weight(t)*fabs(data1(i,t)-data1(j,t));
                        }
                    d(i,j) = tmp;
                    }
                }
            return_val = 0;
            """
        elif p == 2.0:
            code = """
            int i,j,t;
            double tmp, tmp2;
            for (i=0;i<size1-1;i++) {
                for (j=i+1;j<size1;j++) {
                    tmp = 0.0;
                    for(t=0;t<F;t++) {
                        tmp2 = weight(t)*fabs(data1(i,t)-data1(j,t));
                        tmp = tmp + tmp2*tmp2;
                        }
                    d(i,j) = tmp;
                    }
                }
            return_val = 0;
            """
        else:
            code = """
            int i,j,t;
            double tmp;
            for (i=0;i<size1-1;i++) {
                for (j=i+1;j<size1;j++) {
                    tmp = 0.0;
                    for(t=0;t<F;t++) {
                        tmp = tmp+pow(weight(t)*fabs(data1(i,t)-data1(j,t)),p);
                        }
                    d(i,j) = tmp;
                    }
                }
            return_val = 0;
            """
            pass
        counter = weave.inline(code,
                           ['data1','size1','F','weight','d','p'],
                           type_converters=converters.blitz,
                           compiler = 'gcc')
        d = d+N.triu(d).T # copy upper part to lower part
        return d**(1.0/p)
    
        pass
    size2 = data2.shape[0]
    F2 = data2.shape[1]
    assert(F1==F2==weight.size) # Assert correct dimensions
    F = F1
    d = N.zeros((size1,size2),'d')
    if p == 1.0:
        code = """
        int i,j,t;
        double tmp;
        for (i=0;i<size1;i++) {
            for (j=0;j<size2;j++) {
                tmp = 0.0;
                for(t=0;t<F;t++) {
                    tmp = tmp+weight(t)*fabs(data1(i,t)-data2(j,t));
                    }
                d(i,j) = tmp;
                }
            }
        return_val = 0;

        """
    elif p == 2.0:
        code = """
        int i,j,t;
        double tmp, tmp2;
        for (i=0;i<size1;i++) {
            for (j=0;j<size2;j++) {
                tmp = 0.0;
                for(t=0;t<F;t++) {
                    tmp2 = weight(t)*(data1(i,t)-data2(j,t));
                    tmp = tmp+tmp2*tmp2;
                    }
                d(i,j) = tmp;
                }
            }
        return_val = 0;

        """        
    else:
        code = """
        int i,j,t;
        double tmp;
        for (i=0;i<size1;i++) {
            for (j=0;j<size2;j++) {
                tmp = 0.0;
                for(t=0;t<F;t++) {
                    tmp = tmp+pow(weight(t)*fabs(data1(i,t)-data2(j,t)),p);
                    }
                d(i,j) = tmp;
                }
            }
        return_val = 0;
        """
        pass
    counter = weave.inline(code,
                           ['data1','data2','size1','size2','F','weight','d','p'],
                           type_converters=converters.blitz,
                           compiler = 'gcc')
    return d**(1.0/p)

