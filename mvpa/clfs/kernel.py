#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   Copyright (c) 2008 Emanuele Olivetti <emanuele@relativita.com>
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Kernels for Gaussian Process Regression and Classification."""

__docformat__ = 'restructuredtext'


import numpy as N

if __debug__:
    import time
    from mvpa.misc import debug


class Kernel(object):
    """Kernel function base class.

    """

    def __init__(self):
        self.euclidean_distance_matrix = None

    def __repr__(self):
        return "Kernel()"

    def euclidean_distance(self, data1, data2=None, weight=None,
                           symmetric=False):
        """Compute weighted euclidean distance matrix between two datasets.


        :Parameters:
          data1 : numpy.ndarray
              first dataset
          data2 : numpy.ndarray
              second dataset. If None set symmetric to True.
              (Defaults to None)
          weight : numpy.ndarray
              vector of weights, each one associated to each dimension of the
              dataset (Defaults to None)
          symmetric : bool
              compute the euclidean distance between the first dataset versus
              itself (True) or the second one (False). Note that
              (Defaults to False)
        """

        if data2 is None:
            data2 = data1
            symmetric = True
            pass

        size1 = data1.shape[0]
        size2 = data2.shape[0]
        F = data1.shape[1]
        if weight is None:
            weight = N.ones(F,'d') # unitary weight
            pass

        euclidean_distance_matrix = N.zeros((data1.shape[0], data2.shape[0]),
                                            'd')
        # In the following you can find faster implementations of the
        # following code:
        #
        # for i in range(size1):
        #     for j in range(size2):
        #         euclidean_distance_matrix[i,j] = ((data1[i,:]-data2[j,:])**2*weight).sum()
        #         pass
        #     pass

        # Fast computation of distance matrix in Python+NumPy,
        # adapted from Bill Baxter's post on [numpy-discussion].
        # Basically: (x-y)**2*w = x*w*x - 2*x*w*y + y*y*w
        data1w = data1*weight
        euclidean_distance_matrix = (data1w*data1).sum(1)[:,None] \
                                    -2*N.dot(data1w,data2.T)+ \
                                    (data2*data2*weight).sum(1)
        # correction to some possible numerical instabilities:
        euclidean_distance_matrix[euclidean_distance_matrix<0] = 0
        self.euclidean_distance_matrix = euclidean_distance_matrix
        return self.euclidean_distance_matrix
    pass


class KernelSquaredExponential(Kernel):
    """The Squared Exponential kernel class.

    Note that it can handle a length scale for each dimension for
    Automtic Relevance Determination.

    """
    def __init__(self, length_scale=0.01, **kwargs):
        """Initialize a Squared Exponential kernel instance.

        :Parameters:
          length_scale : float OR numpy.ndarray
            the characteristic length-scale (or length-scales) of the
            phenomenon under investigation.
            (Defaults to 0.01)
        """
        # init base class first
        Kernel.__init__(self, **kwargs)

        self.length_scale = length_scale
        self.kernel_matrix = None


    def __repr__(self):
        return "%s(length_scale=%s)" % (self.__class__.__name__, str(self.length_scale))

    def compute(self, data1, data2=None):
        """Compute kernel matrix.

        :Parameters:
          data1 : numpy.ndarray
            data
          data2 : numpy.ndarray
            data
            (Defaults to None)
        """
        self.kernel_matrix = N.exp(-self.euclidean_distance(data1, data2, weight=(0.5/self.length_scale**2)))
        return self.kernel_matrix

    def gradient(self,data1,data2):
        """Compute gradient of the kernel matrix. A must for fast
        model selection with high-dimensional data.
        """
        # TODO SOON
        grad = None
        return grad

    def set_hyperparameters(self,*length_scale):
        """Facility to set lengthscales. Used model selection.
        """
        self.length_scale = N.array(length_scale)
        return
    pass


class KernelConstant(Kernel):
    """The constant kernel class.
    """
    def __init__(self, coefficient=None, **kwargs):
        """Initialize the constant kernel instance.

        :Parameters:
          coefficient : numpy.ndarray
            the coefficients of the linear kernel
            (Defaults to None)
        """
        # init base class first
        Kernel.__init__(self, **kwargs)

        self.coefficient = coefficient
        self.kernel_matrix = None


    def __repr__(self):
        return "%s(coefficient=%s)" % (self.__class__.__name__, str(self.coefficient))

    def compute(self, data1, data2=None):
        """Compute kernel matrix.

        :Parameters:
          data1 : numpy.ndarray
            data
          data2 : numpy.ndarray
            data
            (Defaults to None)
        """
        if data2 == None:
            data2 = data1
            pass
        self.kernel_matrix = (self.coefficient**2)*N.ones((data1.shape[0],data2.shape[0]))
        return self.kernel_matrix


    def set_hyperparameters(self,*coefficient):
        self.coefficient = N.array(coefficient)
        return
    pass


class KernelLinear(KernelConstant):
    """The linear kernel class.
    """
    def compute(self, data1, data2=None):
        """Compute kernel matrix.

        :Parameters:
          data1 : numpy.ndarray
            data
          data2 : numpy.ndarray
            data
            (Defaults to None)
        """
        if data2 == None:
            data2 = data1
            pass
        self.kernel_matrix = N.dot(data1*(self.coefficient**2),data2.T)
        return self.kernel_matrix

    pass


class KernelMatern(Kernel):
    """The Matern kernel class.
    """
    # TODO
    pass



def generate_dataset_wr1996(size=200):
    """
    Generate a known dataset ('6d robot arm',Williams and Rasmussen 1996)
    in order to test the correctness of the implementation of kernel ARD.
    For full details see:
    http://www.gaussianprocess.org/gpml/code/matlab/doc/regression.html#ard

    x_1 picked randomly in [-1.932, -0.453]
    x_2 picked randomly in [0.534, 3.142]
    r_1 = 2.0
    r_2 = 1.3
    f(x_1,x_2) = r_1 cos (x_1) + r_2 cos(x_1 + x_2) + N(0,0.0025)
    etc.

    Expected relevances:
    ell_1      1.804377
    ell_2      1.963956
    ell_3      8.884361
    ell_4     34.417657
    ell_5   1081.610451
    ell_6    375.445823
    sigma_f    2.379139
    sigma_n    0.050835
    """
    intervals = N.array([[-1.932, -0.453],[0.534, 3.142]])
    r = N.array([2.0,1.3])
    x = N.random.rand(size,2)
    x *= N.array(intervals[:,1]-intervals[:,0])
    x += N.array(intervals[:,0])
    # print x[:,0].min(), x[:,0].max()
    # print x[:,1].min(), x[:,1].max()
    y = r[0]*N.cos(x[:,0] + r[1]*N.cos(x.sum(1))) + N.random.randn(size)*N.sqrt(0.0025)
    y -= y.mean()
    x34 = x + N.random.randn(size,2)*0.02
    x56 = N.random.randn(size,2)
    x = N.hstack([x,x34,x56])
    return x,y


if __name__ == "__main__":

    N.random.seed(1)
    data = N.random.rand(4, 2)

    k = Kernel()
    print k
    edm = k.euclidean_distance(data)

    kse = KernelSquaredExponential()
    print kse
    ksem = kse.compute(data)

    data,labels = generate_dataset_wr1996()

    kl = KernelLinear(coefficient=N.ones(data.shape[1]))
    print kl

    kc = KernelConstant(coefficient=1.0)
    print kc

