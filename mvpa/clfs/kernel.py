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
    from mvpa.base import debug

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
    def __init__(self, length_scale=1.0, **kwargs):
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
        self.kernel_matrix = N.exp(-self.euclidean_distance(data1, data2, weight=0.5/(self.length_scale**2)))
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
        # self.kernel_matrix = (self.coefficient**2)*N.ones((data1.shape[0],data2.shape[0]))
        self.kernel_matrix = N.dot(N.dot(data1,(self.coefficient**2)*N.eye(data1.shape[1])),data2.T)
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
    def __init__(self):
        raise NotImplementedError
    pass


if __name__ == "__main__":

    from mvpa.misc import data_generators

    N.random.seed(1)
    data = N.random.rand(4, 2)

    k = Kernel()
    print k
    edm = k.euclidean_distance(data)

    kse = KernelSquaredExponential()
    print kse
    ksem = kse.compute(data)

    dataset = data_generators.wr1996()
    print dataset
    data = dataset.samples
    labels = dataset.labels

    kl = KernelLinear(coefficient=N.ones(data.shape[1]))
    print kl

    kc = KernelConstant(coefficient=1.0)
    print kc

