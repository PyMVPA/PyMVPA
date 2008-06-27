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
        # In the following you can find faster implementations of this
        # basic code:
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
    def __init__(self, sigma_0=1.0, **kwargs):
        """Initialize the constant kernel instance.

        :Parameters:
          sigma_0 : float
            standard deviation of the Gaussian prior probability
            N(0,sigma_0**2) of the intercept of the constant regression.
            (Defaults to 1.0)
        """
        # init base class first
        Kernel.__init__(self, **kwargs)

        self.sigma_0 = sigma_0
        self.kernel_matrix = None

    def __repr__(self):
        return "%s(sigma_0=%s)" % (self.__class__.__name__, str(self.sigma_0))

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
        self.kernel_matrix = (self.sigma_0**2)*N.ones((data1.shape[0],data2.shape[0]))
        return self.kernel_matrix

    def set_hyperparameters(self,*sigma_0):
        self.sigma_0 = N.array(sigma_0)
        return

    pass


class KernelLinear(KernelConstant):
    """The linear kernel class.
    """
    def __init__(self, Sigma_p=None, sigma_0=1.0, **kwargs):
        """Initialize the linear kernel instance.

        :Parameters:
          Sigma_p : numpy.ndarray
            Covariance matrix of the Gaussian prior probability N(0,Sigma_p)
            on the weights of the linear regression.
            (Defaults to None)
          sigma_0 : float
            the standard deviation of the Gaussian prior N(0,sigma_0**2)
            of the intercept of the linear regression.
            (Deafults to 1.0)
        """
        # init base class first
        Kernel.__init__(self, **kwargs)

        self.Sigma_p = Sigma_p
        self.sigma_0 = sigma_0
        self.kernel_matrix = None

    def __repr__(self):
        return "%s(Sigma_p=%s, sigma_0=%s)" % (self.__class__.__name__, str(self.Sigma_p), str(self.sigma_0))

    def compute(self, data1, data2=None):
        """Compute kernel matrix.
        Set Sigma_p to correct dimensions and default value if necessary.

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

        # if Sigma_p is not set use Identity matrix instead:
        if self.Sigma_p == None:
            self.Sigma_p = N.eye(data1.shape[1])
        elif N.isscalar(self.Sigma_p): # if scalar use Identitiy matrix times scalar
            self.Sigma_p =  N.diagflat(N.ones(data.shape[1]*self.Sigma_p))
        elif len(self.Sigma_p.shape)==1 and self.Sigma_p.shape[1]==data1.shape[1]: # if vector use it as diagonal matrix
            self.Sigma_p == N.diagflat(self.Sigma_p)
            pass
        # XXX if Sigma_p is changed a warning should be issued!
        # XXX other cases of incorrect Sigma_p could be catched
        
        self.kernel_matrix = N.dot(data1, N.dot(self.Sigma_p,data2.T)) \
                             +self.sigma_0**2
        return self.kernel_matrix

    pass


class KernelMatern(Kernel):
    """The Matern kernel class.
    """
    # TODO
    def __init__(self):
        raise NotImplementedError
    pass


# dictionary of avalable kernels with names as keys:
kernel_dictionary = {'squared exponential':KernelSquaredExponential,
                     'constant':KernelConstant,
                     'linear':KernelLinear}

if __name__ == "__main__":

    from mvpa.misc import data_generators

    # N.random.seed(1)
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

    kl = KernelLinear(Sigma_p=N.eye(data.shape[1]))
    print kl

    kc = KernelConstant(sigma_0=1.0)
    print kc

    # In the following we draw some 2D functions at random from the
    # distribution N(O,kernel) defined by each available kernel and
    # plot them. These plots shows the flexibility of a given kernel
    # (with default parameters) when doing interpolation. The choice
    # of a kernel defines a prior probability over the function space
    # used for regression/classfication with GPR/GPC.
    import pylab
    pylab.ion()
    count = 1
    for k in kernel_dictionary.keys():
        pylab.subplot(3,4,count)
        pylab.ioff()
        # X = N.random.rand(size)*12.0-6.0
        # X.sort()
        X = N.arange(-1,1,.02)
        X = X[:,N.newaxis]
        ker = kernel_dictionary[k]()
        K = ker.compute(X,X)
        for i in range(10):
            f = N.random.multivariate_normal(N.zeros(X.shape[0]),K)
            pylab.plot(X[:,0],f,"b-")
            pass
        pylab.ion()
        pylab.title(k)
        pylab.axis('tight')
        count += 1
        pass

    
    
