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


class InvalidHyperparameter(Exception):
    """Generic exception to be raised when setting improper values
    as hyperparameters."""
    pass


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

        size1, F = data1.shape[0:2]
        size2 = data2.shape[0]
        if weight is None:
            weight = N.ones(F,'d') # unitary weight
            pass

        # In the following you can find faster implementations of this
        # basic code:
        #
        # euclidean_distance_matrix = N.zeros((data1.shape[0], data2.shape[0]),
        #                                    'd')
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
        if data2 is None:
            data2 = data1
            pass
        self.kernel_matrix = (self.sigma_0**2)*N.ones((data1.shape[0],data2.shape[0]))
        return self.kernel_matrix

    def set_hyperparameters(self, hyperparameter):
        if hyperparameter<0:
            raise InvalidHyperparameter()
        self.sigma_0 = hyperparameter
        return

    pass


class KernelLinear(Kernel):
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
        if data2 is None:
            data2 = data1
            pass

        # if Sigma_p is not set use a scalar 1.0
        if self.Sigma_p is None:
            self.Sigma_p = 1.0

        # it is better to use separate lines of computation, to don't
        # incure computation cost without need (otherwise
        # N.dot(self.Sigma_p, data2.T) can take forever for relatively
        # large number of features)

        #if scalar - scale second term appropriately
        if N.isscalar(self.Sigma_p):
            if self.Sigma_p == 1.0:
                data2_sc = data2.T
            else:
                data2_sc = self.Sigma_p * data2.T

        # if vector use it as diagonal matrix -- ie scale each row by
        # the given value
        elif len(self.Sigma_p.shape) == 1 and \
                 self.Sigma_p.shape[0] == data1.shape[1]:
            # which due to numpy broadcasting is the same as product
            # with scalar above
            data2_sc = (self.Sigma_p * data1).T

        # if it is a full matrix -- full-featured and lengthy
        # matrix product
        else:
            data2_sc = N.dot(self.Sigma_p, data2.T)
            pass

        # XXX if Sigma_p is changed a warning should be issued!
        # XXX other cases of incorrect Sigma_p could be catched
        self.kernel_matrix = N.dot(data1, data2_sc) + self.sigma_0**2
        return self.kernel_matrix

    def set_hyperparameters(self, hyperparameter):
        # XXX in the next line we assume that the values we want to
        # assign to Sigma_p are a constant or a vector (the diagonal
        # of Sigma_p actually). This is a limitation since these
        # values could be in general an hermitian matrix (i.e., a
        # covariance matrix)... but how to tell ModelSelector/OpenOpt
        # to proved just "hermitian" set of values? So for now we skip
        # the general case, which seems not to useful indeed.
        if N.any(hyperparameter<0):
            raise InvalidHyperparameter()
        self.sigma_0 = N.array(hyperparameter[0])
        self.Sigma_p = N.diagflat(hyperparameter[1:])
        return

    pass


class KernelExponential(Kernel):
    """The Exponential kernel class.

    Note that it can handle a length scale for each dimension for
    Automtic Relevance Determination.

    """
    def __init__(self, length_scale=1.0, sigma_f = 1.0, **kwargs):
        """Initialize an Exponential kernel instance.

        :Parameters:
          length_scale : float OR numpy.ndarray
            the characteristic length-scale (or length-scales) of the
            phenomenon under investigation.
            (Defaults to 1.0)
          sigma_f : float
            Signal standard deviation.
            (Defaults to 1.0)
        """
        # init base class first
        Kernel.__init__(self, **kwargs)

        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.kernel_matrix = None


    def __repr__(self):
        return "%s(length_scale=%s, sigma_f=%s)" % (self.__class__.__name__, str(self.length_scale), str(self.sigma_f))

    def compute(self, data1, data2=None):
        """Compute kernel matrix.

        :Parameters:
          data1 : numpy.ndarray
            data
          data2 : numpy.ndarray
            data
            (Defaults to None)
        """
        # XXX the following computation can be (maybe) made more
        # efficient since length_scale is squared and then
        # square-rooted uselessly.
        self.kernel_matrix = self.sigma_f*N.exp(-N.sqrt(self.euclidean_distance(data1, data2, weight=0.5/(self.length_scale**2))))
        return self.kernel_matrix

    def gradient(self,data1,data2):
        """Compute gradient of the kernel matrix. A must for fast
        model selection with high-dimensional data.
        """
        # TODO SOON
        # grad = ...
        # return grad
        raise NotImplementedError

    def set_hyperparameters(self, hyperparameter):
        """Set hyperaparmeters from a vector.

        Used by model selection.
        """
        if N.any(hyperparameter<0):
            raise InvalidHyperparameter()
        self.sigma_f = hyperparameter[0]
        self.length_scale = hyperparameter[1:]
        return

    pass


class KernelSquaredExponential(Kernel):
    """The Squared Exponential kernel class.

    Note that it can handle a length scale for each dimension for
    Automtic Relevance Determination.

    """
    def __init__(self, length_scale=1.0, sigma_f=1.0, **kwargs):
        """Initialize a Squared Exponential kernel instance.

        :Parameters:
          length_scale : float OR numpy.ndarray
            the characteristic length-scale (or length-scales) of the
            phenomenon under investigation.
            (Defaults to 1.0)
          sigma_f : float
            Signal standard deviation.
            (Defaults to 1.0)
        """
        # init base class first
        Kernel.__init__(self, **kwargs)

        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.kernel_matrix = None


    def __repr__(self):
        return "%s(length_scale=%s, sigma_f=%s)" % (self.__class__.__name__, str(self.length_scale), str(self.sigma_f))

    def compute(self, data1, data2=None):
        """Compute kernel matrix.

        :Parameters:
          data1 : numpy.ndarray
            data
          data2 : numpy.ndarray
            data
            (Defaults to None)
        """
        self.kernel_matrix = self.sigma_f*N.exp(-self.euclidean_distance(data1, data2, weight=0.5/(self.length_scale**2)))
        return self.kernel_matrix

    def gradient(self,data1,data2):
        """Compute gradient of the kernel matrix. A must for fast
        model selection with high-dimensional data.
        """
        # TODO SOON
        # grad = ...
        # return grad
        raise NotImplementedError

    def set_hyperparameters(self, hyperparameter):
        """Set hyperaparmeters from a vector.

        Used by model selection.
        """
        if N.any(hyperparameter<0):
            raise InvalidHyperparameter()
        self.sigma_f = hyperparameter[0]
        self.length_scale = hyperparameter[1:]
        return

    pass


class KernelMatern_3_2(Kernel):
    """The Matern kernel class for the case ni=3/2 or ni=5/2.

    Note that it can handle a length scale for each dimension for
    Automtic Relevance Determination.

    """
    def __init__(self, length_scale=1.0, sigma_f=1.0, numerator=3.0, **kwargs):
        """Initialize a Squared Exponential kernel instance.

        :Parameters:
          length_scale : float OR numpy.ndarray
            the characteristic length-scale (or length-scales) of the
            phenomenon under investigation.
            (Defaults to 1.0)
          sigma_f : float
            Signal standard deviation.
            (Defaults to 1.0)
          numerator: float
            the numerator of parameter ni of Matern covariance functions.
            Currently only numerator=3.0 and numerator=5.0 are implemented.
            (Defaults to 3.0)
        """
        # init base class first
        Kernel.__init__(self, **kwargs)

        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.kernel_matrix = None
        if numerator==3.0 or numerator==5.0:
            self.numerator = numerator
        else:
            raise NotImplementedError

    def __repr__(self):
        return "%s(length_scale=%s, ni=%d/2)" % (self.__class__.__name__, str(self.length_scale), self.numerator)

    def compute(self, data1, data2=None):
        """Compute kernel matrix.

        :Parameters:
          data1 : numpy.ndarray
            data
          data2 : numpy.ndarray
            data
            (Defaults to None)
        """
        tmp = self.euclidean_distance(data1, data2, weight=0.5/(self.length_scale**2))
        if self.numerator == 3.0:
            tmp = N.sqrt(tmp)
            self.kernel_matrix = self.sigma_f*(1.0+N.sqrt(3.0)*tmp)*N.exp(-N.sqrt(3.0)*tmp)
        elif self.numerator == 5.0:
            tmp2 = N.sqrt(tmp)
            self.kernel_matrix = self.sigma_f*(1.0+N.sqrt(5.0)*tmp2+5.0/3.0*tmp)*N.exp(-N.sqrt(5.0)*tmp2)
        return self.kernel_matrix

    def gradient(self,data1,data2):
        """Compute gradient of the kernel matrix. A must for fast
        model selection with high-dimensional data.
        """
        # TODO SOON
        # grad = ...
        # return grad
        raise NotImplementedError

    def set_hyperparameters(self, hyperparameter):
        """Set hyperaparmeters from a vector.

        Used by model selection.
        Note: 'numerator' is not considered as an hyperparameter.
        """
        if N.any(hyperparameter<0):
            raise InvalidHyperparameter()
        self.sigma_f = hyperparameter[0]
        self.length_scale = hyperparameter[1:]
        return

    pass


class KernelMatern_5_2(KernelMatern_3_2):
    """The Matern kernel class for the case ni=5/2.

    This kernel is just KernelMatern_3_2(numerator=5.0).
    """
    def __init__(self, **kwargs):
        """Initialize a Squared Exponential kernel instance.

        :Parameters:
          length_scale : float OR numpy.ndarray
            the characteristic length-scale (or length-scales) of the
            phenomenon under investigation.
            (Defaults to 1.0)
        """
        KernelMatern_3_2.__init__(self, numerator=5.0, **kwargs)
        pass


class KernelRationalQuadratic(Kernel):
    """The Rational Quadratic (RQ) kernel class.

    Note that it can handle a length scale for each dimension for
    Automtic Relevance Determination.

    """
    def __init__(self, length_scale=1.0, sigma_f=1.0, alpha=0.5, **kwargs):
        """Initialize a Squared Exponential kernel instance.

        :Parameters:
          length_scale : float OR numpy.ndarray
            the characteristic length-scale (or length-scales) of the
            phenomenon under investigation.
            (Defaults to 1.0)
          sigma_f : float
            Signal standard deviation.
            (Defaults to 1.0)
          alpha: float
            The parameter of the RQ functions family.
            (Defaults to 2.0)
        """
        # init base class first
        Kernel.__init__(self, **kwargs)

        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.kernel_matrix = None
        self.alpha = alpha

    def __repr__(self):
        return "%s(length_scale=%s, alpha=%f)" % (self.__class__.__name__, str(self.length_scale), self.alpha)

    def compute(self, data1, data2=None):
        """Compute kernel matrix.

        :Parameters:
          data1 : numpy.ndarray
            data
          data2 : numpy.ndarray
            data
            (Defaults to None)
        """
        tmp = self.euclidean_distance(data1, data2, weight=0.5/(self.length_scale**2))
        self.kernel_matrix = self.sigma_f*(1.0+tmp/(2.0*self.alpha))**-self.alpha
        return self.kernel_matrix

    def gradient(self,data1,data2):
        """Compute gradient of the kernel matrix. A must for fast
        model selection with high-dimensional data.
        """
        # TODO SOON
        # grad = ...
        # return grad
        raise NotImplementedError

    def set_hyperparameters(self, hyperparameter):
        """Set hyperaparmeters from a vector.

        Used by model selection.
        Note: 'alpha' is not considered as an hyperparameter.
        """
        if N.any(hyperparameter<0):
            raise InvalidHyperparameter()
        self.sigma_f = hyperparameter[0]
        self.length_scale = hyperparameter[1:]
        return

    pass


# dictionary of avalable kernels with names as keys:
kernel_dictionary = {'constant':KernelConstant,
                     'linear':KernelLinear,
                     'exponential':KernelExponential,
                     'squared exponential':KernelSquaredExponential,
                     'Matern ni=3/2':KernelMatern_3_2,
                     'Matern ni=5/2':KernelMatern_5_2,
                     'rational quadratic':KernelRationalQuadratic}
