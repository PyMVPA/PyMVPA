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

from mvpa.misc.exceptions import InvalidHyperparameterError
from mvpa.clfs.distance import squared_euclidean_distance

if __debug__:
    from mvpa.base import debug, warning


class Kernel(object):
    """Kernel function base class.

    """

    def __init__(self):
        pass

    def __repr__(self):
        return "Kernel()"

    def compute(self, data1, data2=None):
        raise NotImplementedError

    def compute_gradient(self,alphaalphaTK):
        raise NotImplementedError

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
        self.kernel_matrix = \
            (self.sigma_0 ** 2) * N.ones((data1.shape[0], data2.shape[0]))
        return self.kernel_matrix

    def set_hyperparameters(self, hyperparameter):
        if hyperparameter < 0:
            raise InvalidHyperparameterError()
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
        return "%s(Sigma_p=%s, sigma_0=%s)" \
            % (self.__class__.__name__, str(self.Sigma_p), str(self.sigma_0))

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
        self.kernel_matrix = N.dot(data1, data2_sc) + self.sigma_0 ** 2
        return self.kernel_matrix

    def set_hyperparameters(self, hyperparameter):
        # XXX in the next line we assume that the values we want to
        # assign to Sigma_p are a constant or a vector (the diagonal
        # of Sigma_p actually). This is a limitation since these
        # values could be in general an hermitian matrix (i.e., a
        # covariance matrix)... but how to tell ModelSelector/OpenOpt
        # to proved just "hermitian" set of values? So for now we skip
        # the general case, which seems not to useful indeed.
        if N.any(hyperparameter < 0):
            raise InvalidHyperparameterError()
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
        return "%s(length_scale=%s, sigma_f=%s)" \
          % (self.__class__.__name__, str(self.length_scale), str(self.sigma_f))

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
        self.kernel_matrix = \
            self.sigma_f * N.exp(-N.sqrt(
                squared_euclidean_distance(
                    data1, data2, weight=0.5 / (self.length_scale ** 2))))
        return self.kernel_matrix

    def gradient(self, data1, data2):
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
        if N.any(hyperparameter < 0):
            raise InvalidHyperparameterError()
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
        return "%s(length_scale=%s, sigma_f=%s)" \
          % (self.__class__.__name__, str(self.length_scale), str(self.sigma_f))

    def compute(self, data1, data2=None):
        """Compute kernel matrix.

        :Parameters:
          data1 : numpy.ndarray
            data
          data2 : numpy.ndarray
            data
            (Defaults to None)
        """
        # weighted squared euclidean distance matrix:
        self.wdm2 = squared_euclidean_distance(data1, data2, weight=(self.length_scale**-2))
        self.kernel_matrix = self.sigma_f**2 * N.exp(-0.5*self.wdm2)
        # XXX EO: old implementation:
        # self.kernel_matrix = \
        #     self.sigma_f * N.exp(-squared_euclidean_distance(
        #         data1, data2, weight=0.5 / (self.length_scale ** 2)))
        return self.kernel_matrix

    def gradient(self, data1, data2):
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
        if N.any(hyperparameter < 0):
            raise InvalidHyperparameterError()
        self.sigma_f = hyperparameter[0]
        self.length_scale = hyperparameter[1:]
        return

    def compute_lml_gradient(self,alpha,Kinv,y):
        """Compute grandient of the kernel and return the portion of
        log marginal likelihood gradient due to the kernel.
        """
        grad_sigma_f = 2.0/self.sigma_f*self.kernel_matrix
        grad_l = self.wdm2*self.kernel_matrix*(1.0/self.length_scale)
        # self.lml_gradient = N.array([0.5*N.trace(N.dot(alphaalphaT_Kinv,grad_sigma_f)),0.5*N.trace(N.dot(alphaalphaT_Kinv,grad_l))])
        # Faster formula: N.trace(N.dot(A,B)) = (A*(B.T)).sum()
        self.lml_gradient = N.array([0.5*(N.dot(y,N.dot(Kinv,N.dot(grad_log,alpha[:,None]))) - (Kinv*grad_log.T).sum()) for grad_log in [grad_sigma_f, grad_l]])
        return self.lml_gradient

    def compute_lml_gradient_compact(self,alphaalphaT_Kinv):
        """Compute grandient of the kernel and return the portion of
        log marginal likelihood gradient due to the kernel.
        Shorter formula.
        """
        grad_sigma_f = 2.0/self.sigma_f*self.kernel_matrix
        grad_l = self.wdm2*self.kernel_matrix*(1.0/self.length_scale)
        # self.lml_gradient = N.array([0.5*N.trace(N.dot(alphaalphaT_Kinv,grad_sigma_f)),0.5*N.trace(N.dot(alphaalphaT_Kinv,grad_l))])
        # Faster formula: N.trace(N.dot(A,B)) = (A*(B.T)).sum()
        self.lml_gradient = 0.5*N.array([(alphaalphaT_Kinv*(grad_sigma_f).T).sum(),(alphaalphaT_Kinv*(grad_l.T)).sum()])
        return self.lml_gradient

    def compute_lml_gradient_logscale(self,alpha,Kinv,y):
        """Compute grandient of the kernel and return the portion of
        log marginal likelihood gradient due to the kernel.
        Hyperparameters are in log scale which is sometimes more
        stable.
        """
        grad_log_sigma_f = 2.0*self.kernel_matrix
        grad_log_l = self.wdm2*self.kernel_matrix
        # self.lml_gradient = N.array([0.5*(N.dot(y,N.dot(Kinv,N.dot(grad_log,alpha[:,None]))) - N.trace(N.dot(Kinv,grad_log))) for grad_log in [grad_log_sigma_f, grad_log_l]])
        # Faster formula: N.trace(N.dot(A,B)) = (A*(B.T)).sum()
        self.lml_gradient = N.array([0.5*(N.dot(y,N.dot(Kinv,N.dot(grad_log,alpha[:,None]))) - (Kinv*grad_log.T).sum()) for grad_log in [grad_log_sigma_f, grad_log_l]])
        return self.lml_gradient

    def compute_lml_gradient_logscale_compact(self,alphaalphaT_Kinv):
        """Compute grandient of the kernel and return the portion of
        log marginal likelihood gradient due to the kernel.
        Hyperparameters are in log scale which is sometimes more
        stable. Shorter formula.
        """
        grad_log_sigma_f = 2.0*self.kernel_matrix
        grad_log_l = self.wdm2*self.kernel_matrix
        # self.lml_gradient = N.array([0.5*N.trace(N.dot(alphaalphaT_Kinv,grad_log_sigma_f)),0.5*N.trace(N.dot(alphaalphaT_Kinv,grad_log_l))])
        # Faster formula: N.trace(N.dot(A,B)) = (A*(B.T)).sum()
        self.lml_gradient = 0.5*N.array([(alphaalphaT_Kinv*(grad_log_sigma_f).T).sum(),(alphaalphaT_Kinv*(grad_log_l.T)).sum()])
        return self.lml_gradient
        
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
        if numerator == 3.0 or numerator == 5.0:
            self.numerator = numerator
        else:
            raise NotImplementedError

    def __repr__(self):
        return "%s(length_scale=%s, ni=%d/2)" \
            % (self.__class__.__name__, str(self.length_scale), self.numerator)

    def compute(self, data1, data2=None):
        """Compute kernel matrix.

        :Parameters:
          data1 : numpy.ndarray
            data
          data2 : numpy.ndarray
            data
            (Defaults to None)
        """
        tmp = squared_euclidean_distance(
                data1, data2, weight=0.5 / (self.length_scale ** 2))
        if self.numerator == 3.0:
            tmp = N.sqrt(tmp)
            self.kernel_matrix = \
                self.sigma_f * (1.0 + N.sqrt(3.0) * tmp) \
                * N.exp(-N.sqrt(3.0) * tmp)
        elif self.numerator == 5.0:
            tmp2 = N.sqrt(tmp)
            self.kernel_matrix = \
                self.sigma_f * (1.0 + N.sqrt(5.0) * tmp2 + 5.0 / 3.0 * tmp) \
                * N.exp(-N.sqrt(5.0) * tmp2)
        return self.kernel_matrix

    def gradient(self, data1, data2):
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
        if N.any(hyperparameter < 0):
            raise InvalidHyperparameterError()
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
        return "%s(length_scale=%s, alpha=%f)" \
            % (self.__class__.__name__, str(self.length_scale), self.alpha)

    def compute(self, data1, data2=None):
        """Compute kernel matrix.

        :Parameters:
          data1 : numpy.ndarray
            data
          data2 : numpy.ndarray
            data
            (Defaults to None)
        """
        tmp = squared_euclidean_distance(
                data1, data2, weight=0.5 / (self.length_scale ** 2))
        self.kernel_matrix = \
            self.sigma_f * (1.0 + tmp / (2.0 * self.alpha)) ** -self.alpha
        return self.kernel_matrix

    def gradient(self, data1, data2):
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
        if N.any(hyperparameter < 0):
            raise InvalidHyperparameterError()
        self.sigma_f = hyperparameter[0]
        self.length_scale = hyperparameter[1:]
        return

    pass


# dictionary of avalable kernels with names as keys:
kernel_dictionary = {'constant': KernelConstant,
                     'linear': KernelLinear,
                     'exponential': KernelExponential,
                     'squared exponential': KernelSquaredExponential,
                     'Matern ni=3/2': KernelMatern_3_2,
                     'Matern ni=5/2': KernelMatern_5_2,
                     'rational quadratic': KernelRationalQuadratic}
