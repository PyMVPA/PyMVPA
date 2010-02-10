# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Penalized logistic regression classifier."""

__docformat__ = 'restructuredtext'


import numpy as N

from mvpa.misc.exceptions import ConvergenceError
from mvpa.clfs.base import Classifier, accepts_dataset_as_samples

if __debug__:
    from mvpa.base import debug


class PLR(Classifier):
    """Penalized logistic regression `Classifier`.
    """

    def __init__(self, lm=1, criterion=1, reduced=False, maxiter=20, **kwargs):
        """
        Initialize a penalized logistic regression analysis

        Parameters
        ----------
        lm : int
          the penalty term lambda.
        criterion : int
          the criterion applied to judge convergence.
        reduced : Bool
          if not False, the rank of the data is reduced before
          performing the calculations. In that case, reduce is taken
          as the fraction of the first singular value, at which a
          dimension is not considered significant anymore. A
          reasonable criterion is reduced=0.01
        maxiter : int
          maximum number of iterations. If no convergence occurs
          after this number of iterations, an exception is raised.

        """
        # init base class first
        Classifier.__init__(self, **kwargs)

        self.__lm   = lm
        self.__criterion = criterion
        self.__reduced = reduced
        self.__maxiter = maxiter


    def __repr__(self):
        """String summary over the object
        """
        return """PLR(lm=%f, criterion=%d, reduced=%s, maxiter=%d, enable_ca=%s)""" % \
               (self.__lm, self.__criterion, self.__reduced, self.__maxiter,
                str(self.ca.enabled))


    def _train(self, data):
        """Train the classifier using `data` (`Dataset`).
        """
        # Set up the environment for fitting the data
        X = data.samples.T
        d = data.sa[self.params.targets_attr].value
        if not list(set(d)) == [0, 1]:
            raise ValueError, \
                  "Regressors for logistic regression should be [0,1]"

        if self.__reduced:
            # Data have reduced rank
            from scipy.linalg import svd

            # Compensate for reduced rank:
            # Select only the n largest eigenvectors
            U, S, V = svd(X.T)
            S /= S[0]
            V = N.matrix(V[:, :N.max(N.where(S > self.__reduced)) + 1])
            # Map Data to the subspace spanned by the eigenvectors
            X = (X.T * V).T

        nfeatures, npatterns = X.shape

        # Weighting vector
        w  = N.matrix(N.zeros( (nfeatures + 1, 1), 'd'))
        # Error for convergence criterion
        dw = N.matrix(N.ones(  (nfeatures + 1, 1), 'd'))
        # Patterns of interest in the columns
        X = N.matrix( \
                N.concatenate((X, N.ones((1, npatterns), 'd')), 0) \
                )
        p = N.matrix(N.zeros((1, npatterns), 'd'))
        # Matrix implementation of penalty term
        Lambda = self.__lm * N.identity(nfeatures + 1, 'd')
        Lambda[nfeatures, nfeatures] = 0
        # Gradient
        g = N.matrix(N.zeros((nfeatures + 1, 1), 'd'))
        # Fisher information matrix
        H = N.matrix(N.identity(nfeatures + 1, 'd'))

        # Optimize
        k = 0
        while N.sum(N.ravel(dw.A ** 2)) > self.__criterion:
            p[:, :] = self.__f(w.T * X)
            g[:, :] = X * (d - p).T - Lambda * w
            H[:, :] = X * N.diag(p.A1 * (1 - p.A1)) * X.T + Lambda
            dw[:, :] = H.I * g
            w += dw
            k += 1
            if k > self.__maxiter:
                raise ConvergenceError, \
                      "More than %d Iterations without convergence" % \
                      (self.__maxiter)

        if __debug__:
            debug("PLR", \
                  "PLR converged after %d steps. Error: %g" % \
                  (k, N.sum(N.ravel(dw.A ** 2))))

        if self.__reduced:
            # We have computed in rank reduced space ->
            # Project to original space
            self.w = V * w[:-1]
            self.offset = w[-1]
        else:
            self.w = w[:-1]
            self.offset = w[-1]


    def __f(self, y):
        """This is the logistic function f, that is used for determination of
        the vector w"""
        return 1. / (1 + N.exp(-y))


    @accepts_dataset_as_samples
    def _predict(self, data):
        """
        Predict the class labels for the provided data

        Returns a list of class labels
        """
        # make sure the data are in matrix form
        data = N.matrix(N.asarray(data))

        # get the values and then predictions
        values = N.ravel(self.__f(self.offset + data * self.w))
        predictions = values > 0.5

        # save the state if desired, relying on State._setitem_ to
        # decide if we will actually save the values
        self.ca.predictions = predictions
        self.ca.estimates = values

        return predictions

