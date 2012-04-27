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


import numpy as np

from mvpa2.misc.exceptions import ConvergenceError
from mvpa2.base.learner import FailedToTrainError
from mvpa2.clfs.base import Classifier, accepts_dataset_as_samples

if __debug__:
    from mvpa2.base import debug


class PLR(Classifier):
    """Penalized logistic regression `Classifier`.
    """

    __tags__ = [ 'plr', 'binary', 'linear', 'has_sensitivity' ]

    def __init__(self, lm=1, criterion=1, reduced=0.0, maxiter=20, **kwargs):
        """
        Initialize a penalized logistic regression analysis

        Parameters
        ----------
        lm : int
          the penalty term lambda.
        criterion : int
          the criterion applied to judge convergence.
        reduced : float
          if not 0, the rank of the data is reduced before
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
        d = self._attrmap.to_numeric(data.sa[self.get_space()].value)
        if set(d) != set([0, 1]):
            raise ValueError, \
                  "Regressors for logistic regression should be [0,1]. Got %s" \
                  %(set(d),)

        if self.__reduced != 0 :
            # Data have reduced rank
            from scipy.linalg import svd

            # Compensate for reduced rank:
            # Select only the n largest eigenvectors
            U, S, V = svd(X.T)
            if S[0] == 0:
                raise FailedToTrainError(
                    "Data provided to PLR seems to be degenerate -- "
                    "0-th singular value is 0")
            S /= S[0]
            V = np.matrix(V[:, :np.max(np.where(S > self.__reduced)) + 1])
            # Map Data to the subspace spanned by the eigenvectors
            X = (X.T * V).T

        nfeatures, npatterns = X.shape

        # Weighting vector
        w  = np.matrix(np.zeros( (nfeatures + 1, 1), 'd'))
        # Error for convergence criterion
        dw = np.matrix(np.ones(  (nfeatures + 1, 1), 'd'))
        # Patterns of interest in the columns
        X = np.matrix( \
                np.concatenate((X, np.ones((1, npatterns), 'd')), 0) \
                )
        p = np.matrix(np.zeros((1, npatterns), 'd'))
        # Matrix implementation of penalty term
        Lambda = self.__lm * np.identity(nfeatures + 1, 'd')
        Lambda[nfeatures, nfeatures] = 0
        # Gradient
        g = np.matrix(np.zeros((nfeatures + 1, 1), 'd'))
        # Fisher information matrix
        H = np.matrix(np.identity(nfeatures + 1, 'd'))

        # Optimize
        k = 0
        while np.sum(np.ravel(dw.A ** 2)) > self.__criterion:
            p[:, :] = self.__f(w.T * X)
            g[:, :] = X * (d - p).T - Lambda * w
            H[:, :] = X * np.diag(p.A1 * (1 - p.A1)) * X.T + Lambda
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
                  (k, np.sum(np.ravel(dw.A ** 2))))

        if self.__reduced:
            # We have computed in rank reduced space ->
            # Project to original space
            self.w = V * w[:-1]
            self.bias = w[-1]
        else:
            self.w = w[:-1]
            self.bias = w[-1]


    def __f(self, y):
        """This is the logistic function f, that is used for determination of
        the vector w"""
        return 1. / (1 + np.exp(-y))


    @accepts_dataset_as_samples
    def _predict(self, data):
        """
        Predict the class labels for the provided data

        Returns a list of class labels
        """
        # make sure the data are in matrix form
        data = np.matrix(np.asarray(data))

        # get the values and then predictions
        values = np.ravel(self.__f(self.bias + data * self.w))
        predictions = (values > 0.5).astype(int)

        # save the state if desired, relying on State._setitem_ to
        # decide if we will actually save the values
        self.ca.predictions = predictions
        self.ca.estimates = values

        return predictions

    def get_sensitivity_analyzer(self, **kwargs):
        """Returns a sensitivity analyzer for PLR."""
        return PLRWeights(self, **kwargs)



from mvpa2.base.state import ConditionalAttribute
from mvpa2.base.types import asobjarray
from mvpa2.measures.base import Sensitivity
from mvpa2.datasets.base import Dataset


class PLRWeights(Sensitivity):
    """`Sensitivity` reporting linear weights of PLR"""

    _LEGAL_CLFS = [ PLR ]

    def _call(self, dataset=None):
        """Extract weights from PLR classifier.

        PLR always has weights available, so nothing has to be computed here.
        """
        clf = self.clf
        attrmap = clf._attrmap

        if attrmap:
            # labels (values of the corresponding space) which were used
            # for mapping Here we rely on the fact that they are sorted
            # originally (just an arange())
            labels_num = attrmap.values()
            labels = attrmap.to_literal(asobjarray([tuple(sorted(labels_num))]),
                                        recurse=True)
        else:
            labels = [(0, 1)]           # we just had our good old numeric ones

        ds = Dataset(clf.w.T, sa={clf.get_space(): labels,
                                  'biases' : [clf.bias]})
        return ds
