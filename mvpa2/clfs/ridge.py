# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Ridge regression classifier."""

__docformat__ = 'restructuredtext'


import numpy as np
from mvpa2.base import externals

if externals.exists("scipy", raise_=True):
    from scipy.linalg import lstsq

from mvpa2.clfs.base import Classifier, accepts_dataset_as_samples

class RidgeReg(Classifier):
    """Ridge regression `Classifier`.

    This ridge regression adds an intercept term so your labels do not
    have to be zero-centered.
    """

    __tags__ = ['ridge', 'regression', 'linear']

    def __init__(self, lm=None, **kwargs):
        """
        Initialize a ridge regression analysis.

        Parameters
        ----------
        lm : float
          the penalty term lambda.
          (Defaults to .05*nFeatures)
        """
        # init base class first
        Classifier.__init__(self, **kwargs)

        # pylint happiness
        self.w = None

        # It does not make sense to calculate a confusion matrix for a
        # ridge regression
        self.ca.enable('training_stats', False)

        # verify that they specified lambda
        self.__lm = lm

        # store train method config
        self.__implementation = 'direct'


    def __repr__(self):
        """String summary of the object
        """
        if self.__lm is None:
            return """Ridge(lm=.05*nfeatures, enable_ca=%s)""" % \
                (str(self.ca.enabled))
        else:
            return """Ridge(lm=%f, enable_ca=%s)""" % \
                (self.__lm, str(self.ca.enabled))


    def _train(self, data):
        """Train the classifier using `data` (`Dataset`).
        """

        if self.__implementation == "direct":
            # create matrices to solve with additional penalty term
            # determine the lambda matrix
            if self.__lm is None:
                # Not specified, so calculate based on .05*nfeatures
                Lambda = .05*data.nfeatures*np.eye(data.nfeatures)
            else:
                # use the provided penalty
                Lambda = self.__lm*np.eye(data.nfeatures)

            # add the penalty term
            a = np.concatenate( \
                (np.concatenate((data.samples, np.ones((data.nsamples, 1))), 1),
                    np.concatenate((Lambda, np.zeros((data.nfeatures, 1))), 1)))
            b = np.concatenate((data.sa[self.get_space()].value,
                               np.zeros(data.nfeatures)))

            # perform the least sq regression and save the weights
            self.w = lstsq(a, b)[0]
        else:
            raise ValueError, "Unknown implementation '%s'" \
                              % self.__implementation


    @accepts_dataset_as_samples
    def _predict(self, data):
        """
        Predict the output for the provided data.
        """
        # predict using the trained weights
        pred = np.dot(np.concatenate((data, np.ones((len(data), 1))), 1),
                     self.w)
        # estimates equal predictions in this case
        self.ca.estimates = pred
        return pred

