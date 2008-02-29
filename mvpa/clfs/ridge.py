#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Ridge regression classifier."""

__docformat__ = 'restructuredtext'


import numpy as N
from scipy.linalg import lstsq

from mvpa.clfs.classifier import Classifier

if __debug__:
    from mvpa.misc import debug


class RidgeReg(Classifier):
    """Ridge regression `Classifier`.

    This ridge regression adds an intercept term so your labels do not
    have to be zero-centered.
    """

    def __init__(self, lm=None, **kwargs):
        """
        Initialize a ridge regression analysis.

        :Parameters:
          lm : float
            the penalty term lambda.  
            (Defaults to .05*nFeatures)

        """
        # init base class first
        Classifier.__init__(self, **kwargs)

        # It does not make sense to calculate a confusion matrix for a
        # ridge regression
        self.states.enable('training_confusion', False)

        # verify that they specified lambda
        self.__lm = lm


    def __repr__(self):
        """String summary of the object
        """
        if self.__lm is None:
            return """Ridge(lm=.05*nfeatures, enabled_states=%s)""" % \
                (str(self.states.enabled))
        else:
            return """Ridge(lm=%f, enabled_states=%s)""" % \
                (self.__lm, str(self.states.enabled))


    def _train(self, data):
        """Train the classifier using `data` (`Dataset`).
        """

        # create matrices to solve with additional penalty term
        # determine the lambda matrix
        if self.__lm is None:
            # Not specified, so calculate based on .05*nfeatures
            Lambda = .05*data.nfeatures*N.eye(data.nfeatures)
        else:
            # use the provided penalty
            Lambda = self.__lm*N.eye(data.nfeatures)

        # add the penalty term
        a = N.concatenate( \
            (N.concatenate((data.samples, N.ones((data.nsamples, 1))), 1),
                N.concatenate((Lambda, N.zeros((data.nfeatures, 1))), 1)))
        b = N.concatenate((data.labels, N.zeros(data.nfeatures)))

        # perform the least sq regression and save the weights
        self.w = lstsq(a, b)[0]


    def _predict(self, data):
        """
        Predict the output for the provided data.
        """
        # predict using the trained weights
        predictions = N.dot(N.concatenate((data, N.ones((len(data), 1))), 1),
                            self.w)

        # save the state if desired, relying on State._setitem_ to
        # decide if we will actually save the values
        self.predictions = predictions

        return predictions

