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

from mvpa.clfs.classifier import Classifier

if __debug__:
    from mvpa.misc import debug


class RidgeReg(Classifier):
    """Ridge regression `Classifier`.
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

        # verify that they specified lambda
        self.__lm = lm


    def __repr__(self):
        """String summary over the object
        """
        if self.__lm is None:
            return """Ridge(lm=None, enabled_states=%s)""" %\
                (str(self.states.enabled))
        else:
            return """Ridge(lm=%f, enabled_states=%s)""" %\
                (self.__lm, str(self.states.enabled))


    def _train(self, data):
        """Train the classifier using `data` (`Dataset`).
        """

        # create matrices to solve with additional penalty term
        if self.__lm is None:
            # Not specified, so calculate based on .05*nfeatures
            Lambda = .05*data.nfeatures*N.eye(data.nfeatures)
        else:
            # use the provided penalty
            Lambda = self.__lm*N.eye(data.nfeatures)
        a = N.concatenate((data.samples,Lambda))
        b = N.concatenate((data.labels,N.zeros(data.nfeatures)))

        # perform the least sq regression
        res = N.linalg.lstsq(a,b)

        # save the weights
        self.w = res[0]

    def _predict(self, data):
        """
        Predict the output for the provided data.
        """
        # predict using the trained weights
        predictions = N.dot(data,self.w)
        
        # save the state if desired, relying on State._setitem_ to
        # decide if we will actually save the values
        self.predictions = predictions

        return predictions

