# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   Copyright (c) 2008 Emanuele Olivetti <emanuele@relativita.com>
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Bayesian Linear Regression (BLR)."""

__docformat__ = 'restructuredtext'


import numpy as np

from mvpa2.base.state import ConditionalAttribute
from mvpa2.clfs.base import Classifier, accepts_dataset_as_samples

if __debug__:
    from mvpa2.misc import debug


class BLR(Classifier):
    """Bayesian Linear Regression (BLR).

    """

    predicted_variances = ConditionalAttribute(enabled=False,
        doc="Variance per each predicted value")

    log_marginal_likelihood = ConditionalAttribute(enabled=False,
        doc="Log Marginal Likelihood")


    __tags__ = [ 'blr', 'regression', 'linear' ]

    def __init__(self, sigma_p = None, sigma_noise=1.0, **kwargs):
        """Initialize a BLR regression analysis.

        Parameters
        ----------
        sigma_noise : float
          the standard deviation of the gaussian noise.
          (Defaults to 0.1)

        """
        # init base class first
        Classifier.__init__(self, **kwargs)

        # pylint happiness
        self.w = None

        # It does not make sense to calculate a confusion matrix for a
        # BLR:
        self.ca.enable('training_stats', False)

        # set the prior on w: N(0,sigma_p) , specifying the covariance
        # sigma_p on w:
        self.sigma_p = sigma_p

        # set noise level:
        self.sigma_noise = sigma_noise

        self.ca.predicted_variances = None
        self.ca.log_marginal_likelihood = None
        # Yarik: what was those about??? just for future in
        #        compute_log_marginal_likelihood ?
        # self.targets = None
        pass

    def __repr__(self):
        """String summary of the object
        """
        return """BLR(w=%s, sigma_p=%s, sigma_noise=%f, enable_ca=%s)""" % \
               (self.w, self.sigma_p, self.sigma_noise, str(self.ca.enabled))


    def compute_log_marginal_likelihood(self):
        """
        Compute log marginal likelihood using self.train_fv and self.targets.
        """
        # log_marginal_likelihood = None
        # return log_marginal_likelihood
        raise NotImplementedError


    def _train(self, data):
        """Train regression using `data` (`Dataset`).
        """
        # BLR relies on numerical labels
        train_labels = self._attrmap.to_numeric(data.sa[self.get_space()].value)
        # provide a basic (i.e. identity matrix) and correct prior
        # sigma_p, if not provided before or not compliant to 'data':
        if self.sigma_p == None: # case: not provided
            self.sigma_p = np.eye(data.samples.shape[1]+1)
        elif self.sigma_p.shape[1] != (data.samples.shape[1]+1): # case: wrong dimensions
            self.sigma_p = np.eye(data.samples.shape[1]+1)
        else:
            # ...then everything is OK :)
            pass

        # add one fake column of '1.0' to model the intercept:
        self.samples_train = np.hstack([data.samples,np.ones((data.samples.shape[0],1))])
        if type(self.sigma_p)!=type(self.samples_train): # if sigma_p is a number...
            self.sigma_p = np.eye(self.samples_train.shape[1])*self.sigma_p # convert in matrix
            pass

        self.A_inv = np.linalg.inv(1.0/(self.sigma_noise**2) *
                                  np.dot(self.samples_train.T,
                                        self.samples_train) +
                                  np.linalg.inv(self.sigma_p))
        self.w = 1.0/(self.sigma_noise**2) * np.dot(self.A_inv,
                                                   np.dot(self.samples_train.T,
                                                         train_labels))
        pass


    @accepts_dataset_as_samples
    def _predict(self, data):
        """
        Predict the output for the provided data.
        """

        data = np.hstack([data,np.ones((data.shape[0],1),dtype=data.dtype)])
        predictions = np.dot(data,self.w)

        if self.ca.is_enabled('predicted_variances'):
            # do computation only if conditional attribute was enabled
            self.ca.predicted_variances = np.dot(data, np.dot(self.A_inv, data.T)).diagonal()[:,np.newaxis]
        self.ca.estimates = predictions
        return predictions


    def set_hyperparameters(self,*args):
        """
        Set hyperparameters' values.

        Note that this is a list so the order of the values is
        important.
        """
        args=args[0]
        self.sigma_noise = args[0]
        if len(args)>1:
            self.sigma_p = np.array(args[1:]) # XXX check if this is ok
            pass
        return

    pass


if __name__ == "__main__":
    import pylab
    pylab.close("all")
    pylab.ion()

    from mvpa2.misc.data_generators import linear_awgn

    train_size = 10
    test_size = 100
    F = 1 # dimensions of the dataset

    # np.random.seed(1)

    slope = np.random.rand(F)
    intercept = np.random.rand(1)
    print "True slope:",slope
    print "True intercept:",intercept

    dataset_train = linear_awgn(train_size, intercept=intercept, slope=slope)

    dataset_test = linear_awgn(test_size, intercept=intercept, slope=slope, flat=True)

    regression = True
    logml = False

    b = BLR(sigma_p=np.eye(F+1), sigma_noise=0.1)
    b.ca.enable("predicted_variances")
    b.train(dataset_train)
    predictions = b.predict(dataset_test.samples)
    print "Predicted slope and intercept:",b.w

    if F==1:
        pylab.plot(dataset_train.samples,
                   dataset_train.sa[b.get_space()].value,
                   "ro", label="train")

        pylab.plot(dataset_test.samples, predictions, "b-", label="prediction")
        pylab.plot(dataset_test.samples,
                   predictions+np.sqrt(b.ca.predicted_variances),
                   "b--", label="pred(+/-)std")
        pylab.plot(dataset_test.samples,
                   predictions-np.sqrt(b.ca.predicted_variances),
                   "b--", label=None)
        pylab.legend()
        pylab.xlabel("samples")
        pylab.ylabel("labels")
        pylab.title("Bayesian Linear Regression on dataset 'linear_AWGN'")
        pass

