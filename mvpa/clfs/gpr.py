#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   Copyright (c) 2008 Emanuele Olivetti <emanuele@relativita.com>
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Gaussian Process Regression (GPR)."""

__docformat__ = 'restructuredtext'


import numpy as N

from mvpa.misc.state import StateVariable
from mvpa.clfs.base import Classifier
from mvpa.clfs.kernel import KernelSquaredExponential

if __debug__:
    from mvpa.misc import debug

class GPR(Classifier):
    """Gaussian Process Regression (GPR).

    """

    predicted_variances = StateVariable(enabled=False,
        doc="Variance per each predicted value")

    log_marginal_likelihood = StateVariable(enabled=False,
        doc="Log Marginal Likelihood ;-)")


    _clf_internals = [ 'gpr', 'regression', 'non-linear' ]

    def __init__(self, kernel=KernelSquaredExponential(length_scale=0.01),
                 sigma_noise=0.001, **kwargs):
        """Initialize a GPR regression analysis.

        :Parameters:
          kernel : Kernel
            a kernel object defining the covariance between instances.
            (Default to KernelSquaredExponential(lengthscale=0.01)
          sigma_noise : float
            the standard deviation of the gaussian noise.
            (Defaults to 0.001)

        """
        # init base class first
        Classifier.__init__(self, **kwargs)

        # pylint happiness
        self.w = None

        # It does not make sense to calculate a confusion matrix for a GPR
        self.states.enable('training_confusion', False)

        # set kernel:
        self.__kernel = kernel

        # set noise level:
        self.sigma_noise = sigma_noise


    def __repr__(self):
        """String summary of the object
        """
        return """GPR(kernel=%s, enable_states=%s)""" % \
               (self.__kernel, str(self.states.enabled))


    def _train(self, data):
        """Train the classifier using `data` (`Dataset`).
        """

        self.train_fv = data.samples
        self.train_labels = data.labels
        if __debug__:
            debug("GPR", "Computing train train kernel matrix")
        self.km_train_train = self.__kernel.compute(self.train_fv)

        self.L = N.linalg.cholesky(self.km_train_train +
              self.sigma_noise**2*N.identity(self.km_train_train.shape[0], 'd'))
        self.alpha = N.linalg.solve(self.L.transpose(),
                                    N.linalg.solve(self.L, self.train_labels))


    def _predict(self, data):
        """
        Predict the output for the provided data.
        """
        if __debug__:
            debug('GPR', "Computing train test kernel matrix")
        km_train_test = self.__kernel.compute(self.train_fv, data)
        if __debug__:
            debug('GPR', "Computing test test kernel matrix")
        km_test_test = self.__kernel.compute(data)

        predictions = N.dot(km_train_test.transpose(), self.alpha)

        if self.states.isEnabled('predicted_variances'):
            # do computation only if state variable was enabled
            v = N.linalg.solve(L, km_train_test)
            self.predicted_variances = \
                           N.diag(km_test_test-N.dot(v.transpose(), v))

        if self.states.isEnabled('log_marginal_likelihood'):
            log_marginal_likelihood = -0.5*N.dot(train_label, alpha) - \
                                      N.log(L.diagonal()).sum() - \
                                      km_train_train.shape[0]*0.5*N.log(2*N.pi)

        return predictions

def gen_data(n_instances, n_features, flat=False):
    data = None
    if flat:
        data = (N.arange(0.0, 1.0, 1.0/n_instances)*N.pi)
        data.resize(n_instances, n_features)
        # print data
    else:
        data = N.random.rand(n_instances, n_features)*N.pi
        pass
    label = N.sin((data**2).sum(1)).round()
    data = N.matrix(data)
    return data, label


if __name__ == "__main__":


    N.random.seed(1)

    from mvpa.datasets import Dataset

    train_size = 15
    test_size = 100
    F = 1


    data_train, label_train = gen_data(train_size, F)
    print label_train

    data_test, label_test = gen_data(test_size, F, flat=True)
    # print label_test

    dataset = Dataset(samples=data_train, labels=label_train)

    kse = KernelSquaredExponential(length_scale=2e-1)
    g = GPR(kse, sigma_noise=0.001)
    print g

    g.train(dataset)
    prediction = g.predict(data_test)

    print label_test
    print prediction.round()
    # accuracy = 1-N.sqrt(((prediction-label_test)**2).sum()/prediction.size) # 1-RMSE
    accuracy = (prediction.round().astype('l')==label_test.astype('l')).sum()/float(prediction.size)
    print "accuracy:", accuracy

    import pylab
    pylab.close("all")
    pylab.ion()

    if F == 1:
        pylab.plot(data_train, label_train, "ro", label="train")
        pylab.plot(data_test, prediction, "b+-", label="prediction")
        pylab.plot(data_test, label_test, "gx-", label="test")
        pylab.text(0.5, -1.2, "accuracy="+str(accuracy))
        pylab.legend()
        pass

