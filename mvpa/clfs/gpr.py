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
    from mvpa.base import debug


class GPR(Classifier):
    """Gaussian Process Regression (GPR).

    """

    predicted_variances = StateVariable(enabled=False,
        doc="Variance per each predicted value")

    log_marginal_likelihood = StateVariable(enabled=False,
        doc="Log Marginal Likelihood")


    _clf_internals = [ 'gpr', 'regression', 'non-linear' ]

    def __init__(self, kernel=KernelSquaredExponential(),
                 sigma_noise=0.001, **kwargs):
        """Initialize a GPR regression analysis.

        :Parameters:
          kernel : Kernel
            a kernel object defining the covariance between instances.
            (Defaults to KernelSquaredExponential())
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

        self.predicted_variances = None
        self.log_marginal_likelihood = None
        self.train_fv = None
        self.labels = None
        self.km_train_train = None
        pass

    def __repr__(self):
        """String summary of the object
        """
        return """GPR(kernel=%s, sigma_noise=%f, enable_states=%s)""" % \
               (self.__kernel, self.sigma_noise, str(self.states.enabled))


    def compute_log_marginal_likelihood(self):
        """
        Compute log marginal likelihood using self.train_fv and self.labels.
        """
        log_marginal_likelihood = -0.5*N.dot(self.train_labels, self.alpha) - \
                                  N.log(self.L.diagonal()).sum() - \
                                  self.km_train_train.shape[0]*0.5*N.log(2*N.pi)
        self.log_marginal_likelihood = log_marginal_likelihood

        return log_marginal_likelihood


    def _train(self, data):
        """Train the classifier using `data` (`Dataset`).
        """

        self.train_fv = data.samples
        self.train_labels = data.labels

        if __debug__:
            debug("GPR", "Computing train train kernel matrix")

        self.km_train_train = self.__kernel.compute(self.train_fv)

        # note scipy.cho_factor and scipy.cho_solve seems to be more appropriate
        # but preliminary tests show them to be slower and less stable.

        self.L = N.linalg.cholesky(self.km_train_train +
              self.sigma_noise**2*N.identity(self.km_train_train.shape[0], 'd'))
        self.alpha = N.linalg.solve(self.L.transpose(),
                                    N.linalg.solve(self.L, self.train_labels))

        # compute only if the state is enabled
        if self.states.isEnabled('log_marginal_likelihood'):
            self.compute_log_marginal_likelihood()

        pass


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
            v = N.linalg.solve(self.L, km_train_test)
            self.predicted_variances = \
                           N.diag(km_test_test-N.dot(v.transpose(), v)) + self.sigma_noise**2

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
            self.__kernel.set_hyperparameters(*args[1:])
            pass
        return

    pass


def compute_prediction(sigma_noise_best,length_scale_best,regression,dataset,data_test,label_test,F,logml=True):
    data_train = dataset.samples
    label_train = dataset.labels
    import pylab
    kse = KernelSquaredExponential(length_scale=length_scale_best)
    g = GPR(kse, sigma_noise=sigma_noise_best,regression=regression)
    print g
    if regression:
        g.states.enable("predicted_variances")
        pass

    if logml:
        g.states.enable("log_marginal_likelihood")
        pass

    g.train(dataset)
    prediction = g.predict(data_test)

    # print label_test
    # print prediction
    accuracy = None
    if regression:
        accuracy = N.sqrt(((prediction-label_test)**2).sum()/prediction.size)
        print "RMSE:",accuracy
    else:
        accuracy = (prediction.astype('l')==label_test.astype('l')).sum()/float(prediction.size)
        print "accuracy:", accuracy
        pass

    if F == 1:
        pylab.figure()
        pylab.plot(data_train, label_train, "ro", label="train")
        pylab.plot(data_test, prediction, "b-", label="prediction")
        pylab.plot(data_test, label_test, "g+", label="test")
        if regression:
            pylab.plot(data_test, prediction-N.sqrt(g.predicted_variances), "b--", label=None)
            pylab.plot(data_test, prediction+N.sqrt(g.predicted_variances), "b--", label=None)
            pylab.text(0.5, -0.8, "RMSE="+"%f" %(accuracy))
        else:
            pylab.text(0.5, -0.8, "accuracy="+str(accuracy))
            pass
        pylab.legend()
        pass

    print "LML:",g.log_marginal_likelihood




if __name__ == "__main__":
    import pylab
    pylab.close("all")
    pylab.ion()

    from mvpa.datasets import Dataset
    from mvpa.misc.data_generators import sinModulated

    train_size = 40
    test_size = 100
    F = 1

    dataset = sinModulated(train_size, F)
    # print dataset.labels

    dataset_test = sinModulated(test_size, F, flat=True)
    # print dataset_test.labels

    regression = True
    logml = True

    if logml :
        print "Looking for better hyperparameters: grid search"

        sigma_noise_steps = N.linspace(0.1, 0.5, num=20)
        length_scale_steps = N.linspace(0.05, 0.6, num=20)
        lml = N.zeros((len(sigma_noise_steps), len(length_scale_steps)))
        lml_best = -N.inf
        length_scale_best = 0.0
        sigma_noise_best = 0.0
        i = 0
        for x in sigma_noise_steps:
            j = 0
            for y in length_scale_steps:
                kse = KernelSquaredExponential(length_scale=y)
                g = GPR(kse, sigma_noise=x,regression=regression)
                g.states.enable("log_marginal_likelihood")
                g.train(dataset)
                lml[i,j] = g.log_marginal_likelihood
                # print x,y,g.log_marginal_likelihood
                # g.train_fv = dataset.samples
                # g.train_labels = dataset.labels
                # lml[i,j] = g.compute_log_marginal_likelihood()
                if lml[i,j] > lml_best:
                    lml_best = lml[i,j]
                    length_scale_best = y
                    sigma_noise_best = x
                    # print x,y,lml_best
                    pass
                j += 1
                pass
            i += 1
            pass
        pylab.figure()
        X = N.repeat(sigma_noise_steps[:,N.newaxis],sigma_noise_steps.size,axis=1)
        Y = N.repeat(length_scale_steps[N.newaxis,:],length_scale_steps.size,axis=0)
        step = (lml.max()-lml.min())/30
        pylab.contour(X,Y, lml, N.arange(lml.min(), lml.max()+step, step),colors='k')
        pylab.plot([sigma_noise_best],[length_scale_best],"k+",markeredgewidth=2, markersize=8)
        pylab.xlabel("noise standard deviation")
        pylab.ylabel("characteristic length_scale")
        pylab.title("log marginal likelihood")
        pylab.axis("tight")
        print "lml_best",lml_best
        print "sigma_noise_best",sigma_noise_best
        print "length_scale_best",length_scale_best
        print "number of expected upcrossing on the unitary intervale:",1.0/(2*N.pi*length_scale_best)
        pass



    compute_prediction(sigma_noise_best,length_scale_best,regression,dataset,
                       dataset_test.samples, dataset_test.labels,F,logml)
    pylab.show()
