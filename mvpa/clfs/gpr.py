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
from mvpa.clfs.kernel import KernelSquaredExponential, KernelLinear

if __debug__:
    from mvpa.base import debug


class GPR(Classifier):
    """Gaussian Process Regression (GPR).

    """

    predicted_variances = StateVariable(enabled=False,
        doc="Variance per each predicted value")

    log_marginal_likelihood = StateVariable(enabled=False,
        doc="Log Marginal Likelihood")

    linear_weights = StateVariable(enabled=False,
        doc="Weights of the linear regression (for KernelLinear)")

    linear_weights_variances = StateVariable(enabled=False,
        doc="Variances of the linear weights (for KernelLinear)")

    _clf_internals = [ 'gpr', 'regression' ]

    # TODO: don't initialize kernel with an instance here since it gets shared
    #       among all instances of GPR
    def __init__(self, kernel=None, sigma_noise=0.001, **kwargs):
        """Initialize a GPR regression analysis.

        :Parameters:
          kernel : Kernel
            a kernel object defining the covariance between instances.
            (Defaults to None)
          sigma_noise : float
            the standard deviation of the gaussian noise.
            (Defaults to 0.001)

        """
        # init base class first
        Classifier.__init__(self, **kwargs)

        # append proper clf_internal depending on the kernel
        # TODO: unify finally all kernel-based machines.
        #       make SMLR to use kernels
        self._clf_internals.append(
            ( 'non-linear', 'linear' )[int(isinstance(kernel, KernelLinear))])

        # pylint happiness
        self.w = None

        # It does not make sense to calculate a confusion matrix for a GPR
        self.states.enable('training_confusion', False)

        # set kernel:
        if kernel == None:
            self.__kernel = KernelSquaredExponential()
        else:
            self.__kernel = kernel
            pass
            
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
        if __debug__: debug("GPR", "Computing log_marginal_likelihood")
        self.log_marginal_likelihood = -0.5*N.dot(self.train_labels, self.alpha) - \
                                  N.log(self.L.diagonal()).sum() - \
                                  self.km_train_train.shape[0]*0.5*N.log(2*N.pi)
        return self.log_marginal_likelihood


    def compute_log_marginal_likelihood_gradient(self):
        """Compute gradient of the log marginal likelihood.
        """
        raise NotImplementedError
    

    def compute_linear_weights(self):
        """In case of KernelLinear compute explicitly the coefficients
        of the linear regression, together with their variances (if
        requested).

        Note that the intercept is not computed.
        """
        # XXX The following two lines does not work since
        # self.__kernel is instance of kernel.KernelLinear and not
        # just KernelLinear. How to fix?
        # if not isinstance(self.__kernel, KernelLinear):
        #    raise NotImplementedError
        self.linear_weights = N.dot(self.__kernel.Sigma_p,
                                    N.dot(self.train_fv.T, self.alpha))
        if self.states.isEnabled('linear_weights_variances'):
            # super ugly formulas that can be quite surely improved:
            tmp = N.linalg.inv(self.L)
            Kyinv = N.dot(tmp.T,tmp)
            self.linear_weights_variances = N.diag(self.__kernel.Sigma_p - N.dot(self.__kernel.Sigma_p, N.dot(self.train_fv.T,N.dot(Kyinv, N.dot(self.train_fv, self.__kernel.Sigma_p)))))
        return self.linear_weights


    def _train(self, data):
        """Train the classifier using `data` (`Dataset`).
        """

        self.train_fv = data.samples
        self.train_labels = data.labels

        if __debug__:
            debug("GPR", "Computing train train kernel matrix")
        self.km_train_train = self.__kernel.compute(self.train_fv)

        if __debug__:
            debug("GPR", "Computing L. sigma_noise=%g" % self.sigma_noise)

        tmp = self.km_train_train + self.sigma_noise**2*N.identity(self.km_train_train.shape[0],'d')
        # The following line could raise N.linalg.linalg.LinAlgError
        # because of numerical reasons due to the too rapid decay of
        # 'tmp' eigenvalues. In that case we try adding a small
        # constant to tmp, e.g. epsilon=1.0e-20. It should be a form
        # of Tikhonov regularization. This is equivalent to adding
        # little white gaussian noise.
        try:
            self.L = N.linalg.cholesky(tmp)
        except N.linalg.linalg.LinAlgError:
            epsilon = 1.0e-20
            self.L = N.linalg.cholesky(tmp+epsilon)
            pass
        # Note: scipy.cho_factor and scipy.cho_solve seems to be more
        # appropriate to perform Cholesky decomposition and the
        # 'solve' step of the following lines, but preliminary tests
        # show them to be slower and less stable than NumPy's way.

        if __debug__:
            debug("GPR", "Computing alpha")
        self.alpha = N.linalg.solve(self.L.transpose(),
                                    N.linalg.solve(self.L, self.train_labels))

        # compute only if the state is enabled
        if self.states.isEnabled('log_marginal_likelihood'):
            self.compute_log_marginal_likelihood()
            pass

        if self.states.isEnabled('linear_weights'):
            self.compute_linear_weights()
            pass
            
        if __debug__:
            debug("GPR", "Done training")

        pass


    def _predict(self, data):
        """
        Predict the output for the provided data.
        """
        if __debug__:
            debug('GPR', "Computing train test kernel matrix")
        km_train_test = self.__kernel.compute(self.train_fv, data)

        predictions = N.dot(km_train_test.transpose(), self.alpha)

        if self.states.isEnabled('predicted_variances'):
            # do computation only if state variable was enabled
            if __debug__:
                debug('GPR', "Computing test test kernel matrix")
            km_test_test = self.__kernel.compute(data)

            if __debug__:
                debug("GPR", "Computing predicted variances")
            v = N.linalg.solve(self.L, km_train_test)
            self.predicted_variances = \
                           N.diag(km_test_test-N.dot(v.transpose(), v)) + self.sigma_noise**2

        if __debug__:
            debug("GPR", "Done predicting")
        return predictions


    def set_hyperparameters(self,hyperparameter):
        """
        Set hyperparameters' values.

        Note that 'hyperparameter' is a sequence so the order of its
        values is important. First value must be sigma_noise, then
        other kernel's hyperparameters values follow in the exact
        order the kernel expect them to be.
        """
        self.sigma_noise = hyperparameter[0]
        if hyperparameter.size>1:
            self.__kernel.set_hyperparameters(hyperparameter[1:])
            pass
        return

    kernel = property(fget=lambda self:self.__kernel)
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
