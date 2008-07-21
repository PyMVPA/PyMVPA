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
from mvpa.misc.param import Parameter
from mvpa.clfs.kernel import KernelSquaredExponential, KernelLinear
from mvpa.measures.base import Sensitivity
from mvpa.misc.exceptions import InvalidHyperparameterError
from scipy.linalg import cho_factor as SLcho_factor
from scipy.linalg import cho_solve as SLcho_solve
from scipy.linalg import cholesky as SLcholesky
import scipy.linalg as SL

if __debug__:
    from mvpa.base import debug

# Some local bindings for bits of speed up
Nlog = N.log
Ndot = N.dot
Ndiag = N.diag
NLAcholesky = N.linalg.cholesky
NLAsolve = N.linalg.solve
NLAError = N.linalg.linalg.LinAlgError

# Some precomputed items. log is relatively expensive
_halflog2pi = 0.5 * Nlog(2*N.pi)


class GPR(Classifier):
    """Gaussian Process Regression (GPR).

    """

    predicted_variances = StateVariable(enabled=False,
        doc="Variance per each predicted value")

    log_marginal_likelihood = StateVariable(enabled=False,
        doc="Log Marginal Likelihood")

    _clf_internals = [ 'gpr', 'regression', 'retrainable' ]


    # NOTE XXX Parameters of the classifier. Values available as
    # clf.parameter or clf.params.parameter, or as
    # clf.params['parameter'] (as the full Parameter object)
    #
    # __doc__ and __repr__ for class is conviniently adjusted to
    # reflect values of those params

    # Kernel machines/classifiers should be refactored also to behave
    # the same and define kernel parameter appropriately... TODO, but SVMs
    # already kinda do it nicely ;-)

    sigma_noise = Parameter(0.001, allowedtype='float', min=1e-10,
        doc="the standard deviation of the gaussian noise.")

    # XXX For now I don't introduce kernel parameter since yet to unify
    # kernel machines
    #kernel = Parameter(None, allowedtype='Kernel',
    #    doc="Kernel object defining the covariance between instances. "
    #        "(Defaults to KernelSquaredExponential if None in arguments)")

    def __init__(self, kernel=None, **kwargs):
        """Initialize a GPR regression analysis.

        :Parameters:
          kernel : Kernel
            a kernel object defining the covariance between instances.
            (Defaults to KernelSquaredExponential if None in arguments)
        """
        # init base class first
        Classifier.__init__(self, **kwargs)

        # It does not make sense to calculate a confusion matrix for a GPR
        # XXX it does ;) it will be a RegressionStatistics actually ;-)
        # So if someone desires -- let him have it
        # self.states.enable('training_confusion', False)

        # set kernel:
        if kernel is None:
            kernel = KernelSquaredExponential()
        self.__kernel = kernel

        # append proper clf_internal depending on the kernel
        # TODO: unify finally all kernel-based machines.
        #       make SMLR to use kernels
        self._clf_internals = self._clf_internals + \
            (['non-linear'],
             ['linear', 'has_sensitivity'])[int(isinstance(kernel, KernelLinear))]

        # No need to initialize state variables. Unless they got set
        # they would raise an exception self.predicted_variances =
        # None self.log_marginal_likelihood = None
        self._train_fv = None
        self._labels = None
        self._km_train_train = None
        pass


    def __repr__(self):
        """String summary of the object
        """
        return super(GPR, self).__repr__(
            prefixes=['kernel=%s' % self.__kernel])


    def compute_log_marginal_likelihood(self):
        """
        Compute log marginal likelihood using self.train_fv and self.labels.
        """
        if __debug__: debug("GPR", "Computing log_marginal_likelihood")
        self.log_marginal_likelihood = \
                                 -0.5*Ndot(self._train_labels, self._alpha) - \
                                  Nlog(self._L.diagonal()).sum() - \
                                  self._km_train_train.shape[0] * _halflog2pi
        return self.log_marginal_likelihood


    def compute_gradient_log_marginal_likelihood(self):
        """Compute gradient of the log marginal likelihood.
        """
        # XXX EO: check whether the precomputed self.alpha self.Kinv
        # are actually the ones corresponding to the hyperparameters
        # used to compute this gradient!
        # YYY EO: currently this is verified outside gpr.py but it is
        # not an efficient solution.
        # XXX EO: Do some memoizing since it could happen that some
        # hyperparameters are kept constant by user request, so we
        # don't need (somtimes) to recompute the corresponding
        # gradient again.

        # self.Kinv = N.linalg.inv(self._C)
        # Faster:
        self.Kinv = SLcho_solve(self._LL,N.eye(self._L.shape[0]))

        y = self._train_labels
        # Pass tmp to __kernel and let it compute its gradient terms,
        # since this scales up to huge number of hyperparameters:
        grad_LML_hypers = self.__kernel.compute_lml_gradient(self._alpha,self.Kinv,y)
        grad_K_sigma_n = 2.0*self.sigma_noise*N.eye(self.Kinv.shape[0])
        # Add the term related to sigma_noise:
        grad_LML_sigma_n = 0.5*(N.dot(y,N.dot(self.Kinv,N.dot(grad_K_sigma_n,self._alpha[:,None]))) - N.trace(N.dot(self.Kinv,grad_K_sigma_n)))
        # Faster formula: tr(AB) = (A*B.T).sum()
        # grad_LML_sigma_n = 0.5*(N.dot(y,N.dot(self.Kinv,N.dot(grad_K_sigma_n,self._alpha[:,None]))) - (self.Kinv*(grad_K_sigma_n).T).sum())
        self.lml_gradient = N.hstack([grad_LML_sigma_n, grad_LML_hypers[:,0]])
        return self.lml_gradient

    def compute_gradient_log_marginal_likelihood2(self):
        """Compute gradient of the log marginal likelihood.
        """
        # XXX check whether the precomputed self.alpha self.Kinv are
        # actually the ones corresponding to the hyperparameters used
        # to compute this gradient!
        
        # Do some memoizing since it could happen that some
        # hyperparameters are kept constant by user request, so we
        # don't need (somtimes) to recompute the corresponding
        # gradient again.

        # self.Kinv = N.linalg.inv(self._C)
        # Faster:
        self.Kinv = SLcho_solve(self._LL,N.eye(self._L.shape[0]))

        alphalphaT = N.dot(self._alpha[:,None],self._alpha[None,:])
        tmp = alphalphaT - self.Kinv
        # Pass tmp to __kernel and let it compute its gradient terms.
        # This scales up to huge number of hyperparameters:
        grad_LML_hypers = self.__kernel.compute_lml_gradient2(tmp)
        grad_K_sigma_n = 2.0*self.sigma_noise*N.eye(tmp.shape[0])
        # Add the term related to sigma_noise:
        # grad_K_sigma_n = 0.5 * N.trace(N.dot(tmp,grad_K_sigma_n))
        # Faster formula: tr(AB) = (A*B.T).sum()
        grad_LML_sigma_n = 0.5 * (tmp * (grad_K_sigma_n).T).sum()
        self.lml_gradient = N.hstack([grad_LML_sigma_n, grad_LML_hypers])
        return self.lml_gradient

    def compute_gradient_log_marginal_likelihood_logscale(self):
        # self.Kinv = N.linalg.inv(self._C)
        # Faster:
        self.Kinv = SLcho_solve(self._LL,N.eye(self._L.shape[0]))
        y = self._train_labels
        grad_LML_log_hypers = self.__kernel.compute_lml_gradient_logscale(self._alpha,self.Kinv,y)
        grad_K_log_sigma_n = 2.0*self.sigma_noise**2*N.eye(self.Kinv.shape[0])
        # grad_LML_sigma_n = 0.5*(N.dot(y,N.dot(self.Kinv,N.dot(grad_K_log_sigma_n,self._alpha[:,None]))) - N.trace(N.dot(self.Kinv,grad_K_log_sigma_n)))
        # Using faster formula: tr(AB) = (A*B.T).sum()
        grad_LML_log_sigma_n = 0.5*(N.dot(y,N.dot(self.Kinv,N.dot(grad_K_log_sigma_n,self._alpha[:,None]))) - (self.Kinv*(grad_K_log_sigma_n).T).sum())
        self.LML_gradient = N.hstack([grad_LML_log_sigma_n, grad_LML_log_hypers[:,0]])
        return self.LML_gradient
    

    def getSensitivityAnalyzer(self, **kwargs):
        """Returns a sensitivity analyzer for GPR.

        """
        # XXX The following two lines does not work since
        # self.__kernel is instance of kernel.KernelLinear and not
        # just KernelLinear. How to fix?
        # YYY yoh is not sure what is the problem... KernelLinear is actually
        #     kernel.KernelLinear so everything shoudl be ok
        if not isinstance(self.__kernel, KernelLinear):
            raise NotImplementedError

        return GPRLinearWeights(self, **kwargs)


    def _train(self, data):
        """Train the classifier using `data` (`Dataset`).
        """

        # local bindings for faster lookup
        retrainable = self.params.retrainable
        if retrainable:
            newkernel = False
            newL = False
            _changedData = self._changedData

        self._train_fv = train_fv = data.samples
        self._train_labels = train_labels = data.labels

        if not retrainable or _changedData['traindata'] or _changedData.get('kernel_params', False):
            if __debug__:
                debug("GPR", "Computing train train kernel matrix")
            self._km_train_train = km_train_train = self.__kernel.compute(train_fv)
            newkernel = True
            if retrainable:
                self._km_train_test = None # reset to facilitate recomputation
        else:
            if __debug__:
                debug("GPR", "Not recomputing kernel since retrainable and nothing changed")
            km_train_train = self._km_train_train # reuse

        if not retrainable or newkernel or _changedData['params']:
            if __debug__:
                debug("GPR", "Computing L. sigma_noise=%g" % self.sigma_noise)
            self._C = km_train_train + \
                  self.sigma_noise**2*N.identity(km_train_train.shape[0], 'd')
            # The following decomposition could raise
            # N.linalg.linalg.LinAlgError because of numerical
            # reasons, due to the too rapid decay of 'self._C'
            # eigenvalues. In that case we try adding a small constant
            # to self._C, e.g. epsilon=1.0e-20. It should be a form of
            # Tikhonov regularization. This is equivalent to adding
            # little white gaussian noise to data.
            #
            # XXX EO: how to choose epsilon?
            #
            # Cholesky decomposition is provided by three different
            # NumPy/SciPy routines (fastest first):
            # 1) self._LL = scipy.linalg.cho_factor(self._C, lower=True)
            #    self._L = L = N.tril(self._LL[0])
            # 2) self._L = scipy.linalg.cholesky(self._C, lower=True)
            # 3) self._L = numpy.linalg.cholesky(self._C)
            # Even though 1 is the fastest we choose 2 since 1 does
            # not return a clean lower-triangular matrix (see docstring).
            try:
                self._L = SLcholesky(self._C, lower=True)
                self._LL = (self._L,True)
            except NLAError:
                epsilon = 1.0e-20*N.eye(tmp.shape[0])
                self._L = SLcholesky(self._C+epsilon, lower=True)
                self._LL = (self._L,True)
                pass
            newL = True
        else:
            if __debug__:
                debug("GPR", "Not computing L since kernel, data and params stayed the same")
            L = self._L                 # reuse

        # XXX we leave _alpha being recomputed, although we could check
        #   if newL or _changedData['labels']
        #
        if __debug__:
            debug("GPR", "Computing alpha")
        # self._alpha = NLAsolve(L.transpose(),
        #                              NLAsolve(L, train_labels))
        # Faster:
        self._alpha = SLcho_solve(self._LL, train_labels)

        # compute only if the state is enabled
        if self.states.isEnabled('log_marginal_likelihood'):
            self.compute_log_marginal_likelihood()
            pass

        if retrainable:
            # we must assign it only if it is retrainable
            self.states.retrained = not newkernel or not newL

        if __debug__:
            debug("GPR", "Done training")

        pass


    def _predict(self, data):
        """
        Predict the output for the provided data.
        """
        retrainable = self.params.retrainable

        if not retrainable or self._changedData['testdata'] or self._km_train_test is None:
            if __debug__:
                debug('GPR', "Computing train test kernel matrix")
            km_train_test = self.__kernel.compute(self._train_fv, data)
            if retrainable:
                self._km_train_test = km_train_test
                self.states.repredicted = False
        else:
            if __debug__:
                debug('GPR', "Not recomputing train test kernel matrix")
            km_train_test = self._km_train_test
            self.states.repredicted = True


        predictions = Ndot(km_train_test.transpose(), self._alpha)

        if self.states.isEnabled('predicted_variances'):
            # do computation only if state variable was enabled
            if not retrainable or self._km_test_test is None \
                   or self._changedData['testdata']:
                if __debug__:
                    debug('GPR', "Computing test test kernel matrix")
                km_test_test = self.__kernel.compute(data)
                if retrainable:
                    self._km_test_test = km_test_test
            else:
                if __debug__:
                    debug('GPR', "Not recomputing test test kernel matrix")
                km_test_test = self._km_test_test

            if __debug__:
                debug("GPR", "Computing predicted variances")
            L = self._L
            # v = NLAsolve(L, km_train_test)
            # Faster:
            piv = N.arange(L.shape[0])
            v = SL.lu_solve((L.T,piv), km_train_test, trans=1)
            # self.predicted_variances = \
            #     Ndiag(km_test_test - Ndot(v.T, v)) \
            #     + self.sigma_noise**2
            # Faster formula: N.diag(Ndot(v.T, v)) = (v**2).sum(0):
            self.predicted_variances = Ndiag(km_test_test) - (v**2).sum(0) \
                                       + self.sigma_noise**2
            pass

        if __debug__:
            debug("GPR", "Done predicting")
        return predictions


    def _setRetrainable(self, value, force=False):
        """Internal function : need to set _km_test_test
        """
        super(GPR, self)._setRetrainable(value, force)
        if force or (value and value != self.params.retrainable):
            self._km_test_test = None


    def set_hyperparameters(self, hyperparameter):
        """
        Set hyperparameters' values.

        Note that 'hyperparameter' is a sequence so the order of its
        values is important. First value must be sigma_noise, then
        other kernel's hyperparameters values follow in the exact
        order the kernel expect them to be.
        """
        if hyperparameter[0]<GPR.sigma_noise.min:
            raise InvalidHyperparameterError()
        self.sigma_noise = hyperparameter[0]
        if hyperparameter.size > 1:
            self.__kernel.set_hyperparameters(hyperparameter[1:])
            pass
        return

    kernel = property(fget=lambda self:self.__kernel)
    pass


class GPRLinearWeights(Sensitivity):
    """`SensitivityAnalyzer` that reports the weights GPR trained
    on a given `Dataset`.

    In case of KernelLinear compute explicitly the coefficients
    of the linear regression, together with their variances (if
    requested).

    Note that the intercept is not computed.
    """

    variances = StateVariable(enabled=False,
        doc="Variances of the weights (for KernelLinear)")

    _LEGAL_CLFS = [ GPR ]


    def _call(self, dataset):
        """Extract weights from GPR
        """

        clf = self.clf
        kernel = clf.kernel
        train_fv = clf._train_fv

        weights = Ndot(kernel.Sigma_p,
                        Ndot(train_fv.T, clf._alpha))

        if self.states.isEnabled('variances'):
            # super ugly formulas that can be quite surely improved:
            tmp = N.linalg.inv(self._L)
            Kyinv = Ndot(tmp.T, tmp)
            # XXX in such lengthy matrix manipulations you might better off
            #     using N.matrix where * is a matrix product
            self.states.variances = Ndiag(
                kernel.Sigma_p -
                Ndot(kernel.Sigma_p,
                      Ndot(train_fv.T,
                            Ndot(Kyinv,
                                  Ndot(train_fv, kernel.Sigma_p)))))
        return weights


def compute_prediction(sigma_noise_best, sigma_f, length_scale_best, regression,
                       dataset, data_test, label_test, F, logml=True):
    """XXX Function which seems to be valid only for __main__...

    TODO: remove reimporting of pylab etc. See pylint output for more
          information
    """

    data_train = dataset.samples
    label_train = dataset.labels
    import pylab
    kse = KernelSquaredExponential(length_scale=length_scale_best, sigma_f=sigma_f)
    g = GPR(kse, sigma_noise=sigma_noise_best, regression=regression)
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
        print "RMSE:", accuracy
    else:
        accuracy = (prediction.astype('l')==label_test.astype('l')).sum() \
                   / float(prediction.size)
        print "accuracy:", accuracy
        pass

    if F == 1:
        pylab.figure()
        pylab.plot(data_train, label_train, "ro", label="train")
        pylab.plot(data_test, prediction, "b-", label="prediction")
        pylab.plot(data_test, label_test, "g+", label="test")
        if regression:
            pylab.plot(data_test, prediction-N.sqrt(g.predicted_variances),
                       "b--", label=None)
            pylab.plot(data_test, prediction+N.sqrt(g.predicted_variances),
                       "b--", label=None)
            pylab.text(0.5, -0.8, "RMSE="+"%f" %(accuracy))
        else:
            pylab.text(0.5, -0.8, "accuracy="+str(accuracy))
            pass
        pylab.legend()
        pass

    print "LML:", g.log_marginal_likelihood




if __name__ == "__main__":
    import pylab
    pylab.close("all")
    pylab.ion()

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
                g = GPR(kse, sigma_noise=x, regression=regression)
                g.states.enable("log_marginal_likelihood")
                g.train(dataset)
                lml[i, j] = g.log_marginal_likelihood
                # print x,y,g.log_marginal_likelihood
                # g.train_fv = dataset.samples
                # g.train_labels = dataset.labels
                # lml[i, j] = g.compute_log_marginal_likelihood()
                if lml[i, j] > lml_best:
                    lml_best = lml[i, j]
                    length_scale_best = y
                    sigma_noise_best = x
                    # print x,y,lml_best
                    pass
                j += 1
                pass
            i += 1
            pass
        pylab.figure()
        X = N.repeat(sigma_noise_steps[:, N.newaxis], sigma_noise_steps.size,
                     axis=1)
        Y = N.repeat(length_scale_steps[N.newaxis, :], length_scale_steps.size,
                     axis=0)
        step = (lml.max()-lml.min())/30
        pylab.contour(X, Y, lml, N.arange(lml.min(), lml.max()+step, step),
                      colors='k')
        pylab.plot([sigma_noise_best], [length_scale_best], "k+",
                   markeredgewidth=2, markersize=8)
        pylab.xlabel("noise standard deviation")
        pylab.ylabel("characteristic length_scale")
        pylab.title("log marginal likelihood")
        pylab.axis("tight")
        print "lml_best", lml_best
        print "sigma_noise_best", sigma_noise_best
        print "length_scale_best", length_scale_best
        print "number of expected upcrossing on the unitary intervale:", \
              1.0/(2*N.pi*length_scale_best)
        pass



    compute_prediction(sigma_noise_best, 1.0, length_scale_best, regression, dataset,
                       dataset_test.samples, dataset_test.labels, F, logml)
    pylab.show()
