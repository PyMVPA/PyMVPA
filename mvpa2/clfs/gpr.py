# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   Copyright (c) 2008 Emanuele Olivetti <emanuele@relativita.com>
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Gaussian Process Regression (GPR)."""

__docformat__ = 'restructuredtext'


import numpy as np

from mvpa2.base import externals, warning

from mvpa2.base.state import ConditionalAttribute
from mvpa2.clfs.base import Classifier, accepts_dataset_as_samples
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import EnsureFloat, EnsureNone, EnsureRange
from mvpa2.kernels.np import SquaredExponentialKernel, GeneralizedLinearKernel, \
     LinearKernel
from mvpa2.measures.base import Sensitivity
from mvpa2.misc.exceptions import InvalidHyperparameterError
from mvpa2.datasets import Dataset, dataset_wizard

if externals.exists("scipy", raise_=True):
    from scipy.linalg import cho_solve as SLcho_solve
    from scipy.linalg import cholesky as SLcholesky
    import scipy.linalg as SL
    # Some local binding for bits of speed up
    SLAError = SL.basic.LinAlgError

if __debug__:
    from mvpa2.base import debug

# Some local bindings for bits of speed up
from numpy import array, asarray
Nlog = np.log
Ndot = np.dot
Ndiag = np.diag
NLAcholesky = np.linalg.cholesky
NLAsolve = np.linalg.solve
NLAError = np.linalg.linalg.LinAlgError
eps64 = np.finfo(np.float64).eps

# Some precomputed items. log is relatively expensive
_halflog2pi = 0.5 * Nlog(2 * np.pi)

def _SLcholesky_autoreg(C, nsteps=None, **kwargs):
    """Simple wrapper around cholesky to incrementally regularize the
    matrix until successful computation.

    For `nsteps` we boost diagonal 10-fold each time from the
    'epsilon' of the respective dtype. If None -- would proceed until
    reaching 1.
    """
    if nsteps is None:
        nsteps = -int(np.floor(np.log10(np.finfo(float).eps)))
    result = None
    for step in xrange(nsteps):
        epsilon_value = (10**step) * np.finfo(C.dtype).eps
        epsilon = epsilon_value * np.eye(C.shape[0])
        try:
            result = SLcholesky(C + epsilon, lower=True)
        except SLAError, e:
            warning("Cholesky decomposition lead to failure: %s.  "
                    "As requested, performing auto-regularization but "
                    "for better control you might prefer to regularize "
                    "yourself by providing lm parameter to GPR" % e)
            if step < nsteps-1:
                if __debug__:
                    debug("GPR", "Failed to obtain cholesky on "
                          "auto-regularization step %d value %g. Got %s."
                          " Boosting lambda more to reg. C."
                          % (step, epsilon_value, e))
                continue
            else:
                raise

    if result is None:
        # no loop was done for some reason
        result = SLcholesky(C, lower=True)

    return result


class GPR(Classifier):
    """Gaussian Process Regression (GPR).

    """

    predicted_variances = ConditionalAttribute(enabled=False,
        doc="Variance per each predicted value")

    log_marginal_likelihood = ConditionalAttribute(enabled=False,
        doc="Log Marginal Likelihood")

    log_marginal_likelihood_gradient = ConditionalAttribute(enabled=False,
        doc="Log Marginal Likelihood Gradient")

    __tags__ = [ 'gpr', 'regression', 'retrainable' ]


    # NOTE XXX Parameters of the classifier. Values available as
    # clf.parameter or clf.params.parameter, or as
    # clf.params['parameter'] (as the full Parameter object)
    #
    # __doc__ and __repr__ for class is conviniently adjusted to
    # reflect values of those params

    # Kernel machines/classifiers should be refactored also to behave
    # the same and define kernel parameter appropriately... TODO, but SVMs
    # already kinda do it nicely ;-)

    sigma_noise = Parameter(0.001, 
        constraints=EnsureFloat() & EnsureRange(min=1e-10),
        doc="the standard deviation of the gaussian noise.")

    # XXX For now I don't introduce kernel parameter since yet to unify
    # kernel machines
    #kernel = Parameter(None, allowedtype='Kernel',
    #    doc="Kernel object defining the covariance between instances. "
    #        "(Defaults to KernelSquaredExponential if None in arguments)")

    lm = Parameter(None,
        constraints=((EnsureFloat() & EnsureRange(min=0.0)) | EnsureNone()),
        doc="""The regularization term lambda.
        Increase this when the kernel matrix is not positive definite. If None,
        some regularization will be provided upon necessity""")


    def __init__(self, kernel=None, **kwargs):
        """Initialize a GPR regression analysis.

        Parameters
        ----------
        kernel : Kernel
          a kernel object defining the covariance between instances.
          (Defaults to SquaredExponentialKernel if None in arguments)
        """
        # init base class first
        Classifier.__init__(self, **kwargs)

        # It does not make sense to calculate a confusion matrix for a GPR
        # XXX it does ;) it will be a RegressionStatistics actually ;-)
        # So if someone desires -- let him have it
        # self.ca.enable('training_stats', False)

        # set kernel:
        if kernel is None:
            kernel = SquaredExponentialKernel()
            debug("GPR",
                  "No kernel was provided, falling back to default: %s"
                  % kernel)
        self.__kernel = kernel

        # append proper clf_internal depending on the kernel
        # TODO: add "__tags__" to kernels since the check
        #       below does not scale
        if isinstance(kernel, (GeneralizedLinearKernel, LinearKernel)):
            self.__tags__ += ['linear']
        else:
            self.__tags__ += ['non-linear']

        if externals.exists('openopt') \
               and not 'has_sensitivity' in self.__tags__:
            self.__tags__ += ['has_sensitivity']

        # No need to initialize conditional attributes. Unless they got set
        # they would raise an exception self.predicted_variances =
        # None self.log_marginal_likelihood = None
        self._init_internals()
        pass


    def _init_internals(self):
        """Reset some internal variables to None.

        To be used in constructor and untrain()
        """
        self._train_fv = None
        self._labels = None
        self._km_train_train = None
        self._train_labels = None
        self._alpha = None
        self._L = None
        self._LL = None
        # XXX EO: useful for model selection but not working in general
        # self.__kernel.reset()
        pass


    def __repr__(self):
        """String summary of the object
        """
        return super(GPR, self).__repr__(
            prefixes=['kernel=%s' % self.__kernel])


    def compute_log_marginal_likelihood(self):
        """
        Compute log marginal likelihood using self.train_fv and self.targets.
        """
        if __debug__:
            debug("GPR", "Computing log_marginal_likelihood")
        self.ca.log_marginal_likelihood = \
                                 -0.5*Ndot(self._train_labels, self._alpha) - \
                                  Nlog(self._L.diagonal()).sum() - \
                                  self._km_train_train.shape[0] * _halflog2pi
        return self.ca.log_marginal_likelihood


    def compute_gradient_log_marginal_likelihood(self):
        """Compute gradient of the log marginal likelihood. This
        version use a more compact formula provided by Williams and
        Rasmussen book.
        """
        # XXX EO: check whether the precomputed self.alpha self.Kinv
        # are actually the ones corresponding to the hyperparameters
        # used to compute this gradient!
        # YYY EO: currently this is verified outside gpr.py but it is
        # not an efficient solution.
        # XXX EO: Do some memoizing since it could happen that some
        # hyperparameters are kept constant by user request, so we
        # don't need (somtimes) to recompute the corresponding
        # gradient again. COULD THIS BE TAKEN INTO ACCOUNT BY THE
        # NEW CACHED KERNEL INFRASTRUCTURE?

        # self.Kinv = np.linalg.inv(self._C)
        # Faster:
        Kinv = SLcho_solve(self._LL, np.eye(self._L.shape[0]))

        alphalphaT = np.dot(self._alpha[:,None], self._alpha[None,:])
        tmp = alphalphaT - Kinv
        # Pass tmp to __kernel and let it compute its gradient terms.
        # This scales up to huge number of hyperparameters:
        grad_LML_hypers = self.__kernel.compute_lml_gradient(
            tmp, self._train_fv)
        grad_K_sigma_n = 2.0*self.params.sigma_noise*np.eye(tmp.shape[0])
        # Add the term related to sigma_noise:
        # grad_LML_sigma_n = 0.5 * np.trace(np.dot(tmp,grad_K_sigma_n))
        # Faster formula: tr(AB) = (A*B.T).sum()
        grad_LML_sigma_n = 0.5 * (tmp * (grad_K_sigma_n).T).sum()
        lml_gradient = np.hstack([grad_LML_sigma_n, grad_LML_hypers])
        self.log_marginal_likelihood_gradient = lml_gradient
        return lml_gradient


    def compute_gradient_log_marginal_likelihood_logscale(self):
        """Compute gradient of the log marginal likelihood when
        hyperparameters are in logscale. This version use a more
        compact formula provided by Williams and Rasmussen book.
        """
        # Kinv = np.linalg.inv(self._C)
        # Faster:
        Kinv = SLcho_solve(self._LL, np.eye(self._L.shape[0]))
        alphalphaT = np.dot(self._alpha[:,None], self._alpha[None,:])
        tmp = alphalphaT - Kinv
        grad_LML_log_hypers = \
            self.__kernel.compute_lml_gradient_logscale(tmp, self._train_fv)
        grad_K_log_sigma_n = 2.0 * self.params.sigma_noise ** 2 * np.eye(Kinv.shape[0])
        # Add the term related to sigma_noise:
        # grad_LML_log_sigma_n = 0.5 * np.trace(np.dot(tmp, grad_K_log_sigma_n))
        # Faster formula: tr(AB) = (A * B.T).sum()
        grad_LML_log_sigma_n = 0.5 * (tmp * (grad_K_log_sigma_n).T).sum()
        lml_gradient = np.hstack([grad_LML_log_sigma_n, grad_LML_log_hypers])
        self.log_marginal_likelihood_gradient = lml_gradient
        return lml_gradient


    ##REF: Name was automagically refactored
    def get_sensitivity_analyzer(self, flavor='auto', **kwargs):
        """Returns a sensitivity analyzer for GPR.

        Parameters
        ----------
        flavor : str
          What sensitivity to provide. Valid values are
          'linear', 'model_select', 'auto'.
          In case of 'auto' selects 'linear' for linear kernel
          and 'model_select' for the rest. 'linear' corresponds to
          GPRLinearWeights and 'model_select' to GRPWeights
        """
        # XXX The following two lines does not work since
        # self.__kernel is instance of LinearKernel and not
        # just LinearKernel. How to fix?
        # YYY yoh is not sure what is the problem... LinearKernel is actually
        #     kernel.LinearKernel so everything shoudl be ok
        if flavor == 'auto':
            flavor = ('model_select', 'linear')\
                     [int(isinstance(self.__kernel, GeneralizedLinearKernel)
                          or
                          isinstance(self.__kernel, LinearKernel))]
            if __debug__:
                debug("GPR", "Returning '%s' sensitivity analyzer" % flavor)

        # Return proper sensitivity
        if flavor == 'linear':
            return GPRLinearWeights(self, **kwargs)
        elif flavor == 'model_select':
            # sanity check
            if not ('has_sensitivity' in self.__tags__):
                raise ValueError, \
                      "model_select flavor is not available probably " \
                      "due to not available 'openopt' module"
            return GPRWeights(self, **kwargs)
        else:
            raise ValueError, "Flavor %s is not recognized" % flavor


    def _train(self, data):
        """Train the classifier using `data` (`Dataset`).
        """

        # local bindings for faster lookup
        params = self.params
        retrainable = params.retrainable
        if retrainable:
            newkernel = False
            newL = False
            _changedData = self._changedData

        self._train_fv = train_fv = data.samples
        # GRP relies on numerical labels
        # yoh: yeah -- GPR now is purely regression so no conversion
        #      is necessary
        train_labels = data.sa[self.get_space()].value
        self._train_labels = train_labels

        if not retrainable or _changedData['traindata'] \
               or _changedData.get('kernel_params', False):
            if __debug__:
                debug("GPR", "Computing train train kernel matrix")
            self.__kernel.compute(train_fv)
            self._km_train_train = km_train_train = asarray(self.__kernel)
            newkernel = True
            if retrainable:
                self._km_train_test = None # reset to facilitate recomputation
        else:
            if __debug__:
                debug("GPR", "Not recomputing kernel since retrainable and "
                      "nothing has changed")
            km_train_train = self._km_train_train # reuse

        if not retrainable or newkernel or _changedData['params']:
            if __debug__:
                debug("GPR", "Computing L. sigma_noise=%g" \
                             % params.sigma_noise)
            # XXX it seems that we do not need binding to object, but may be
            # commented out code would return?
            self._C = km_train_train + \
                  params.sigma_noise ** 2 * \
                  np.identity(km_train_train.shape[0], 'd')
            # The following decomposition could raise
            # np.linalg.linalg.LinAlgError because of numerical
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
            #    self._L = L = np.tril(self._LL[0])
            # 2) self._L = scipy.linalg.cholesky(self._C, lower=True)
            # 3) self._L = numpy.linalg.cholesky(self._C)
            # Even though 1 is the fastest we choose 2 since 1 does
            # not return a clean lower-triangular matrix (see docstring).

            # PBS: I just made it so the KernelMatrix is regularized
            # all the time.  I figured that if ever you were going to
            # use regularization, you would want to set it yourself
            # and use the same value for all folds of your data.
            # YOH: Ideally so, but in real "use cases" some might have no
            #      clue, also our unittests (actually clfs_examples) might
            #      fail without any good reason.  So lets return a magic with
            #      an option to forbid any regularization (if lm is None)
            try:
                # apply regularization
                lm, C = params.lm, self._C
                if lm is not None:
                    epsilon = lm * np.eye(C.shape[0])
                    self._L = SLcholesky(C + epsilon, lower=True)
                else:
                    # do 10 attempts to raise each time by 10
                    self._L = _SLcholesky_autoreg(C, nsteps=None, lower=True)
                self._LL = (self._L, True)
            except SLAError:
                raise SLAError("Kernel matrix is not positive, definite. "
                               "Try increasing the lm parameter.")
                pass
            newL = True
        else:
            if __debug__:
                debug("GPR", "Not computing L since kernel, data and params "
                      "stayed the same")

        # XXX we leave _alpha being recomputed, although we could check
        #   if newL or _changedData['targets']
        #
        if __debug__:
            debug("GPR", "Computing alpha")
        # L = self._L                 # reuse
        # self._alpha = NLAsolve(L.transpose(),
        #                              NLAsolve(L, train_labels))
        # Faster:
        self._alpha = SLcho_solve(self._LL, train_labels)

        # compute only if the state is enabled
        if self.ca.is_enabled('log_marginal_likelihood'):
            self.compute_log_marginal_likelihood()
            pass

        if retrainable:
            # we must assign it only if it is retrainable
            self.ca.retrained = not newkernel or not newL

        if __debug__:
            debug("GPR", "Done training")

        pass


    @accepts_dataset_as_samples
    def _predict(self, data):
        """
        Predict the output for the provided data.
        """
        retrainable = self.params.retrainable
        ca = self.ca

        if not retrainable or self._changedData['testdata'] \
               or self._km_train_test is None:
            if __debug__:
                debug('GPR', "Computing train test kernel matrix")
            self.__kernel.compute(self._train_fv, data)
            km_train_test = asarray(self.__kernel)
            if retrainable:
                self._km_train_test = km_train_test
                ca.repredicted = False
        else:
            if __debug__:
                debug('GPR', "Not recomputing train test kernel matrix")
            km_train_test = self._km_train_test
            ca.repredicted = True


        predictions = Ndot(km_train_test.transpose(), self._alpha)

        if ca.is_enabled('predicted_variances'):
            # do computation only if conditional attribute was enabled
            if not retrainable or self._km_test_test is None \
                   or self._changedData['testdata']:
                if __debug__:
                    debug('GPR', "Computing test test kernel matrix")
                self.__kernel.compute(data)
                km_test_test = asarray(self.__kernel)
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
            piv = np.arange(L.shape[0])
            v = SL.lu_solve((L.T, piv), km_train_test, trans=1)
            # self.predicted_variances = \
            #     Ndiag(km_test_test - Ndot(v.T, v)) \
            #     + self.sigma_noise**2
            # Faster formula: np.diag(Ndot(v.T, v)) = (v**2).sum(0):
            ca.predicted_variances = Ndiag(km_test_test) - (v ** 2).sum(0) \
                                       + self.params.sigma_noise ** 2
            pass

        if __debug__:
            debug("GPR", "Done predicting")
        ca.estimates = predictions
        return predictions


    ##REF: Name was automagically refactored
    def _set_retrainable(self, value, force=False):
        """Internal function : need to set _km_test_test
        """
        super(GPR, self)._set_retrainable(value, force)
        if force or (value and value != self.params.retrainable):
            self._km_test_test = None


    def _untrain(self):
        super(GPR, self)._untrain()
        # XXX might need to take special care for retrainable. later
        self._init_internals()


    def set_hyperparameters(self, hyperparameter):
        """
        Set hyperparameters' values.

        Note that 'hyperparameter' is a sequence so the order of its
        values is important. First value must be sigma_noise, then
        other kernel's hyperparameters values follow in the exact
        order the kernel expect them to be.
        """
        if hyperparameter[0] < self.params['sigma_noise'].min:
            raise InvalidHyperparameterError()
        self.params.sigma_noise = hyperparameter[0]
        if hyperparameter.size > 1:
            self.__kernel.set_hyperparameters(hyperparameter[1:])
            pass
        return

    kernel = property(fget=lambda self:self.__kernel)
    pass


class GPRLinearWeights(Sensitivity):
    """`SensitivityAnalyzer` that reports the weights GPR trained
    on a given `Dataset`.

    In case of LinearKernel compute explicitly the coefficients
    of the linear regression, together with their variances (if
    requested).

    Note that the intercept is not computed.
    """

    variances = ConditionalAttribute(enabled=False,
        doc="Variances of the weights (for GeneralizedLinearKernel)")

    _LEGAL_CLFS = [ GPR ]


    def _call(self, dataset):
        """Extract weights from GPR
        """

        clf = self.clf
        kernel = clf.kernel
        train_fv = clf._train_fv
        if isinstance(kernel, LinearKernel):
            Sigma_p = 1.0
        else:
            Sigma_p = kernel.params.Sigma_p

        weights = Ndot(Sigma_p,
                        Ndot(train_fv.T, clf._alpha))

        if self.ca.is_enabled('variances'):
            # super ugly formulas that can be quite surely improved:
            tmp = np.linalg.inv(clf._L)
            Kyinv = Ndot(tmp.T, tmp)
            # XXX in such lengthy matrix manipulations you might better off
            #     using np.matrix where * is a matrix product
            self.ca.variances = Ndiag(
                Sigma_p -
                Ndot(Sigma_p,
                      Ndot(train_fv.T,
                            Ndot(Kyinv,
                                  Ndot(train_fv, Sigma_p)))))
        return Dataset(np.atleast_2d(weights))


if externals.exists('openopt'):

    from mvpa2.clfs.model_selector import ModelSelector

    class GPRWeights(Sensitivity):
        """`SensitivityAnalyzer` that reports the weights GPR trained
        on a given `Dataset`.
        """

        _LEGAL_CLFS = [ GPR ]

        def _call(self, ds_):
            """Extract weights from GPR

            .. note:
              Input dataset is not actually used. New dataset is
              constructed from what is known to the classifier
            """

            clf = self.clf
            # normalize data:
            clf._train_labels = (clf._train_labels - clf._train_labels.mean()) \
                                / clf._train_labels.std()
            # clf._train_fv = (clf._train_fv-clf._train_fv.mean(0)) \
            #                  /clf._train_fv.std(0)
            ds = dataset_wizard(samples=clf._train_fv, targets=clf._train_labels)
            clf.ca.enable("log_marginal_likelihood")
            ms = ModelSelector(clf, ds)
            # Note that some kernels does not have gradient yet!
            # XXX Make it initialize to clf's current hyperparameter values
            #     or may be add ability to specify starting points in the constructor
            sigma_noise_initial = 1.0e-5
            sigma_f_initial = 1.0
            length_scale_initial = np.ones(ds.nfeatures)*1.0e4
            # length_scale_initial = np.random.rand(ds.nfeatures)*1.0e4
            hyp_initial_guess = np.hstack([sigma_noise_initial,
                                          sigma_f_initial,
                                          length_scale_initial])
            fixedHypers = array([0]*hyp_initial_guess.size, dtype=bool)
            fixedHypers = None
            problem =  ms.max_log_marginal_likelihood(
                hyp_initial_guess=hyp_initial_guess,
                optimization_algorithm="scipy_lbfgsb",
                ftol=1.0e-3, fixedHypers=fixedHypers,
                use_gradient=True, logscale=True)
            if __debug__ and 'GPR_WEIGHTS' in debug.active:
                problem.iprint = 1
            lml = ms.solve()
            weights = 1.0/ms.hyperparameters_best[2:] # weight = 1/length_scale
            if __debug__:
                debug("GPR",
                      "%s, train: shape %s, labels %s, min:max %g:%g, "
                      "sigma_noise %g, sigma_f %g" %
                      (clf, clf._train_fv.shape, np.unique(clf._train_labels),
                       clf._train_fv.min(), clf._train_fv.max(),
                       ms.hyperparameters_best[0], ms.hyperparameters_best[1]))

            return weights

