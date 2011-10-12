# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   Copyright (c) 2008 Emanuele Olivetti <emanuele@relativita.com>
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Model selction."""

__docformat__ = 'restructuredtext'


import numpy as np
from mvpa2.base import externals
from mvpa2.misc.exceptions import InvalidHyperparameterError

if externals.exists("scipy", raise_=True):
    import scipy.linalg as SL

# no sense to import this module if openopt is not available
if externals.exists("openopt", raise_=True):
    try:
        from openopt import NLP
    except ImportError:
        from scikits.openopt import NLP

if __debug__:
    from mvpa2.base import debug

def _openopt_debug():
    # shut up or make verbose OpenOpt
    # (-1 = no logs, 0 = small log, 1 = verbose)
    if __debug__:
        da = debug.active
        if 'OPENOPT' in da:
            return 1
        elif 'MOD_SEL' in da:
            return 0
    return -1


class ModelSelector(object):
    """Model selection facility.

    Select a model among multiple models (i.e., a parametric model,
    parametrized by a set of hyperparamenters).
    """

    def __init__(self, parametric_model, dataset):
        """TODO:
        """
        self.parametric_model = parametric_model
        self.dataset = dataset
        self.hyperparameters_best = None
        self.log_marginal_likelihood_best = None
        self.problem = None
        pass

    def max_log_marginal_likelihood(self, hyp_initial_guess, maxiter=1,
            optimization_algorithm="scipy_cg", ftol=1.0e-3, fixedHypers=None,
            use_gradient=False, logscale=False):
        """
        Set up the optimization problem in order to maximize
        the log_marginal_likelihood.

        Parameters
        ----------
        parametric_model : Classifier
          the actual parameteric model to be optimized.
        hyp_initial_guess : numpy.ndarray
          set of hyperparameters' initial values where to start
          optimization.
        optimization_algorithm : string
          actual name of the optimization algorithm. See
          http://scipy.org/scipy/scikits/wiki/NLP
          for a comprehensive/updated list of available NLP solvers.
          (Defaults to 'ralg')
        ftol : float
          threshold for the stopping criterion of the solver,
          which is mapped in OpenOpt NLP.ftol
          (Defaults to 1.0e-3)
        fixedHypers : numpy.ndarray (boolean array)
          boolean vector of the same size of hyp_initial_guess;
          'False' means that the corresponding hyperparameter must
          be kept fixed (so not optimized).
          (Defaults to None, which during means all True)

        Notes
        -----
        The maximization of log_marginal_likelihood is a non-linear
        optimization problem (NLP). This fact is confirmed by Dmitrey,
        author of OpenOpt.
        """
        self.problem = None
        self.use_gradient = use_gradient
        self.logscale = logscale # use log-scale on hyperparameters to enhance numerical stability
        self.optimization_algorithm = optimization_algorithm
        self.hyp_initial_guess = np.array(hyp_initial_guess)
        self.hyp_initial_guess_log = np.log(self.hyp_initial_guess)
        if fixedHypers is None:
            fixedHypers = np.zeros(self.hyp_initial_guess.shape[0],dtype=bool)
            pass
        self.freeHypers = -fixedHypers
        if self.logscale:
            self.hyp_running_guess = self.hyp_initial_guess_log.copy()
        else:
            self.hyp_running_guess = self.hyp_initial_guess.copy()
            pass
        self.f_last_x = None

        def f(x):
            """
            Wrapper to the log_marginal_likelihood to be
            maximized.
            """
            # XXX EO: since some OpenOpt NLP solvers does not
            # implement lower bounds the hyperparameters bounds are
            # implemented inside PyMVPA: (see dmitrey's post on
            # [SciPy-user] 20080628).
            #
            # XXX EO: OpenOpt does not implement optimization of a
            # subset of the hyperparameters so it is implemented here.
            #
            # XXX EO: OpenOpt does not implement logrithmic scale of
            # the hyperparameters (to enhance numerical stability), so
            # it is implemented here.
            self.f_last_x = x.copy()
            self.hyp_running_guess[self.freeHypers] = x
            # REMOVE print "guess:",self.hyp_running_guess,x
            try:
                if self.logscale:
                    self.parametric_model.set_hyperparameters(np.exp(self.hyp_running_guess))
                else:
                    self.parametric_model.set_hyperparameters(self.hyp_running_guess)
                    pass
            except InvalidHyperparameterError:
                if __debug__: debug("MOD_SEL","WARNING: invalid hyperparameters!")
                return -np.inf
            try:
                self.parametric_model.train(self.dataset)
            except (np.linalg.linalg.LinAlgError, SL.basic.LinAlgError, ValueError):
                # Note that ValueError could be raised when Cholesky gets Inf or Nan.
                if __debug__: debug("MOD_SEL", "WARNING: Cholesky failed! Invalid hyperparameters!")
                return -np.inf
            log_marginal_likelihood = self.parametric_model.compute_log_marginal_likelihood()
            # REMOVE print log_marginal_likelihood
            return log_marginal_likelihood

        def df(x):
            """
            Proxy to the log_marginal_likelihood first
            derivative. Necessary for OpenOpt when using derivatives.
            """
            self.hyp_running_guess[self.freeHypers] = x
            # REMOVE print "df guess:",self.hyp_running_guess,x
            # XXX EO: Most of the following lines can be skipped if
            # df() is computed just after f() with the same
            # hyperparameters. The partial results obtained during f()
            # are what is needed for df(). For now, in order to avoid
            # bugs difficult to trace, we keep this redunundancy. A
            # deep check with how OpenOpt works or using memoization
            # should solve this issue.
            try:
                if self.logscale:
                    self.parametric_model.set_hyperparameters(np.exp(self.hyp_running_guess))
                else:
                    self.parametric_model.set_hyperparameters(self.hyp_running_guess)
                    pass
            except InvalidHyperparameterError:
                if __debug__: debug("MOD_SEL", "WARNING: invalid hyperparameters!")
                return -np.inf
            # Check if it is possible to avoid useless computations
            # already done in f(). According to tests and information
            # collected from OpenOpt people, it is sufficiently
            # unexpected that the following test succeed:
            if np.any(x!=self.f_last_x):
                if __debug__: debug("MOD_SEL","UNEXPECTED: recomputing train+log_marginal_likelihood.")
                try:
                    self.parametric_model.train(self.dataset)
                except (np.linalg.linalg.LinAlgError, SL.basic.LinAlgError, ValueError):
                    if __debug__: debug("MOD_SEL", "WARNING: Cholesky failed! Invalid hyperparameters!")
                    # XXX EO: which value for the gradient to return to
                    # OpenOpt when hyperparameters are wrong?
                    return np.zeros(x.size)
                log_marginal_likelihood = self.parametric_model.compute_log_marginal_likelihood() # recompute what's needed (to be safe) REMOVE IN FUTURE!
                pass
            if self.logscale:
                gradient_log_marginal_likelihood = self.parametric_model.compute_gradient_log_marginal_likelihood_logscale()
            else:
                gradient_log_marginal_likelihood = self.parametric_model.compute_gradient_log_marginal_likelihood()
                pass
            # REMOVE print "grad:",gradient_log_marginal_likelihood
            return gradient_log_marginal_likelihood[self.freeHypers]


        if self.logscale:
            # vector of hyperparameters' values where to start the search
            x0 = self.hyp_initial_guess_log[self.freeHypers]
        else:
            x0 = self.hyp_initial_guess[self.freeHypers]
            pass
        self.contol = 1.0e-20 # Constraint tolerance level
        # XXX EO: is it necessary to use contol when self.logscale is
        # True and there is no lb? Ask dmitrey.
        if self.use_gradient:
            # actual instance of the OpenOpt non-linear problem
            self.problem = NLP(f, x0, df=df, contol=self.contol, goal='maximum')
        else:
            self.problem = NLP(f, x0, contol=self.contol, goal='maximum')
            pass
        self.problem.name = "Max LogMargLikelihood"
        if not self.logscale:
             # set lower bound for hyperparameters: avoid negative
             # hyperparameters. Note: problem.n is the size of
             # hyperparameters' vector
            self.problem.lb = np.zeros(self.problem.n)+self.contol
            pass
        # max number of iterations for the optimizer.
        self.problem.maxiter = maxiter
        # check whether the derivative of log_marginal_likelihood converged to
        # zero before ending optimization
        self.problem.checkdf = True
         # set increment of log_marginal_likelihood under which the optimizer stops
        self.problem.ftol = ftol
        self.problem.iprint = _openopt_debug()
        return self.problem


    def solve(self, problem=None):
        """Solve the maximization problem, check outcome and collect results.
        """
        # XXX: this method can be made more abstract in future in the
        # sense that it could work not only for
        # log_marginal_likelihood but other measures as well
        # (e.g. cross-valideted error).

        if np.all(self.freeHypers==False): # no optimization needed
            self.hyperparameters_best = self.hyp_initial_guess.copy()
            try:
                self.parametric_model.set_hyperparameters(self.hyperparameters_best)
            except InvalidHyperparameterError:
                if __debug__: debug("MOD_SEL", "WARNING: invalid hyperparameters!")
                self.log_marginal_likelihood_best = -np.inf
                return self.log_marginal_likelihood_best
            self.parametric_model.train(self.dataset)
            self.log_marginal_likelihood_best = self.parametric_model.compute_log_marginal_likelihood()
            return self.log_marginal_likelihood_best

        result = self.problem.solve(self.optimization_algorithm) # perform optimization!
        if result.stopcase == -1:
            # XXX: should we use debug() for the following messages?
            # If so, how can we track the missing convergence to a
            # solution?
            print "Unable to find a maximum to log_marginal_likelihood"
        elif result.stopcase == 0:
            print "Limits exceeded"
        elif result.stopcase == 1:
            self.hyperparameters_best = self.hyp_initial_guess.copy()
            if self.logscale:
                self.hyperparameters_best[self.freeHypers] = np.exp(result.xf) # best hyperparameters found # NOTE is it better to return a copy?
            else:
                self.hyperparameters_best[self.freeHypers] = result.xf
                pass
            self.log_marginal_likelihood_best = result.ff # actual best vuale of log_marginal_likelihood
            pass
        self.stopcase = result.stopcase
        return self.log_marginal_likelihood_best

    pass
