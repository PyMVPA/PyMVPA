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


import numpy as N
from mvpa.base import externals
from mvpa.misc.exceptions import InvalidHyperparameterError

externals.exists("scipy", raiseException=True)
import scipy.linalg as SL

# no sense to import this module if openopt is not available
if externals.exists("openopt", raiseException=True):
    from scikits.openopt import NLP

if __debug__:
    from mvpa.base import debug

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

        :Parameters:

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

        NOTE: the maximization of log_marginal_likelihood is a non-linear
        optimization problem (NLP). This fact is confirmed by Dmitrey,
        author of OpenOpt.
        """
        self.problem = None
        self.use_gradient = use_gradient
        self.logscale = logscale # use log-scale on hyperparameters to enhance numerical stability
        self.optimization_algorithm = optimization_algorithm
        self.hyp_initial_guess = N.array(hyp_initial_guess)
        self.hyp_initial_guess_log = N.log(self.hyp_initial_guess)
        if fixedHypers is None:
            fixedHypers = N.zeros(self.hyp_initial_guess.shape[0],dtype=bool)
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
                    self.parametric_model.set_hyperparameters(N.exp(self.hyp_running_guess))
                else:
                    self.parametric_model.set_hyperparameters(self.hyp_running_guess)
                    pass
            except InvalidHyperparameterError:
                if __debug__: debug("MOD_SEL","WARNING: invalid hyperparameters!")
                return -N.inf
            try:
                self.parametric_model.train(self.dataset)
            except (N.linalg.linalg.LinAlgError, SL.basic.LinAlgError, ValueError):
                # Note that ValueError could be raised when Cholesky gets Inf or Nan.
                if __debug__: debug("MOD_SEL", "WARNING: Cholesky failed! Invalid hyperparameters!")
                return -N.inf
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
                    self.parametric_model.set_hyperparameters(N.exp(self.hyp_running_guess))
                else:
                    self.parametric_model.set_hyperparameters(self.hyp_running_guess)
                    pass
            except InvalidHyperparameterError:
                if __debug__: debug("MOD_SEL", "WARNING: invalid hyperparameters!")
                return -N.inf
            # Check if it is possible to avoid useless computations
            # already done in f(). According to tests and information
            # collected from OpenOpt people, it is sufficiently
            # unexpected that the following test succeed:
            if N.any(x!=self.f_last_x):
                if __debug__: debug("MOD_SEL","UNEXPECTED: recomputing train+log_marginal_likelihood.")
                try:
                    self.parametric_model.train(self.dataset)
                except (N.linalg.linalg.LinAlgError, SL.basic.LinAlgError, ValueError):
                    if __debug__: debug("MOD_SEL", "WARNING: Cholesky failed! Invalid hyperparameters!")
                    # XXX EO: which value for the gradient to return to
                    # OpenOpt when hyperparameters are wrong?
                    return N.zeros(x.size)
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
            self.problem.lb = N.zeros(self.problem.n)+self.contol
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

        if N.all(self.freeHypers==False): # no optimization needed
            self.hyperparameters_best = self.hyp_initial_guess.copy()
            try:
                self.parametric_model.set_hyperparameters(self.hyperparameters_best)
            except InvalidHyperparameterError:
                if __debug__: debug("MOD_SEL", "WARNING: invalid hyperparameters!")
                self.log_marginal_likelihood_best = -N.inf
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
                self.hyperparameters_best[self.freeHypers] = N.exp(result.xf) # best hyperparameters found # NOTE is it better to return a copy?
            else:
                self.hyperparameters_best[self.freeHypers] = result.xf
                pass
            self.log_marginal_likelihood_best = result.ff # actual best vuale of log_marginal_likelihood
            pass
        self.stopcase = result.stopcase
        return self.log_marginal_likelihood_best

    pass



if __name__ == "__main__":

    import gpr
    import kernel
    from mvpa.misc import data_generators
    from mvpa.base import externals
    N.random.seed(1)

    if externals.exists("pylab", force=True):
        import pylab
    pylab.close("all")
    pylab.ion()

    from mvpa.datasets import Dataset
    from mvpa.misc import data_generators

    print "GPR:",

    train_size = 40
    test_size = 100
    F = 1

    dataset = data_generators.sinModulated(train_size, F)
    # print label_train

    dataset_test = data_generators.sinModulated(test_size, F, flat=True)
    data_test = dataset_test.samples
    label_test = dataset_test.labels
    # print label_test

    regression = True
    logml = True

    k = kernel.KernelSquaredExponential()
    g = gpr.GPR(k,regression=regression)
    g.states.enable("log_marginal_likelihood")
    # g.train_fv = dataset.samples
    # g.train_labels = dataset.labels

    print "GPR hyperparameters' search through maximization of marginal likelihood on train data."
    print
    ms = ModelSelector(g,dataset)

    sigma_noise_initial = 1.0e0 # 0.154142346606
    sigma_f_initial = 1.0e0 # 0.687554871058
    length_scale_initial = 1.0e0 # 0.263620251025

    problem =  ms.max_log_marginal_likelihood(hyp_initial_guess=[sigma_noise_initial,sigma_f_initial,length_scale_initial], optimization_algorithm="ralg", ftol=1.0e-8,fixedHypers=N.array([0,0,0],dtype=bool))
    # problem =  ms.max_log_marginal_likelihood(hyp_initial_guess=[1.0,1.0], optimization_algorithm="ralg", ftol=1.0e-3)

    lml = ms.solve()
    # print ms.hyperparameters_best
    sigma_noise_best, sigma_f_best, length_scale_best = ms.hyperparameters_best
    print
    print "Best sigma_noise:",sigma_noise_best
    print "Best sigma_f:",sigma_f_best
    print "Best length_scale:",length_scale_best
    print "Best log_marginal_likelihood:",lml

    # Best sigma_noise: 0.154142346606
    # Best sigma_f: 0.687554871058
    # Best length_scale: 0.263620251025
    # Best log_marginal_likelihood: -3.54790161194

    gpr.compute_prediction(sigma_noise_best,sigma_f_best,length_scale_best,regression,dataset,data_test,label_test,F)


    print
    print "GPR ARD on dataset from Williams and Rasmussen 1996:"
    dataset =  data_generators.wr1996()
    # dataset.samples = N.hstack([dataset.samples]*10) # enlarge dataset's dimensionality, for testing high dimensions

    # Uncomment the kernel you like:

    # Squared Exponential kernel:
    k = kernel.KernelSquaredExponential()
    sigma_noise_initial = 1.0e-0
    sigma_f_initial = 1.0e-0
    length_scale_initial = N.ones(dataset.samples.shape[1])*1.0e-0
    hyp_initial_guess = N.hstack([sigma_noise_initial,sigma_f_initial,length_scale_initial])
    # fixedHypers = N.array([0,0,0,0,0,0,0,0],dtype=bool)
    fixedHypers = N.array([0]*(hyp_initial_guess.size),dtype=bool)

    # k = kernel.KernelLinear()
    # sigma_noise_initial = 1.0e-0
    # sigma_0_initial = 1.0e-0
    # Sigma_p_initial = N.ones(dataset.samples.shape[1])*1.0e-3
    # hyp_initial_guess = N.hstack([sigma_noise_initial,sigma_0_initial,Sigma_p_initial])
    # # fixedHypers = N.array([0,0,0,0,0,0,0,0],dtype=bool)
    # fixedHypers = N.array([0]*hyp_initial_guess.size,dtype=bool)

    # k = kernel.KernelConstant()
    # sigma_noise_initial = 1.0e-0
    # sigma_0_initial = 1.0e-0
    # hyp_initial_guess = N.array([sigma_noise_initial, sigma_0_initial])
    # # fixedHypers = N.array([0,0],dtype=bool)
    # fixedHypers = N.array([0]*hyp_initial_guess.size,dtype=bool)

    # # Exponential kernel:
    # k = kernel.KernelExponential()
    # sigma_noise_initial = 1.0e0
    # sigma_f_initial = 1.0e0
    # length_scale_initial = N.ones(dataset.samples.shape[1])*1.0e0
    # # length_scale_initial = 1.0
    # hyp_initial_guess = N.hstack([sigma_noise_initial,sigma_f_initial,length_scale_initial])
    # print "hyp_initial_guess:",hyp_initial_guess
    # # fixedHypers = N.array([0,0,0,0,0,0,0,0],dtype=bool)
    # fixedHypers = N.array([0]*(hyp_initial_guess.size),dtype=bool)
    # # expected results (compute with use_gradient=False):
    # # objFunValue: 161.97952 (feasible, max constraint =  0)
    # # [  6.72069299e-04   3.16515151e-01   3.11122154e+00   1.54833211e+00
    # #    1.94703461e+00   3.11122835e+00   4.37916189e+01   2.65398676e+01]


    # # Matern_3_2 kernel:
    # k = kernel.KernelMatern_3_2()
    # sigma_noise_initial = 1.0e0
    # sigma_f_initial = 1.0e0
    # length_scale_initial = N.ones(dataset.samples.shape[1])*1.0e0
    # # length_scale_initial = 1.0
    # hyp_initial_guess = N.hstack([sigma_noise_initial,sigma_f_initial,length_scale_initial])
    # print "hyp_initial_guess:",hyp_initial_guess
    # # fixedHypers = N.array([0,0,0,0,0,0,0,0],dtype=bool)
    # fixedHypers = N.array([0]*(hyp_initial_guess.size),dtype=bool)


    # # Rational Quadratic kernel:
    # k = kernel.KernelRationalQuadratic(alpha=0.5)
    # sigma_noise_initial = 1.0e0
    # sigma_f_initial = 1.0e0
    # length_scale_initial = N.ones(dataset.samples.shape[1])*1.0e0
    # # length_scale_initial = 1.0
    # hyp_initial_guess = N.hstack([sigma_noise_initial,sigma_f_initial,length_scale_initial])
    # print "hyp_initial_guess:",hyp_initial_guess
    # # fixedHypers = N.array([0,0,0,0,0,0,0,0],dtype=bool)
    # fixedHypers = N.array([0]*(hyp_initial_guess.size),dtype=bool)


    g = gpr.GPR(k,regression=regression)
    g.states.enable("log_marginal_likelihood")
    ms = ModelSelector(g,dataset)

    # Note that some kernels does not have gradient yet!
    problem =  ms.max_log_marginal_likelihood(hyp_initial_guess=hyp_initial_guess, optimization_algorithm="ralg", ftol=1.0e-5,fixedHypers=fixedHypers,use_gradient=True, logscale=True)
    lml = ms.solve()
    print ms.hyperparameters_best
