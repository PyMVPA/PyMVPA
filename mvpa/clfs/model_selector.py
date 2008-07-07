#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
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

# no sense to import this module if openopt is not available
if externals.exists("openopt", raiseException=True):
    from scikits.openopt import NLP


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


    def max_log_marginal_likelihood(self, hyp_initial_guess, maxiter=1, optimization_algorithm="scipy_cg", ftol=1.0e-3, fixedHypers=None):
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
        self.optimization_algorithm = optimization_algorithm

        self.hyp_initial_guess = N.array(hyp_initial_guess)
        if fixedHypers is None:
            fixedHypers = N.zeros(self.hyp_initial_guess.shape[0],dtype=bool)
            pass
        self.freeVars = -fixedHypers

        self.hyp_running_guess = self.hyp_initial_guess.copy()

        def f(x):
            """
            Wrapper to the log_marginal_likelihood to be
            maximized.
            """
            # since some OpenOpen MLP solvers does not implement lower bounds
            # the hyperparameters bounds are implemented inside PyMVPA:
            # (see dmitrey post on [SciPy-user] 20080628)
            self.hyp_running_guess[self.freeVars] = x
            try:
                self.parametric_model.set_hyperparameters(self.hyp_running_guess)
            except InvalidHyperparameterError:
                # print "WARNING: invalid hyperparameters!"
                return -N.inf
            self.parametric_model.train(self.dataset)
            log_marginal_likelihood = self.parametric_model.compute_log_marginal_likelihood()
            return log_marginal_likelihood

        def df(x):
            """
            Proxy to the log_marginal_likelihood first
            derivative. Necessary for OpenOpt when using derivatives.
            """
            # TODO
            return

        x0 = self.hyp_initial_guess[self.freeVars] # vector of hyperparameters' values where to start the search
        contol = 1.0e-6
        self.problem = NLP(f, x0, contol=contol, goal='maximum') # actual instance of the OpenOpt non-linear problem
        self.problem.lb = N.zeros(self.problem.n)+contol # set lower bound for hyperparameters: avoid negative hyperparameters. Note: problem.n is the size of hyperparameters' vector
        self.problem.maxiter = maxiter # max number of iterations for the optimizer.
        self.problem.checkdf = True # check whether the derivative of log_marginal_likelihood converged to zero before ending optimization
        self.problem.ftol = ftol # set increment of log_marginal_likelihood under which the optimizer stops
        self.problem.iprint = 0 # shut up OpenOpt (note: -1 = no logs, 0 = small log, 1 = verbose)
        return self.problem


    def solve(self, problem=None):
        """Solve the maximization problem, check outcome and collect results.
        """
        # XXX: this method can be made more abstract in future in the
        # sense that it could work not only for
        # log_marginal_likelihood but other measures as well
        # (e.g. cross-valideted error).
        
        if N.all(self.freeVars==False): # no optimization needed
            self.hyperparameters_best = self.hyp_initial_guess.copy()
            try:
                self.parametric_model.set_hyperparameters(self.hyperparameters_best)
            except InvalidHyperparameterError:
                # print "WARNING: invalid hyperparameters!"
                self.log_marginal_likelihood_best = -N.inf
                return self.log_marginal_likelihood_best
            self.parametric_model.train(self.dataset)
            self.log_marginal_likelihood_best = self.parametric_model.compute_log_marginal_likelihood()
            return self.log_marginal_likelihood_best
            
        result = self.problem.solve(self.optimization_algorithm) # perform optimization!
        if result.stopcase == -1:
            print "Unable to find a maximum to log_marginal_likelihood"
        elif result.stopcase == 0:
            print "Limits exceeded"
        elif result.stopcase == 1:
            self.hyperparameters_best = self.hyp_initial_guess.copy()
            self.hyperparameters_best[self.freeVars] = result.xf # best hyperparameters found # NOTE is it better to return a copy?
            self.log_marginal_likelihood_best = -result.ff # actual best vuale of log_marginal_likelihood

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

    sigma_noise_initial = 1.0
    sigma_f_initial = 1.0
    length_scale_initial = 1.0

    problem =  ms.max_log_marginal_likelihood(hyp_initial_guess=[sigma_noise_initial,sigma_f_initial,length_scale_initial], optimization_algorithm="ralg", ftol=1.0e-3,fixedHypers=N.array([0,1,0],dtype=bool))
    # problem =  ms.max_log_marginal_likelihood(hyp_initial_guess=[1.0,1.0], optimization_algorithm="ralg", ftol=1.0e-3)
    
    lml = ms.solve()
    print ms.hyperparameters_best
    sigma_noise_best, sigma_f_best, length_scale_best = ms.hyperparameters_best
    print
    print "Best sigma_noise:",sigma_noise_best
    print "Best sigma_f:",sigma_f_best
    print "Best length_scale:",length_scale_best
    print "Best log_marginal_likelihood:",lml

    gpr.compute_prediction(sigma_noise_best,length_scale_best,regression,dataset,data_test,label_test,F)

    print
    print "GPR ARD on dataset from Williams and Rasmussen 1996:"
    # data = N.hstack([data]*10) # test a larger set of dimensions: reduce ftol!
    dataset =  data_generators.wr1996()
    # k = kernel.KernelConstant()
    # k = kernel.KernelLinear()
    k = kernel.KernelSquaredExponential()
    # k = kernel.KernelExponential()
    # k = kernel.KernelMatern_3_2()
    # k = kernel.KernelMatern_5_2()
    g = gpr.GPR(k, regression=regression)
    g.states.enable("log_marginal_likelihood")
    # if isinstance(k, kernel.KernelLinear):
    #     g.states.enable("linear_weights")
    #     pass
    ms = ModelSelector(g, dataset)

    sigma_noise_initial = 0.01
    length_scales_initial = 0.5*N.ones(dataset.samples.shape[1])

    # problem =  ms.max_log_marginal_likelihood(hyp_initial_guess=N.ones(2)*0.1, optimization_algorithm="ralg")
    problem =  ms.max_log_marginal_likelihood(hyp_initial_guess=N.ones(dataset.samples.shape[1]+2)*1.0e-1, optimization_algorithm="ralg", fixedHypers = N.array([0,0,0,0,0,0,0,0],dtype=bool))
    # problem =  ms.max_log_marginal_likelihood(hyp_initial_guess=N.hstack([sigma_noise_initial,length_scales_initial]), optimization_algorithm="ralg")
    lml = ms.solve()
    sigma_noise_best = ms.hyperparameters_best[0]
    length_scales_best = ms.hyperparameters_best[1:]
    print
    print "Best sigma_noise:",sigma_noise_best
    print "Best length_scale:",length_scales_best
    print "Best log_marginal_likelihood:",lml

    # g.states.enable("linear_weights")
    # g.states.enable("linear_weights_variances")    
    # g.compute_linear_weights()
    # print g.linear_weights
    # print g.linear_weights_variances
