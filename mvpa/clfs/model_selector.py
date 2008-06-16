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


    def max_log_marginal_likelihood(self, hyp_initial_guess,maxiter=1, optimization_algorithm="scipy_cg",ftol=1.0e-2):
        """
        Set up the optimization problem in order to maximize
        the log_marginal_likelihood.

        :Parameters:

          parametric_model : Classifier
            the actual parameteric model to be optimized.

          hyp_initial_guess : numpy.ndarray
            set of hyperparameters' initial values where to start optimization.

          optimization_algorithm : string
            actual name of the optimization algorithm. See
            http://scipy.org/scipy/scikits/wiki/NLP
            for a comprehensive/updated list of available NLP solvers.
            (Defaults to 'ralg')

        NOTE: the maximization of log_marginal_likelihood is a non-linear
        optimization problem (NLP). This fact is confirmed by Dmitrey,
        author of OpenOpt.
        """
        self.optimization_algorithm = optimization_algorithm

        def f(*args):
            """
            Wrapper to the log_marginal_likelihood to be
            maximized. Necessary for OpenOpt since it performs
            minimization only.
            """
            self.parametric_model.set_hyperparameters(*args)
            self.parametric_model.train(self.dataset)
            log_marginal_likelihood = self.parametric_model.compute_log_marginal_likelihood()
            return -log_marginal_likelihood # minus sign because optimizers do _minimization_.

        def df(*args):
            """
            Proxy to the log_marginal_likelihood first
            derivative. Necessary for OpenOpt when using derivatives.
            """
            # TODO
            return

        x0 = hyp_initial_guess # vector of hyperparameters' values where to start the search
        self.problem = NLP(f,x0) # actual instance of the OpenOpt non-linear problem
        self.problem.lb = N.zeros(self.problem.n) # set lower bound for hyperparameters: avoid negative hyperparameters. Note: problem.n is the size of hyperparameters' vector
        self.problem.maxiter = maxiter # max number of iterations for the optimizer.
        self.problem.checkdf = True # check whether the derivative of log_marginal_likelihood converged to zero before ending optimization
        self.problem.ftol = ftol # set increment of log_marginal_likelihood under which the optimizer stops
        self.problem.iprint = 0 # shut up OpenOpt (note: -1 = no logs, 0 = small log, 1 = verbose)
        return self.problem


    def solve(self, problem=None):
        """Solve the minimization problem, check outcome and collect results.
        """
        result = self.problem.solve(self.optimization_algorithm) # perform optimization!
        if result.stopcase == -1:
            print "Unable to find a maximum to log_marginal_likelihood"
        elif result.stopcase == 0:
            print "Limits exceeded"
        elif result.stopcase == 1:
            self.hyperparameters_best = result.xf # best hyperparameters found # NOTE is it better to return a copy?
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

    print "GPR:",

    train_size = 40
    test_size = 100
    F = 1

    data_train, label_train = gpr.gen_data(train_size, F)
    # print label_train

    data_test, label_test = gpr.gen_data(test_size, F, flat=True)
    # print label_test

    dataset = Dataset(samples=data_train, labels=label_train)

    regression = True
    logml = True

    k = kernel.KernelLinear(coefficient=N.ones(1))
    # k = kernel.KernelConstant()
    g = gpr.GPR(k,regression=regression)
    g.states.enable("log_marginal_likelihood")
    # g.train_fv = dataset.samples
    # g.train_labels = dataset.labels

    print "GPR hyperparameters' search through maximization of marginal likelihood on train data."
    print
    ms = ModelSelector(g,dataset)

    sigma_noise_initial = 1.0
    length_scale_initial = 1.0

    problem =  ms.max_log_marginal_likelihood(hyp_initial_guess=[sigma_noise_initial,length_scale_initial], optimization_algorithm="ralg", ftol=1.0e-4)
    lml = ms.solve()
    sigma_noise_best, length_scale_best = ms.hyperparameters_best
    print
    print "Best sigma_noise:",sigma_noise_best
    print "Best length_scale:",length_scale_best
    print "Best log_marginal_likelihood:",lml

    gpr.compute_prediction(sigma_noise_best,length_scale_best,regression,dataset,data_test,label_test,F)

    print
    print "GPR ARD on dataset from Williams and Rasmussen 1996:"
    data, labels = data_generators.wr1996()
    # data = N.hstack([data]*10) # test a larger set of dimensions: reduce ftol!
    dataset = Dataset(samples=data, labels=labels)
    k = kernel.KernelSquaredExponential(length_scale=N.ones(dataset.samples.shape[1]))
    g = gpr.GPR(k, regression=regression)
    ms = ModelSelector(g, dataset)

    sigma_noise_initial = 0.01
    length_scales_initial = 0.5*N.ones(dataset.samples.shape[1])

    problem =  ms.max_log_marginal_likelihood(hyp_initial_guess=N.hstack([sigma_noise_initial,length_scales_initial]), optimization_algorithm="ralg")
    lml = ms.solve()
    sigma_noise_best = ms.hyperparameters_best[0]
    length_scales_best = ms.hyperparameters_best[1:]
    print
    print "Best sigma_noise:",sigma_noise_best
    print "Best length_scale:",length_scales_best
    print "Best log_marginal_likelihood:",lml

