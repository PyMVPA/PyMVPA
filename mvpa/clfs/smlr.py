#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Sparse Multinomial Logistic Regression classifier."""

__docformat__ = 'restructuredtext'


import numpy as N

from mvpa.clfs.classifier import Classifier
from mvpa.misc.exceptions import ConvergenceError
from mvpa.misc.state import StateVariable

# Uber-fast C-version of the stepwise regression
from mvpa.clfs.libsmlr import stepwise_regression as _c_stepwise_regression

if __debug__:
    from mvpa.misc import debug


class SMLR(Classifier):
    """Sparse Multinomial Logistic Regression `Classifier`.

    This is an implementation of the SMLR algorithm published in
    Krishnapuram et al. (2005, IEEE Transactions on Pattern Analysis
    and Machine Intelligence).  Be sure to cite that article if you
    use this for your work.
    """

    weights = StateVariable(enabled=False,
                            doc="Weights of the trained classifier")


    def __init__(self, lm=.1, convergence_tol=1e-3,
                 maxiter=10000, bias=True, implementation="C", **kwargs):
        """
        Initialize a SMLR analysis.

        :Parameters:
          lm : float
            The penalty term lambda.  Larger values will give rise
            to more sparsification.
          covergence_tol : float
            When the weight change for each cycle drops below this value
            the regression is considered converged.  Smaller values
            lead to tighter convergence.
          maxiter : int
            Maximum number of iterations before stopping if not converged.
          bias : bool
            Whether to add a bias term to allow fits to data not through
            zero.  Defaults to True.
          implementation : basestr
            Use C (default) or Python as the implementation of
            stepwise_regression. C version brings significant speedup thus
            is the default one.

        TODO:
        1) Add in likelihood calculation
        2) Add optional bias term
        3) Add kernels, not just direct methods.
        """
        # init base class first
        Classifier.__init__(self, **kwargs)

        # set the parameters
        self.__lm = lm
        self.__convergence_tol = convergence_tol
        self.__maxiter = maxiter
        self.__bias = bias

        if not implementation.upper() in ['C', 'PYTHON']:
            raise ValueError, \
                  "Unknown implementation %s of stepwise_regression" % \
                  implementation
        self.__implementation = implementation

    def __repr__(self):
        """String representation of the object
        """
        return """SMLR(lm=%f, convergence_tol=%g, maxiter=%d, bias=%s, implementation='%s', enabled_states=%s)""" % \
            (self.__lm, self.__convergence_tol,
             self.__maxiter, self.__bias, self.__implementation,
             str(self.states.enabled))

    def __label_to_oneofm(self,labels,ulabels):
        """Convert labels to one-of-M form."""
        # allocate for the new one-of-M labels
        new_labels = N.zeros((len(labels),len(ulabels)))

        # loop and convert to one-of-M
        for i,c in enumerate(ulabels):
            new_labels[labels==c,i] = 1

        return new_labels

    def _python_stepwise_regression(self, w, X, XY, Xw, E,
                                    auto_corr,
                                    lambda_over_2_auto_corr,
                                    S,
                                    maxiter,
                                    convergence_tol,
                                    verbose,
                                    seed = None):
        """The (much slower) python version of the stepwise
        regression.  I'm keeping this around for now so that we can
        compare results."""

        # get the data information into easy vars
        ns,nd = X.shape

        # yoh: shouldn't be here and should be derived from the interface
        M = len(self.__ulabels)

        # initialize the iterative optimization
        converged = False
        incr = N.finfo(N.float).max
        non_zero = 0
        basis = 0
        m = 0
        wasted_basis = 0
        cycles = 0
        decrease_factor = 1.
        test_zero_basis = 1.
        sum2_w_diff = 0.0
        sum2_w_old = 0.0
        w_diff = 0.0

        N.random.seed(seed)
        if __debug__:
            debug("SMLR_", "random seed=%s" % seed)


        # perform the optimization
        while not converged and cycles<maxiter:
            # get the starting weight
            w_old = w[basis,m]

            # see if we're gonna update
            if (w_old != 0) or N.random.rand()<= test_zero_basis:
                # let's do it
                # get the probability
                P = E[:,m]/S

                # set the gradient
                grad = XY[basis,m] - N.dot(X[:,basis],P)

                # calculate the new weight with the Laplacian prior
                w_new = w_old + grad/auto_corr[basis]

                # keep weights within bounds
                if w_new > lambda_over_2_auto_corr[basis]:
                    w_new -= lambda_over_2_auto_corr[basis]
                    changed = True
                    # unmark from being zero if necessary
                    if w_old == 0:
                        non_zero+=1
                elif w_new < -lambda_over_2_auto_corr[basis]:
                    w_new += lambda_over_2_auto_corr[basis]
                    changed = True
                    # unmark from being zero if necessary
                    if w_old == 0:
                        non_zero+=1
                else:
                    # gonna zero it out
                    w_new = 0.0

                    # set number of non-zero
                    if w_old == 0:
                        changed = False
                        wasted_basis+=1
                    else:
                        changed = True
                        non_zero-=1

                # process any changes
                if changed:
                    #print "w[%d,%d] = %g" % (basis,m,w_new)
                    # update the expected values
                    w_diff = w_new - w_old;
                    Xw[:,m] = Xw[:,m] + X[:,basis]*w_diff
                    E_new_m = N.exp(Xw[:,m])
                    S += E_new_m - E[:,m]
                    E[:,m] = E_new_m

                    # update the weight
                    w[basis,m] = w_new

                    # keep track of the sqrt sum squared distances
                    sum2_w_diff += w_diff*w_diff;
                    sum2_w_old += w_old*w_old;

            # update the class and basis
            m = N.mod(m+1,M-1)
            if m == 0:
                # we completed a cycle of labels
                basis = N.mod(basis+1,nd)
                if basis == 0:
                    # we completed a cycle of features
                    cycles += 1

                    # assess convergence
                    incr = N.sqrt(sum2_w_diff) / \
                           (N.sqrt(sum2_w_old)+N.finfo(N.float).eps);

                    # save the new weights
                    converged = incr < convergence_tol

                    # update the zero test factors
                    decrease_factor *= (non_zero/float((M-1)*nd))
                    test_zero_basis *= decrease_factor

                    if __debug__:
                        debug("SMLR_", \
                                  "cycle=%d ; incr=%g ; non_zero=%d ; sum2_w_old=%g ; sum2_w_diff=%g" % \
                          (cycles,incr,non_zero,sum2_w_old,sum2_w_diff))

                    # reset the sum diffs
                    sum2_w_diff = 0.0
                    sum2_w_old = 0.0


        if not converged:
            raise ConvergenceError, \
                "More than %d Iterations without convergence" % \
                (maxiter)

        # calcualte the log likelihoods and posteriors for the training data
        #log_likelihood = x

#        if __debug__:
#            debug("SMLR_", \
#                  "SMLR converged after %d steps. Error: %g" % \
#                  (cycles, XXX))

#        print 'cycles=%d ; wasted basis=%g\n' % (cycles,wasted_basis/((M-1)*nd))

        # save the weights
        self.__weights = w
        self.weights = w


    def _train(self, dataset):
        """Train the classifier using `dataset` (`Dataset`).
        """
        # Process the labels to turn into 1 of N encoding
        labels = self.__label_to_oneofm(dataset.labels,dataset.uniquelabels)
        self.__ulabels = dataset.uniquelabels.copy()
        Y = labels
        M = len(self.__ulabels)

        # get the dataset information into easy vars
        X = dataset.samples

        # see if we are adding a bias term
        if self.__bias:
            # append the bias term to the features
            X = N.hstack((X,N.ones((X.shape[0],1),dtype=X.dtype)))

        if self.__implementation.upper() == 'C':
            _stepwise_regression = _c_stepwise_regression
            #
            # TODO: avoid copying to non-contig arrays, use strides in ctypes?
            if not (X.flags['C_CONTIGUOUS'] and X.flags['ALIGNED']):
                if __debug__:
                    debug("SMLR_", "Copying data to get it C_CONTIGUOUS/ALIGNED")
                X = N.array(X, copy=True, dtype=N.double, order='C')
        elif self.__implementation.upper() == 'PYTHON':
            _stepwise_regression = self._python_stepwise_regression
        else:
            raise ValueError, \
                  "Unknown implementation %s of stepwise_regression" % \
                  implementation

        # currently must be double for the C code
        if X.dtype != N.double:
            # must cast to double
            X = X.astype(N.double)

        # set the feature dimensions
        ns,nd = X.shape

        # Precompute what we can
        auto_corr = ((M-1.)/(2.*M))*(N.sum(X*X,0))
        XY = N.dot(X.T,Y[:,:(M-1)])
        lambda_over_2_auto_corr = (self.__lm/2.)/auto_corr

        # set starting values
        w = N.zeros((nd,M-1),dtype=N.double)
        Xw = N.zeros((ns,M-1),dtype=N.double)
        E = N.ones((ns,M-1),dtype=N.double)
        S = M*N.ones(ns,dtype=N.double)

        # not vebose for now... must get this to work with the pymvpa
        # verbose and debug systems
        verbosity = int( "SMLR_" in debug.active )

        seed = None

        # call the chosen version of stepwise_regression
        cycles = _stepwise_regression(w,
                                      X,
                                      XY,
                                      Xw,
                                      E,
                                      auto_corr,
                                      lambda_over_2_auto_corr,
                                      S,
                                      self.__maxiter,
                                      self.__convergence_tol,
                                      verbosity,
                                      seed)

        if cycles >= self.__maxiter:
            # did not converge
            raise ConvergenceError, \
                  "More than %d Iterations without convergence" % \
                  (self.__maxiter)

        # save the weights
        self.__weights = w

        # save the weights state
        self.weights = w

        if __debug__:
            debug('SMLR_', 'train finished in %s cycles on data.shape=%s min:max(data)=%f:%f, got min:max(w)=%f:%f' %
                  (`cycles`, `X.shape`, N.min(X), N.max(X),
                   N.min(w), N.max(w)))


    def _predict(self, data):
        """
        Predict the output for the provided data.
        """
        # see if we are adding a bias term
        if self.__bias:
            # append the bias term to the features
            data = N.hstack((data,N.ones((data.shape[0],1),dtype=data.dtype)))

        # append the zeros column to the weights
        w = N.hstack((self.__weights,N.zeros((self.__weights.shape[0],1))))

        # determine the probability values for making the prediction
        dot_prod = N.dot(data,w)
        E = N.exp(dot_prod)
        S = N.sum(E, 1)

        if __debug__:
            debug('SMLR_', 'predict on data.shape=%s min:max(data)=%f:%f min:max(w)=%f:%f min:max(dot_prod)=%f:%f min:max(E)=%f:%f' %
                  (`data.shape`, N.min(data), N.max(data),
                   N.min(w), N.max(w),
                   N.min(dot_prod), N.max(dot_prod),
                   N.min(E), N.max(E)))
            
        values = E / S[:,N.newaxis].repeat(E.shape[1],axis=1)
        self.values = values

        # generate predictions
        predictions = N.asarray([self.__ulabels[N.argmax(vals)] for vals in values])
        self.predictions = predictions
        
        return predictions

