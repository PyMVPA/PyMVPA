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

# Uber-fast C-version of the stepwise regression
from mvpa.clfs.libsmlr import stepwise_regression as _cStepwiseRegression

if __debug__:
    from mvpa.misc import debug

def _label2oneofm(labels, ulabels):
    """Convert labels to one-of-M form.

    TODO: Might be useful elsewhere so could migrate into misc/
    """

    # allocate for the new one-of-M labels
    new_labels = N.zeros((len(labels), len(ulabels)))

    # loop and convert to one-of-M
    for i, c in enumerate(ulabels):
        new_labels[labels == c, i] = 1

    return new_labels



class SMLR(Classifier):
    """Sparse Multinomial Logistic Regression `Classifier`.

    This is an implementation of the SMLR algorithm published in
    Krishnapuram et al. (2005, IEEE Transactions on Pattern Analysis
    and Machine Intelligence).  Be sure to cite that article if you
    use this for your work.
    """

    def __init__(self, lm=.1, convergence_tol=1e-3, resamp_decay=.5,
                 min_resamp=.001, maxiter=10000, has_bias=True,
                 fit_all_weights=False, implementation="C", 
                 seed=None, **kwargs):
        """
        Initialize an SMLR classifier.

        :Parameters:
          lm : float
            The penalty term lambda.  Larger values will give rise
            to more sparsification.
          covergence_tol : float
            When the weight change for each cycle drops below this value
            the regression is considered converged.  Smaller values
            lead to tighter convergence.
          resamp_decay : float
            Rate of decay in the probability of resampling a zero weight.
            1.0 will immediately decrease to the min_resamp from 1.0,
            0.0 will never decrease from 1.0.
          min_resamp : float
            Minimum resampling probability for zeroed weights.
          maxiter : int
            Maximum number of iterations before stopping if not converged.
          has_bias : bool
            Whether to add a bias term to allow fits to data not through
            zero.
          fit_all_weights : bool
            Whether to fit weights for all classes or to the number of 
            classes minus one.  Both should give nearly identical results, but
            if you set fit_all_weights to True it will take a little longer
            and yield weights that are fully analyzable for each class.  Also,
            note that the convergence rate may be different, but convergence
            point is the same.
          implementation : basestr
            Use C (default) or Python as the implementation of
            stepwise_regression. C version brings significant speedup thus
            is the default one.
          seed : int or None
            Seed to be used to initialize random generator, might be used to
            replicate the run

        TODO:
         # Add in likelihood calculation
         # Add kernels, not just direct methods.
        """
        # init base class first
        Classifier.__init__(self, **kwargs)

        # set the parameters
        self.__lm = lm
        self.__convergence_tol = convergence_tol
        self.__resamp_decay = resamp_decay
        self.__min_resamp = min_resamp
        self.__maxiter = maxiter
        self.__has_bias = has_bias
        self.__fit_all_weights = fit_all_weights
        self.__seed = seed

        # pylint friendly initializations
        self.__ulabels = None
        """Unigue labels from the training set."""
        self.__weights_all = None
        """Contains all weights including bias values"""
        self.__weights = None
        """Just the weights, without the biases"""
        self.__biases = None
        """The biases, will remain none if has_bias is False"""

        if not implementation.upper() in ['C', 'PYTHON']:
            raise ValueError, \
                  "Unknown implementation %s of stepwise_regression" % \
                  implementation

        self.__implementation = implementation


    def __repr__(self):
        """String representation of the object
        """
        return "SMLR(lm=%f, convergence_tol=%g, " % \
               (self.__lm, self.__convergence_tol) + \
               "resamp_decay=%f, min_resamp=%f, maxiter=%d, " % \
               (self.__resamp_decay, self.__min_resamp, self.__maxiter) + \
               "has_bias=%s, fit_all_weights=%s, implementation='%s', " % \
               (self.__has_bias, self.__fit_all_weights, self.__implementation) + \
               "seed=%s, enabled_states=%s)" % \
               (self.__seed, str(self.states.enabled))


    def _pythonStepwiseRegression(self, w, X, XY, Xw, E,
                                  auto_corr,
                                  lambda_over_2_auto_corr,
                                  S,
                                  M,
                                  maxiter,
                                  convergence_tol,
                                  resamp_decay,
                                  min_resamp,
                                  verbose,
                                  seed = None):
        """The (much slower) python version of the stepwise
        regression.  I'm keeping this around for now so that we can
        compare results."""

        # get the data information into easy vars
        ns, nd = X.shape

        # initialize the iterative optimization
        converged = False
        incr = N.finfo(N.float).max
        non_zero, basis, m, wasted_basis, cycles = 0, 0, 0, 0, 0
        sum2_w_diff, sum2_w_old, w_diff = 0.0, 0.0, 0.0
        p_resamp = N.ones(w.shape, dtype=N.float)

        # set the random seed
        N.random.seed(seed)

        if __debug__:
            debug("SMLR_", "random seed=%s" % seed)

        # perform the optimization
        while not converged and cycles < maxiter:
            # get the starting weight
            w_old = w[basis, m]

            # see if we're gonna update
            if (w_old != 0) or N.random.rand() < p_resamp[basis, m]:
                # let's do it
                # get the probability
                P = E[:, m]/S

                # set the gradient
                grad = XY[basis, m] - N.dot(X[:, basis], P)

                # calculate the new weight with the Laplacian prior
                w_new = w_old + grad/auto_corr[basis]

                # keep weights within bounds
                if w_new > lambda_over_2_auto_corr[basis]:
                    w_new -= lambda_over_2_auto_corr[basis]
                    changed = True
                    # unmark from being zero if necessary
                    if w_old == 0:
                        non_zero += 1
                        # reset the prob of resampling
                        p_resamp[basis, m] = 1.0
                elif w_new < -lambda_over_2_auto_corr[basis]:
                    w_new += lambda_over_2_auto_corr[basis]
                    changed = True
                    # unmark from being zero if necessary
                    if w_old == 0:
                        non_zero += 1
                        # reset the prob of resampling
                        p_resamp[basis, m] = 1.0
                else:
                    # gonna zero it out
                    w_new = 0.0

                    # decrease the p_resamp
                    p_resamp[basis, m] -= (p_resamp[basis, m] - \
                                           min_resamp) * resamp_decay

                    # set number of non-zero
                    if w_old == 0:
                        changed = False
                        wasted_basis += 1
                    else:
                        changed = True
                        non_zero -= 1

                # process any changes
                if changed:
                    #print "w[%d, %d] = %g" % (basis, m, w_new)
                    # update the expected values
                    w_diff = w_new - w_old
                    Xw[:, m] = Xw[:, m] + X[:, basis]*w_diff
                    E_new_m = N.exp(Xw[:, m])
                    S += E_new_m - E[:, m]
                    E[:, m] = E_new_m

                    # update the weight
                    w[basis, m] = w_new

                    # keep track of the sqrt sum squared diffs
                    sum2_w_diff += w_diff*w_diff

                # add to the old no matter what
                sum2_w_old += w_old*w_old

            # update the class and basis
            m = N.mod(m+1, w.shape[1])
            if m == 0:
                # we completed a cycle of labels
                basis = N.mod(basis+1, nd)
                if basis == 0:
                    # we completed a cycle of features
                    cycles += 1

                    # assess convergence
                    incr = N.sqrt(sum2_w_diff) / \
                           (N.sqrt(sum2_w_old)+N.finfo(N.float).eps)

                    # save the new weights
                    converged = incr < convergence_tol

                    if __debug__:
                        debug("SMLR_", \
                              "cycle=%d ; incr=%g ; non_zero=%d ; " %
                              (cycles, incr, non_zero) +
                              "wasted_basis=%d ; " %
                              (wasted_basis) +
                              "sum2_w_old=%g ; sum2_w_diff=%g" %
                              (sum2_w_old, sum2_w_diff))

                    # reset the sum diffs and wasted_basis
                    sum2_w_diff = 0.0
                    sum2_w_old = 0.0
                    wasted_basis = 0


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

#        print 'cycles=%d ; wasted basis=%g\n' % (cycles, wasted_basis/((M-1)*nd))


    def _train(self, dataset):
        """Train the classifier using `dataset` (`Dataset`).
        """
        # Process the labels to turn into 1 of N encoding
        labels = _label2oneofm(dataset.labels, dataset.uniquelabels)
        self.__ulabels = dataset.uniquelabels.copy()

        Y = labels
        M = len(self.__ulabels)

        # get the dataset information into easy vars
        X = dataset.samples

        # see if we are adding a bias term
        if self.__has_bias:
            if __debug__:
                debug("SMLR_", "hstacking 1s for bias")

            # append the bias term to the features
            X = N.hstack((X, N.ones((X.shape[0], 1), dtype=X.dtype)))

        if self.__implementation.upper() == 'C':
            _stepwise_regression = _cStepwiseRegression
            #
            # TODO: avoid copying to non-contig arrays, use strides in ctypes?
            if not (X.flags['C_CONTIGUOUS'] and X.flags['ALIGNED']):
                if __debug__:
                    debug("SMLR_",
                          "Copying data to get it C_CONTIGUOUS/ALIGNED")
                X = N.array(X, copy=True, dtype=N.double, order='C')

            # currently must be double for the C code
            if X.dtype != N.double:
                if __debug__:
                    debug("SMLR_", "Converting data to double")
                # must cast to double
                X = X.astype(N.double)

        # set the feature dimensions
        elif self.__implementation.upper() == 'PYTHON':
            _stepwise_regression = self._pythonStepwiseRegression
        else:
            raise ValueError, \
                  "Unknown implementation %s of stepwise_regression" % \
                  self.__implementation

        # set the feature dimensions
        ns, nd = X.shape

        # decide the size of weights based on num classes estimated
        if self.__fit_all_weights:
            c_to_fit = M
        else:
            c_to_fit = M-1

        # Precompute what we can
        auto_corr = ((M-1.)/(2.*M))*(N.sum(X*X, 0))
        XY = N.dot(X.T, Y[:, :c_to_fit])
        lambda_over_2_auto_corr = (self.__lm/2.)/auto_corr

        # set starting values
        w = N.zeros((nd, c_to_fit), dtype=N.double)
        Xw = N.zeros((ns, c_to_fit), dtype=N.double)
        E = N.ones((ns, c_to_fit), dtype=N.double)
        S = M*N.ones(ns, dtype=N.double)

        # set verbosity
        if __debug__:
            verbosity = int( "SMLR_" in debug.active )
            debug('SMLR_', 'Calling stepwise_regression. Seed %s' % self.__seed)
        else:
            verbosity = 0

        # call the chosen version of stepwise_regression
        cycles = _stepwise_regression(w,
                                      X,
                                      XY,
                                      Xw,
                                      E,
                                      auto_corr,
                                      lambda_over_2_auto_corr,
                                      S,
                                      M,
                                      self.__maxiter,
                                      self.__convergence_tol,
                                      self.__resamp_decay,
                                      self.__min_resamp,
                                      verbosity,
                                      self.__seed)

        if cycles >= self.__maxiter:
            # did not converge
            raise ConvergenceError, \
                  "More than %d Iterations without convergence" % \
                  (self.__maxiter)

        # save the weights
        self.__weights_all = w
        self.__weights = w[:dataset.nfeatures, :]

        if self.states.isEnabled('feature_ids'):
            self.feature_ids = N.where(N.max(N.abs(w[:dataset.nfeatures,:]), axis=1)>0)[0]

        # and a bias
        if self.__has_bias:
            self.__biases = w[-1, :]

        if __debug__:
            debug('SMLR', "train finished in %s cycles on data.shape=%s " %
                  (`cycles`, `X.shape`) +
                  "min:max(data)=%f:%f, got min:max(w)=%f:%f" %
                  (N.min(X), N.max(X), N.min(w), N.max(w)))


    def _predict(self, data):
        """
        Predict the output for the provided data.
        """
        # see if we are adding a bias term
        if self.__has_bias:
            # append the bias term to the features
            data = N.hstack((data, 
                             N.ones((data.shape[0], 1), dtype=data.dtype)))

        # append the zeros column to the weights if necessary
        if self.__fit_all_weights:
            w = self.__weights_all
        else:
            w = N.hstack((self.__weights_all, 
                          N.zeros((self.__weights_all.shape[0], 1))))

        # determine the probability values for making the prediction
        dot_prod = N.dot(data, w)
        E = N.exp(dot_prod)
        S = N.sum(E, 1)

        if __debug__:
            debug('SMLR', "predict on data.shape=%s min:max(data)=%f:%f " %
                  (`data.shape`, N.min(data), N.max(data)) +
                  "min:max(w)=%f:%f min:max(dot_prod)=%f:%f min:max(E)=%f:%f" %
                  (N.min(w), N.max(w), N.min(dot_prod), N.max(dot_prod),
                   N.min(E), N.max(E)))

        values = E / S[:, N.newaxis].repeat(E.shape[1], axis=1)
        self.values = values

        # generate predictions
        predictions = N.asarray([self.__ulabels[N.argmax(vals)]
                                 for vals in values])
        # no need to assign state variable here -- would be done in Classifier._postpredict anyway
        #self.predictions = predictions

        return predictions

    has_bias = property(lambda self: self.__has_bias)
    biases = property(lambda self: self.__biases)
    weights = property(lambda self: self.__weights)
