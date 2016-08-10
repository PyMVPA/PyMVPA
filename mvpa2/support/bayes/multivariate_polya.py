# emacs: coding: utf-8; -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PDF of the multivariate Pólya distribution.

See: http://en.wikipedia.org/wiki/Multivariate_P%C3%B3lya_distribution
"""

import numpy as np
from scipy.misc import factorial
from scipy.special import gamma, gammaln


def multivariate_polya(x, alpha):
    """Multivariate Pólya PDF. Basic implementation.
    """
    x = np.atleast_1d(x).flatten()
    alpha = np.atleast_1d(alpha).flatten()
    assert(x.size==alpha.size)
    N = x.sum()
    A = alpha.sum()
    likelihood = factorial(N) * gamma(A) / gamma(N + A)
    # likelihood = gamma(A) / gamma(N + A)
    for i, xi in enumerate(x):
        likelihood /= factorial(xi)
        likelihood *= gamma(xi + alpha[i]) / gamma(alpha[i])
    return likelihood


def log_multivariate_polya_vectorized(X, alpha):
    """Multivariate Pólya log PDF. Vectorized and stable implementation.
    """
    X = np.atleast_1d(X)
    alpha = np.atleast_1d(alpha)
    assert(X.size==alpha.size)
    N = X.sum()
    A = alpha.sum()
    log_likelihood = gammaln(N+1) - gammaln(X+1).sum() # log(\frac{N!}{\prod_i (X_i)!})
    log_likelihood += gammaln(A) - gammaln(alpha).sum() # log(\frac{\Gamma(\sum_i alpha_i)}{\prod_i(\Gamma(\alpha_i))})
    log_likelihood += gammaln(X + alpha).sum() - gammaln(N + A) # log(\frac{\prod_i(\Gamma(X_i +\alpha_i))}{\Gamma(\sum_i X_i+\alpha_i)})
    return log_likelihood


if __name__ == '__main__':

    import numpy as np
    np.random.seed(0)

    # x = np.array([1,2,2,1])
    # x = np.array([6,0,0,0])
    # alpha = np.array([1,1,1,1])

    x = np.array([100,0,0])
    # x = np.array([0,50,50])
    alpha = np.array([1,10,10])
    print "x:", x
    print "alpha:", alpha
    print "Likelihood:"
    print "log of the basic formula:", np.log(multivariate_polya(x, alpha))
    print "log of the basic vectorized formula:", np.log(multivariate_polya_vectorized(x, alpha))
    print "Log-scale stable formula:", log_multivariate_polya_vectorized(x, alpha)
    print "Monte Carlo estimations in log-scale:"
    for i in range(5):
        print "\t", i, log_multivariate_polya_mc(x, alpha, iterations=1e5)

