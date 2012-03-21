# -*- coding: iso-8859-15 -*-

"""PDF of the multivariate P�lya distribution.

See: http://en.wikipedia.org/wiki/Multivariate_P%C3%B3lya_distribution
"""

import numpy as np
from scipy import factorial, comb as binomial_coefficient
from scipy.special import gamma, gammaln
from numpy.random import dirichlet
from logvar import logmean

def multivariate_polya(x, alpha):
    """Multivariate P�lya PDF. Basic implementation.
    """
    x = np.atleast_1d(x).flatten()
    alpha = np.atleast_1d(alpha).flatten()
    assert(x.size==alpha.size)
    N = x.sum()
    A = alpha.sum()
    likelihood = factorial(N) * gamma(A) / gamma(N + A)
    # likelihood = gamma(A) / gamma(N + A)
    for i in range(len(x)):
        likelihood /= factorial(x[i])
        likelihood *= gamma(x[i] + alpha[i]) / gamma(alpha[i])
    return likelihood

def multivariate_polya_vectorized(x,alpha):
    """Multivariate P�lya PDF. Vectorized implementation.
    """
    x = np.atleast_1d(x)
    alpha = np.atleast_1d(alpha)
    assert(x.size==alpha.size)
    N = x.sum()
    A = alpha.sum()
    likelihood = factorial(N) / factorial(x).prod() * gamma(A) / gamma(N + A)
    likelihood *= (gamma(x + alpha) / gamma(alpha)).prod()
    return likelihood


def log_multivariate_polya_vectorized(X, alpha):
    """Multivariate P�lya log PDF. Vectorized and stable implementation.
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


def log_multivariate_polya_mc(X, alpha, iterations=1e5):
    """Montecarlo estimation of the log-likelihood of the Dirichlet
    compound multinomial (DCM) distribution, a.k.a. the multivariate
    Polya distribution.
    """
    Theta = dirichlet(alpha, size=int(iterations))
    logp_Hs = gammaln(X.sum() + 1) - gammaln(X + 1).sum()
    logp_Hs += (X * np.log(Theta)).sum(1)

    return logmean(logp_Hs)
    

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

