"""Partial independence model: one margin fixed (rows margin).

Analytical solution + montecarlo checks.
"""

import numpy as np
from numpy.random import dirichlet
from multivariate_polya import log_multivariate_polya_vectorized as log_multivariate_polya
from scipy.special import gammaln
from logvar import logmean

def compute_logp_independent_block(X, alpha=None):
    """Compute the analytical log likelihood of a matrix under the
    assumption of independence.
    """
    if alpha is None: alpha = np.ones(X.shape[1])
    logp_ib = gammaln(alpha.sum()) - (gammaln(alpha)).sum()
    logp_ib += gammaln(X.sum(0) + alpha).sum() - gammaln(X.sum() + alpha.sum())
    logp_ib += gammaln(X.sum(1) + 1).sum() - gammaln(X + 1).sum()
    return logp_ib


def compute_logp_H(X, psi, alpha=None):
    """Compute the analytical log likelihood of the confusion matrix X
    with hyper-prior alpha (in a multivariate-Dirichlet sense)
    according to a partitioning scheme psi.
    """
    if alpha is None: alpha = np.ones(X.shape)
    logp_H = 0.0
    for group in psi:
        if len(group) == 1: logp_H += log_multivariate_polya(X[group[0],:], alpha[group[0],:])
        else:
            nogroup = filter(lambda a: a not in group, range(X.shape[1]))
            logp_H += np.sum([log_multivariate_polya([X[i,group].sum()] + X[i,nogroup].tolist(),
                                                     [alpha[i,group].sum()] + alpha[i,nogroup].tolist())
                              for i in group])
            logp_H += compute_logp_independent_block(X[np.ix_(group,group)],
                                                     alpha[np.ix_(group,group)].sum(0)) # should we use sum(0) or mean(0)? or else?
    return logp_H


def compute_logp_independent_block_mc(X, alpha=None, iterations=1e5):
    """Compute the montecarlo log likelihood of a matrix under the
    assumption of independence.
    """
    if alpha is None: alpha = np.ones(X.shape[1])
    Theta = dirichlet(alpha, size=int(iterations)).T
    logp_ibs = gammaln(X.sum(1)+1).sum() - gammaln(X+1).sum(1).sum() + (np.log(Theta[:,None,:])*X[:,:,None]).sum(1).sum(0) # log(\prod(one Multinomial pdf for each row))
    return logmean(logp_ibs)


if __name__ == '__main__':

    np.random.seed(0)

    X = np.array([[10,10, 0],
                  [10,10, 0],
                  [ 0, 0,20]])
    # X = np.array([[13, 3, 4],
    #               [ 3,14, 3],
    #               [ 4, 3,13]])
    # X = np.array([[10, 10, 10,  0,  0],
    #               [10, 10, 10,  0,  0],
    #               [10, 10, 10,  0,  0],
    #               [ 0,  0,  0, 30,  0],
    #               [ 0,  0,  0,  0, 30]], dtype=np.float32)

    print "X:"
    print X

    psi = [[0,1],[2]]
    # psi = [[0],[1],[2]]
    # psi = [[0],[1,2]]
    # psi = [[0,1,2]]
    # psi = [[0],[1],[2],[3],[4]]
    # psi = [[0,1],[2],[3],[4]]
    # psi = [[0,1,2],[3],[4]]
    # psi = [[0,1,2,3],[4]]
    # psi = [[0,1,2],[3,4]]
    # psi = [[0,1,2,3,4]]

    print "psi:", psi

    alpha = np.ones(X.shape)
    print "alpha:"
    print alpha
    
    logp_H = compute_logp_H(X, psi, alpha)

    print "Analytical estimate:", logp_H
