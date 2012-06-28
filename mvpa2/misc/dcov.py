# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Compute dcov/dcorr measures for independence testing

References
----------

http://en.wikipedia.org/wiki/Distance_covariance

"""

"""
TODO: consider use of  numexpr to speed all those up -- there is plenty of temp storage
"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import warning, externals

if externals.exists('cran-energy'):
    import rpy2.robjects
    import rpy2.robjects.numpy2ri
    if hasattr(rpy2.robjects.numpy2ri,'activate'):
        rpy2.robjects.numpy2ri.activate()
    RRuntimeError = rpy2.robjects.rinterface.RRuntimeError
    r = rpy2.robjects.r
    r.library('energy')


def _euclidean_distances(x, uv):
    """Compute euclidean distances for samples in columns

    Helper function for dcov computations

    Parameters
    ----------

    uv : bool, optional
      if True, then for each observable distance computed separately
      from the others.  If not -- then distances computed for
      multivariate patterns and output as observable == 1

    output observable x sample x sample
    """
    # TODO: could possibly be optimized to not compute the same i,j
    # and i,i distance twice but I wanted to avoid any explicit Python
    # loop here
    dx = x[:, None, :] - x[:, :, None]
    if uv:
        return np.sqrt(np.square(dx))
    else:
        return np.sqrt(np.sum(np.square(dx), axis=0))[None,:]


def _Aij(d):
    """Given distances matrix observable x sample x sample
    return normalized one where means get subtracted
    """
    mean_i = np.mean(d, axis=1)
    mean_j = np.mean(d, axis=2)
    mean_ij = np.mean(mean_i, axis=1)
    # ain't broadcasting is cool?
    return d - mean_i[:, None] - mean_j[:, :, None] + mean_ij[:, None, None]

if externals.exists('cran-energy'):
    def dCOV_R(x, y, uv=False, all_est=True):
        """Implementation of dCOV interfaced to original R energy library -- used primarily for testing
        """
        # trust no one!
        if uv:
            N = len(x)
            M = len(y)
            dCovs = np.zeros((N, M))
            dCors = np.zeros((N, M))
            Varx = np.zeros((N,))
            Vary = np.zeros((M,))
            for ix, x_ in enumerate(x):
                for iy, y_ in enumerate(y):
                    out = r.DCOR(x_, y_)
                    #outr = r.dcor(x_, y_)
                    #outv = r.dcov(x_, y_)
                    dCovs[ix, iy] = out[0][0]
                    dCors[ix, iy] = out[1][0]
                    Varx[ix] = out[2][0]
                    Vary[iy] = out[3][0]
            outputs = dCovs, dCors, Varx, Vary
        else:
            out = r.DCOR(x.T, y.T)
            outputs = tuple([o[0] for o in out])

        if not all_est:
            outputs = outputs[:1]

        if uv:
            return outputs
        else:
            # return corresponding scalars if it was a multivariate estimate
            return tuple(np.asscalar(np.asanyarray(x)) for x in outputs)


def dCOV(x, y, rowvar=1, uv=False, all_est=True):
    """Estimate dCov measure(s) between x and y.  Allows uni- or multi-variate estimations

    Name dCOV was chosen to match implementation in R energy toolbox:
    http://cran.r-project.org/web/packages/energy/index.html

    Parameters
    ----------
    rowvar : int, optional
        If `rowvar` is 1 (default), then each row represents a
        variable, with observations in the columns.  If 0, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    uv : bool, optional
        dCov is a multivariate measure of dependence so it would
        produce a single estimate for two matrices NxT and MxT.
        With uv=True (univariate estimation) it will return estimates
        for every pair of variables from x and y, thus NxM matrix,
        somewhat similar to what numpy.corrcoef does besides not estimating
        within x or y
    all_est : bool, True
        Since majority of computation of dCor(x,y), dVar(x) and
        dVar(y) is spend while estimating dVar(x, y) it makes sense to
        estimate all of them at the same time if any of the later is
        necessary.  So output would then consist of dCov, dCor, dVar(x),
        dVar(y) tuple, matching the order of energy toolbox dCOV output
        in R.

    """
    # Assure that we have correct dimensionality
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if rowvar == 0:
        # operate on transposes
        x = x.T
        y = y.T
    elif rowvar == 1:
        pass                            # default mode
    else:
        raise ValueError("rowvar must be either 0 (samples are rows) "
                         "or 1 (observables are rows). Got %d" % rowvar)

    # number of samples
    nsamples = x.shape[1]
    assert(nsamples == y.shape[1])

    if nsamples < 3:
        warning("You are trying to estimate dCov on %d sample(s). "
                "Please verify correctness of input" % nsamples)
    Dx = _euclidean_distances(x, uv=uv)
    Dy = _euclidean_distances(y, uv=uv)

    N, M = len(Dx), len(Dy)
    # .reshape is here to combine TxT into a single T**2 dimension to ease sums
    Ax = _Aij(Dx).reshape((N, -1))
    Ay = _Aij(Dy).reshape((M, -1))

    # and once again use cool broadcasting although at the cost of
    # memory since per se temporary storage is not necessary
    Axy = Ax[:, None] * Ay[None, :]
    dCov = np.sqrt(np.mean(Axy, axis=2))

    if not all_est:
        outputs = (dCov,)
    else:
        # if all estimates were requested -- be so
        dVar_x = np.sqrt(np.mean(np.square(Ax), axis=1))
        dVar_y = np.sqrt(np.mean(np.square(Ay), axis=1))
        dVar_xy = np.sqrt(dVar_x[:, None] * dVar_y[None, :])
        dCor = np.zeros(shape=dCov.shape)
        # So that we do not / 0.  R's dCOV seems to return 0s for
        # those cases, so we will
        dVar_xy_nz = dVar_xy.nonzero()
        dCor[dVar_xy_nz] = dCov[dVar_xy_nz] / dVar_xy[dVar_xy_nz]

        outputs = dCov, dCor, dVar_x, dVar_y

    if uv:
        return outputs
    else:
        # return corresponding scalars if it was a multivariate estimate
        return tuple(np.asscalar(x) for x in outputs)


def dcorcoef(x, y,  rowvar=1, uv=False):
    """Return dCor coefficient(s) only (convenience function).

    See :func:`dCOV` for more information
    """
    _, dCor, _, _ = dCOV(x, y, rowvar=rowvar, uv=uv, all_est=True)
    return dCor


