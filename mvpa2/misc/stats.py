# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Little statistics helper"""

__docformat__ = 'restructuredtext'

from mvpa2.base import externals

if externals.exists('scipy', raise_=True):
    import scipy.stats as st
    # evaluate once the fact of life
    __scipy_prior0101 = externals.versions['scipy'] < '0.10.1'

import numpy as np
import copy

def chisquare(obs, exp='uniform'):
    """Compute the chisquare value of a contingency table with arbitrary
    dimensions.

    Parameters
    ----------
    obs : array
      Observations matrix
    exp : ('uniform', 'indep_rows') or array, optional
      Matrix of expected values of the same size as `obs`.  If no
      array is given, then for 'uniform' -- evenly distributes all
      observations.  In 'indep_rows' case contingency table takes into
      account frequencies relative across different columns, so, if
      the contingency table is predictions vs targets, it would
      account for dis-balance among different targets.  Although
      'uniform' is the default, for confusion matrices 'indep_rows' is
      preferable.

    Returns
    -------
    tuple
     chisquare-stats, associated p-value (upper tail)
    """
    obs = np.array(obs)

    # get total number of observations
    nobs = np.sum(obs)

    # if no expected value are supplied assume equal distribution
    if not isinstance(exp, np.ndarray):
        ones = np.ones(obs.shape, dtype=float)
        if exp == 'indep_rows':
            # multiply each column
            exp = np.sum(obs, axis=0)[None, :] * ones / obs.shape[0]
        elif exp == 'indep_cols':
            # multiply each row
            exp = np.sum(obs, axis=1)[:, None] * ones / obs.shape[1]
        elif exp == 'uniform':
            # just evenly distribute
            exp = nobs * np.ones(obs.shape, dtype=float) / np.prod(obs.shape)
        else:
            raise ValueError, \
                  "Unknown specification of expected values exp=%r" % (exp,)
    else:
        assert(exp.shape == obs.shape)

    # make sure to have floating point data
    exp = exp.astype(float)

    # compute chisquare value
    exp_zeros = exp == 0
    exp_nonzeros = np.logical_not(exp_zeros)
    if np.sum(exp_zeros) != 0 and (obs[exp_zeros] != 0).any():
        raise ValueError, \
              "chisquare: Expected values have 0-values, but there are actual" \
              " observations -- chi^2 cannot be computed"
    chisq = np.sum(((obs - exp) ** 2)[exp_nonzeros] / exp[exp_nonzeros])

    # return chisq and probability (upper tail)
    # taking only the elements with something expected
    return chisq, st.chisqprob(chisq, np.sum(exp_nonzeros) - 1)


def _chk_asanyarray(a, axis):
    a = np.asanyarray(a)
    if axis is None:
        a = a.ravel()
        outaxis = 0
    else:
        outaxis = axis
    return a, outaxis


def ttest_1samp(a, popmean=0, axis=0, mask=None, alternative='two-sided'):
    """
    Calculates the T-test for the mean of ONE group of scores `a`.

    This is a refinement for the :func:`scipy.stats.ttest_1samp` for
    the null hypothesis testing that the expected value (mean) of a
    sample of independent observations is equal to the given
    population mean, `popmean`.  It adds ability to test carry single
    tailed test as well as operate on samples with varying number of
    active measurements, as specified by `mask` argument.

    Since it is only a refinement and otherwise it should perform the
    same way as the original ttest_1samp -- the name was overloaded.

    Note
    ----

    Initially it was coded before discovering scipy.mstats which
    should work with masked arrays.  But ATM (scipy 0.10.1) its
    ttest_1samp does not support axis argument making it of limited
    use anyways.


    Parameters
    ----------
    a : array_like
        sample observations
    popmean : float or array_like
        expected value in null hypothesis, if array_like than it must have the
        same shape as `a` excluding the axis dimension
    axis : int, optional, (default axis=0)
        Axis can equal None (ravel array first), or an integer (the axis
        over which to operate on a).
    mask : array_like, bool
        bool array to specify which measurements should participate in the test
    alternative : ('two-sided', 'greater', 'less')
        alternative two test

    Returns
    -------
    t : float or array
        t-statistic
    prob : float or array
        p-value

    Examples
    --------
    TODO

    """

    # would also flatten if no axis specified
    a, axis = _chk_asanyarray(a, axis)

    if isinstance(a, np.ma.core.MaskedArray):
        if mask is not None:
            raise ValueError(
                "Provided array is already masked, so no additional "
                "mask should be provided")
        n = a.count(axis=axis)
    elif mask is not None:
        # Create masked array
        a = np.ma.masked_array(a, mask= ~np.asanyarray(mask))
        n = a.count(axis=axis)
    else:
        # why bother doing anything?
        n = a.shape[axis]

    df = n - 1

    d = np.mean(a, axis) - popmean
    # yoh: there is a bug in old (e.g. 1.4.1) numpy's while operating on
    #      masked arrays -- for some reason refuses to compute var
    #      correctly whenever only 2 elements are available and it is
    #      multi-dimensional:
    # (Pydb) print np.var(a[:, 9:11], axis, ddof=1)
    # [540.0 --]
    # (Pydb) print np.var(a[:, 10:11], axis, ddof=1)
    # [--]
    # (Pydb) print np.var(a[:, 10], axis, ddof=1)
    # 648.0
    v = np.var(a, axis, ddof=1)
    denom = np.sqrt(v / n)

    t = np.divide(d, denom)

    # t, prob might be full arrays if no masking was actually done
    def _filled(a):
        if isinstance(a, np.ma.core.MaskedArray):
            return a.filled(np.nan)
        else:
            return a

    t, prob = _ttest_finish(_filled(df), _filled(t), alternative=alternative)

    return t, prob


def _ttest_finish(df, t, alternative):
    """Common code between all 3 t-test functions."""
    dist_gen = st.distributions.t
    if alternative == 'two-sided':
        prob = dist_gen.sf(np.abs(t), df) * 2 # use np.abs to get upper alternative
    elif alternative == 'greater':
        prob = dist_gen.sf(t, df)
    elif alternative == 'less':
        prob = dist_gen.cdf(t, df)
    else:
        raise ValueError("Unknown alternative %r" % alternative)

    t_isnan = np.isnan(t)
    if np.any(t_isnan) and __scipy_prior0101:
        # older scipy's would return 0 for nan values of the argument
        # which is incorrect
        if np.isscalar(prob):
            prob = np.nan
        else:
            prob[t_isnan] = np.nan

    if t.ndim == 0:
        t = np.asscalar(t)

    return t, prob
