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
    population mean, `popmean`.  It adds ability to carry single
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


def binomial_proportion_ci(n, X, alpha=.05, meth='jeffreys'):
    """Compute the confidence interval for a set of Bernoulli trials

    Most, if not all, implemented methods assume statistical independence
    of the Bernoulli trial outcomes. Computed confidence intervals
    may be invalid if this condition is violated.

    This is a re-implementation of Matlab code originally written by
    Anderson Winkler and Tom Nichols.

    Parameters
    ----------
    n : int
      Number of trials
    X : int or array
      Number of successful trials. This can be a 1D array.
    alpha : float
      Coverage of the confidence interval. For a 95% CI (default), use
      alpha = 0.05.
    meth : {'wald', 'wilson', 'agresti-coull', 'jeffreys', 'clopper-pearson', 'arc-sine', 'logit', 'anscombe'}
      Interval estimation method.

    Returns
    -------
    2-item array or 2D array
      With the lower and upper bound for the confidence interval. If X was given
      as a vector with p items a 2xp array is returned.

    References
    ----------
    .. [1] Brown LD, Cai TT, DasGupta AA. Interval estimation for a
       binomial proportion. Statistical Science. 2001 16(2):101-133.
       http://brainder.org/2012/04/21/confidence-intervals-for-bernoulli-trials
    """

    from scipy import stats
    from numpy import sqrt, sin, arcsin, log, exp

    n = float(n)
    X = np.asanyarray(X, dtype=float)
    k  = stats.norm.ppf(1 - alpha / 2.)
    p  = X / n          # Proportion of successes
    q  = 1 - p          # Proportion of failures
    Xt = X + (k**2) / 2 # Modified number of sucesses
    nt = n + k**2       # Modified number of trials
    pt = Xt / nt        # Modified proportion of successes
    qt = 1 - pt         # Modified proportion of failures

    # be tolerant
    meth = meth.lower()
    if meth == 'wald':
        L = p - k * sqrt(p * q / n)
        U = p + k * sqrt(p * q / n)
    elif meth == 'wilson':
        a = k * sqrt(n * p * q + (k**2) / 4) / nt
        L = pt - a
        U = pt + a
    elif meth == 'agresti-coull':
        a = k * sqrt(pt * qt / nt)
        L = pt - a
        U = pt + a
    elif meth == 'jeffreys':
        L = stats.beta.ppf(    alpha / 2, X + .5, n - X + .5)
        U = stats.beta.ppf(1 - alpha / 2, X + .5, n - X + .5)
    elif meth == 'clopper-pearson':
        L = stats.beta.ppf(    alpha / 2, X,     n - X + 1)
        U = stats.beta.ppf(1 - alpha / 2, X + 1, n - X)
    elif meth == 'arc-sine':
        pa = (X + 3 / 8) / (n + 3 / 4)
        as_ = arcsin(sqrt(pa))
        a = k / (2 * sqrt(n))
        L  = sin(as_ - a)**2
        U  = sin(as_ + a)**2
    elif meth == 'logit':
        lam  = log(X / (n - X))
        sqVhat = sqrt(n / (X * (n - X)))
        exlamL = exp(lam - k * sqVhat)
        exlamU = exp(lam + k * sqVhat)
        L    = exlamL / (1 + exlamL)
        U    = exlamU / (1 + exlamU)
    elif meth == 'anscombe':
        lam  = log((X + .5) / (n - X + .5))
        sqVhat = sqrt((n + 1) * (n + 2) / (n * (X + 1) * (n - X + 1)))
        exlamL = exp(lam - k * sqVhat)
        exlamU = exp(lam + k * sqVhat)
        L    = exlamL / (1 + exlamL)
        U    = exlamU / (1 + exlamU)
    else:
        raise ValueError('unknown confidence interval method')

    return np.array((L, U))


def binomial_proportion_ci_from_bool(arr, axis=0, *args, **kwargs):
    """Convenience wrapper for ``binomial_proportion_ci()`` with boolean input

    Parameters
    ----------
    arr : array
      Boolean array
    axis : int
    *args, **kwargs
      All other arguments are passed on to binomial_proportion_ci().
    """
    return binomial_proportion_ci(arr.shape[axis], np.sum(arr, axis=axis),
                                  *args, **kwargs)


def _mask_nan(x):
    return np.ma.masked_array(x, np.isnan(x))

def compute_ts_boxplot_stats(data, outlier_abs_minthresh=None,
                             outlier_thresh=3.0, greedy_outlier=False,
                             aggfx=None, *args):
    """Compute boxplot-like statistics across a set of time series.

    This function can handle missing values and supports data aggregation.

    Parameters
    ----------
    data : array
      Typically a 2-dimensional array (series x samples). Multi-feature samples
      are supported (series x samples x features), but they have to be
      aggregated into a scalar. See ``aggfx``.
    outlier_abs_minthresh : float or None
      Absolute minimum threshold of outlier detection. Only value larger than
      this this threshold will ever be considered as an outlier
    outlier_thresh : float or None
      Outlier classification threshold in units of standard deviation.
    greedy_outlier : bool
      If True, an entire time series is marked as an outlier, if any of its
      observations matches the criterion. If False, only individual observations
      are marked as outlier.
    aggfx : functor or None
      Aggregation function used to collapse multi-feature samples into a scalar
      value
    *args :
      Additional arguments for ``aggfx``.

    Returns
    -------
    tuple
      This 2-item tuple contains all computed statistics in the first item and
      all classified outliers in the second item. Statistics are computed for
      each time series observation across time series. Available information:
      mean value, median, standard deviation, minimum, maximum, 25% and 75%
      percentile, as well as number of non-outlier data points for each sample.
      The outlier data points are returned a masked array of the same size as
      the input data. All data points classified as non-outliers are masked.
    """
    if len(data) < 2:
        raise ValueError("needs at least two time series")
    # data comes in as (subj x volume x parameter)
    orig_input = data
    # reduce data to L2-norm
    if aggfx is not None:
        data = np.apply_along_axis(aggfx, 2, data, *args)
    # need to deal with missing data
    data = _mask_nan(np.asanyarray(data))
    if len(data.shape) < 2:
        raise ValueError("needs at least two observation per time series")
    # outlier detection
    meand = np.ma.mean(data, axis=0)
    stdd = np.ma.std(data, axis=0)
    outlierd = None
    if outlier_thresh > 0.0:
        # deal properly with NaNs so that they are not considered outliers
        outlier = np.logical_and(np.logical_not(np.isnan(data)),
                                 np.ma.greater(
                                        (np.absolute(data - meand)),
                                        outlier_thresh * stdd))

        if outlier_abs_minthresh is not None:
            # apply absolute filter in addition
            outlier = np.logical_and(outlier,
                                     np.ma.greater(data,
                                                   outlier_abs_minthresh))
        if greedy_outlier:
            # expect outlier mask to all elements in that series
            outlier[np.sum(outlier, axis=1) > 0] = True
        # apply outlier mask to original data, but merge with existing mask
        # to keep NaNs out of the game
        data = np.ma.masked_array(data.data,
                                  mask=np.logical_or(data.mask, outlier))
        outlierd = np.ma.masked_array(data.data,
                                      mask=np.logical_not(outlier))

    res = {}
    res['mean'] = np.ma.mean(data, axis=0)
    res['median'] = np.ma.median(data, axis=0)
    res['std'] = np.ma.std(data, axis=0)
    res['min'] = np.ma.min(data, axis=0)
    res['max'] = np.ma.max(data, axis=0)
    res['p75'] = np.percentile(data, 75, axis=0)
    res['p25'] = np.percentile(data, 25, axis=0)
    res['n'] = len(data) - data.mask.sum(axis=0)
    return res, outlierd
