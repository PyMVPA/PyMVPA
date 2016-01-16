# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wrap models of the StatsModels package into a FeaturewiseMeasure."""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import externals

# do conditional to be able to build module reference
if externals.exists('scipy', raise_=True):
    from mvpa2.support.scipy.stats import scipy
    import scipy.stats as stats

if externals.exists('statsmodels', raise_=True):
    import statsmodels.api as sm

from mvpa2.measures.base import FeaturewiseMeasure
from mvpa2.datasets.base import Dataset

__all__ = [ 'UnivariateStatsModels', 'GLM' ]

class UnivariateStatsModels(FeaturewiseMeasure):
    """Adaptor for some models from the StatsModels package

    This adaptor allows for fitting several statistical models to univariate
    (in StatsModels terminology "endogeneous") data. A model, based on
    "exogeneous" data (i.e. a design matrix) and optional parameters, is fitted
    to each feature vector in a given dataset individually. The adaptor
    supports a variety of models provided by the StatsModels package, including
    simple ordinary least squares (OLS), generalized least squares (GLS) and
    others. This feature-wise measure can extract a variety of properties from
    the model fit results, and aggregate them into a result dataset. This
    includes, for example, all attributes of a StatsModels ``RegressionResult``
    class, such as model parameters and their error estimates, Aikake's
    information criteria, and a number of statistical properties. Moreover,
    it is possible to perform t-contrasts/t-tests of parameter estimates, as
    well as F-tests for contrast matrices.

    Examples
    --------
    Some example data: two features, seven samples

    >>> endog = Dataset(np.transpose([[1, 2, 3, 4, 5, 6, 8],
    ...                               [1, 2, 1, 2, 1, 2, 1]]))
    >>> exog = range(7)

    Set up a model generator -- it yields an instance of an OLS model for
    a particular design and feature vector. The generator will be called
    internally for each feature in the dataset.

    >>> model_gen = lambda y, x: sm.OLS(y, x)

    Configure the adaptor with the model generator and a common design for all
    feature model fits. Tell the adaptor to auto-add a constant to the design.
 
    >>> usm = UnivariateStatsModels(exog, model_gen, add_constant=True)

    Run the measure. By default it extracts the parameter estimates from the
    models (two per feature/model: regressor + constant).

    >>> res = usm(endog)
    >>> print res
    <Dataset: 2x2@float64, <sa: descr>>
    >>> print res.sa.descr
    ['params' 'params']

    Alternatively, extract t-values for a test of all parameter estimates
    against zero.

    >>> usm = UnivariateStatsModels(exog, model_gen, res='tvalues',
    ...                             add_constant=True)
    >>> res = usm(endog)
    >>> print res
    <Dataset: 2x2@float64, <sa: descr>>
    >>> print res.sa.descr
    ['tvalues' 'tvalues']

    Compute a t-contrast: first parameter is non-zero. This returns additional
    test statistics, such as p-value and effect size in the result dataset. The
    contrast vector is pass on to the ``t_test()`` function (``r_matrix``
    argument) of the StatsModels result class.

    >>> usm = UnivariateStatsModels(exog, model_gen, res=[1,0],
    ...                             add_constant=True)
    >>> res = usm(endog)
    >>> print res
    <Dataset: 6x2@float64, <sa: descr>>
    >>> print res.sa.descr
    ['tvalue' 'pvalue' 'effect' 'sd' 'df' 'zvalue']

    F-test for a contrast matrix, again with additional test statistics in the
    result dataset. The contrast vector is pass on to the ``f_test()`` function
    (``r_matrix`` argument) of the StatsModels result class.

    >>> usm = UnivariateStatsModels(exog, model_gen, res=[[1,0],[0,1]],
    ...                             add_constant=True)
    >>> res = usm(endog)
    >>> print res
    <Dataset: 4x2@float64, <sa: descr>>
    >>> print res.sa.descr
    ['fvalue' 'pvalue' 'df_num' 'df_denom']

    For any custom result extraction, a callable can be passed to the ``res``
    argument. This object will be called with the result of each model fit. Its
    return value(s) will be aggregated into a result dataset.

    >>> def extractor(res):
    ...     return [res.aic, res.bic]
    >>>
    >>> usm = UnivariateStatsModels(exog, model_gen, res=extractor,
    ...                             add_constant=True)
    >>> res = usm(endog)
    >>> print res
    <Dataset: 2x2@float64>

    """

    is_trained = True

    def __init__(self, exog, model_gen, res='params', add_constant=True,
                 **kwargs):
        """
        Parameters
        ----------
        exog : array-like
          Column ordered (observations in rows) design matrix.
        model_gen : callable
          Callable that returns a StatsModels model when called like
          ``model_gen(endog, exog)``.
        res : {'params', 'tvalues', ...} or 1d array or 2d array or callable
          Variable of interest that should be reported as feature-wise
          measure. If a str, the corresponding attribute of the model fit result
          class is returned (e.g. 'tvalues'). If a 1d-array, it is passed
          to the fit result class' ``t_test()`` function as a t-contrast vector.
          If a 2d-array, it is passed to the ``f_test()`` function as a
          contrast matrix.  In both latter cases a number of common test
          statistics are returned in the rows of the result dataset. A description
          is available in the 'descr' sample attribute. Any other datatype
          passed to this argument will be treated as a callable, the model
          fit result is passed to it, and its return value(s) is aggregated
          in the result dataset.
        add_constant : bool, optional
          If True, a constant will be added to the design matrix that is
          passed to ``exog``.
        """
        FeaturewiseMeasure.__init__(self, **kwargs)
        self._exog = exog
        if add_constant:
            self._exog = sm.add_constant(exog)
        self._res = res
        if isinstance(res, (np.ndarray, list, tuple)):
            self._res = np.atleast_1d(res)
        self._model_gen = model_gen


    def __fitmodel1d(self, Y):
        """Helper for apply_along_axis()"""
        res = self._res
        results = self._model_gen(Y, self._exog).fit()
        t_to_z = lambda t, df: stats.norm.ppf(stats.t.cdf(t, df))
        if isinstance(res, np.ndarray):
            if len(res.shape) == 1:
                tstats = results.t_test(self._res)
                return [np.asscalar(i) for i in [tstats.tvalue,
                                                 tstats.pvalue,
                                                 tstats.effect,
                                                 tstats.sd,
                                                 np.array(tstats.df_denom),
                                                 t_to_z(tstats.tvalue, tstats.df_denom)]]

            elif len(res.shape) == 2:
                fstats = results.f_test(self._res)
                return [np.asscalar(i) for i in
                            [fstats.fvalue,
                             fstats.pvalue]] + [fstats.df_num,
                                                fstats.df_denom]
            else:
                raise ValueError("Test specification (via `res`) has to be 1d or 2d array")
        elif isinstance(res, str):
            return results.__getattribute__(res)
        else:
            return res(results)


    def _call(self, dataset):
        # compute the regression once per feature
        results = np.apply_along_axis(self.__fitmodel1d, 0, dataset.samples)
        # figure out potential description of the results
        sa = None
        res = self._res
        if isinstance(res, np.ndarray):
            if len(res.shape) == 1:
                sa = ['tvalue', 'pvalue', 'effect', 'sd', 'df', 'zvalue']
            elif len(res.shape) == 2:
                sa = ['fvalue', 'pvalue', 'df_num', 'df_denom']
        elif isinstance(res, str):
            sa = [res] * len(results)
        if sa is not None:
            sa = {'descr': sa}
        # reassign the input feature attributes to the results
        return Dataset(results, sa=sa, fa=dataset.fa)



class GLM(UnivariateStatsModels):
    """Adaptor to the statsmodels-based UnivariateStatsModels

    This class is deprecated and only here to ease the transition of user code
    to the new classes. For all new code, please use the UnivariateStatsModels
    class.
    """
    def __init__(self, design, voi='pe', **kwargs):
        if isinstance(voi, str):
            # Possibly remap to adjusted interface
            voi = {'pe': 'params', 'zstat': 'zvalue'}.get(voi, voi)
        UnivariateStatsModels.__init__(
                              self,
                              design,
                              res=voi,
                              add_constant=False,
                              model_gen=lambda y, x: sm.OLS(y, x),
                              **kwargs)
