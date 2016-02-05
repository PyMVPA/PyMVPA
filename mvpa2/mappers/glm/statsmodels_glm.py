# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""GLMMapper implementation based on the statsmodels package."""

__docformat__ = 'restructuredtext'

from mvpa2.base import externals
if externals.exists('statsmodels', raise_=True):
    from mvpa2.measures.statsmodels_adaptor import UnivariateStatsModels
    import statsmodels.api as sm

import numpy as np

from mvpa2.datasets import Dataset
from mvpa2.mappers.glm import GLMMapper

class StatsmodelsGLMMapper(GLMMapper):
    """Statsmodels-based GLMMapper implementation

    This is basically a front-end for
    :class:`~mvpa2.measures.statsmodels_adaptor.UnivariateStatsModels`.
    In particular, it supports all ``model_gen`` and ``results`` arguments
    as described in the documentation for this class.
    """
    def __init__(self, regs, model_gen=None, results='params',
                 **kwargs):
        """
        Parameters
        ----------
        regs : list
          Names of sample attributes to be extracted from an input dataset and
          used as design matrix columns.
        model_gen : callable, optional
          See UnivariateStatsModels documentation for details on the
          specification of the model fitting procedure. By default an
          OLS model is used.
        results : str or array, optional
          See UnivariateStatsModels documentation for details on the
          specification of model fit results. By default parameter
          estimates are returned.
        """
        GLMMapper.__init__(self, regs, **kwargs)
        self.result_expr = results
        if model_gen is None:
            model_gen=lambda y, x: sm.OLS(y, x)
        self.model_gen = model_gen

    def _fit_model(self, ds, X, reg_names):
        mod = UnivariateStatsModels(
                X,
                res=self.result_expr,
                add_constant=False,
                model_gen=self.model_gen)
        res = mod(ds)
        if self.result_expr == 'params':
            res.sa[self.get_space()] = reg_names
        return mod, res
