# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""

"""

__docformat__ = 'restructuredtext'

from mvpa2.base import externals
if externals.exists('statsmodels', raise_=True):
    from mvpa2.measures.statsmodels_adaptor import UnivariateStatsModels
    import statsmodels.api as sm

import numpy as np

from mvpa2.datasets import Dataset
from mvpa2.mappers.glm import GLMMapper

class StatsmodelsGLMMapper(GLMMapper):
    """
    """
    def __init__(self, regs, model_gen=None, results='params',
                 **kwargs):
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
        res.sa[self.get_space()] = reg_names
        return mod, res
