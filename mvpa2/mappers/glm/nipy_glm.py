# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""GLMMapper implementation based on the NiPy package."""

__docformat__ = 'restructuredtext'

from mvpa2.base import externals
if externals.exists('nipy', raise_=True):
    from nipy.modalities.fmri.glm import GeneralLinearModel

import numpy as np

from mvpa2.datasets import Dataset
from mvpa2.mappers.glm import GLMMapper

class NiPyGLMMapper(GLMMapper):
    """NiPy-based GLMMapper implementation

    This is basically a front-end for
    :class:`~ nipy.modalities.fmri.glm.GeneralLinearModel`.
    In particular, it supports all keyword arguments of its
    ``fit()`` method.
    """

    def __init__(self, regs, glmfit_kwargs=None, **kwargs):
        """
        Parameters
        ----------
        regs : list
          Names of sample attributes to be extracted from an input dataset and
          used as design matrix columns.
        glmfit_kwargs : dict, optional
          Keyword arguments to be passed to GeneralLinearModel.fit().
          By default an AR1 model is used.
        """
        GLMMapper.__init__(self, regs, **kwargs)
        if glmfit_kwargs is None:
            glmfit_kwargs = {}
        self.glmfit_kwargs = glmfit_kwargs

    def _fit_model(self, ds, X, reg_names):
        glm = GeneralLinearModel(X)
        glm.fit(ds.samples, **self.glmfit_kwargs)
        out = Dataset(glm.get_beta(),
                      sa={self.get_space(): reg_names})
        return glm, out
