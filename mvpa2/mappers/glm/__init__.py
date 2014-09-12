# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Transform datasets into parameter estimates of a general linear model fit.

This module provides the base class, as well as multiple implementations of a
GLM based on different 3rd-party packages.
"""

__docformat__ = 'restructuredtext'

__all__ = [ ] # GLMMapper is to be used only to derive specific implementations

import numpy as np

from mvpa2.datasets import Dataset
from mvpa2.mappers.base import Mapper
from mvpa2.base.param import Parameter

class GLMMapper(Mapper):
    """Transform a dataset into parameter estimates for a general linear model.

    This is a univariate measure were the model is fitted independently to each
    input feature.

    The GLM design matrix is created from two different source: a) sample
    attributes of the input dataset, and b) common regressors stored in the
    mapper itself upon instantiation. The order of the design matrix columns
    is as follows: First regressors from the dataset in the order in which
    their names were specified, then additional regressors stored in the mapper
    -- appended in their given order, and, lastly, a potential constant column.

    The nature of the values returned with the mapped dataset depends on the
    implementation details and parameter settings of the actual GLMMapper
    subclass. Most commonly, however, is a mapped dataset that has the same
    number of features as the input, and each sample contains the parameter
    estimates corresponding to a design matrix column.

    This is a base class, thus is not supposed to be used directly by users
    which should use specific implementations suchas NiPyGLMMapper and
    StatsmodelsGLMMapper.
    """
    # TODO optimize design matrix generation in case no regressor comes from the
    # input dataset and everything can be precomputed

    add_constant = Parameter(False, constraints='bool', doc="""\
            If True, a constant will be added as last column in the
            design matrix.""")

    return_design = Parameter(False, constraints='bool', doc="""\
            If True, the mapped dataset will contain a sample attribute
            ``regressors`` with the design matrix columns.""")

    return_model = Parameter(False, constraints='bool', doc="""\
            If True, the mapped dataset will contain am attribute
            ``model`` for an instance of the fitted GLM. The type of
            this instance dependent on the actual implementation used.""")

    def __init__(self, regs, add_regs=None, **kwargs):
        """
        Parameters
        ----------
        regs : list
          Names of sample attributes to be extracted from an input dataset and
          used as design matrix columns.
        add_regs : tuple, optional
          Additional regressors to be used in the design matrix. Each tuple
          element is a 2-tuple: the first element is a literal label for the
          regressor, and the second element is a 1D array with the regressor
          values. The length of the array needs to match the length of any
          input dataset.
        """
        if not 'space' in kwargs:
            kwargs['space'] = 'regressor_names'
        # so far no separate training
        Mapper.__init__(self, auto_train=True, **kwargs)
        self.regs = list(regs)
        if add_regs is None:
            add_regs = tuple()
        self.add_regs = tuple(add_regs)

    def _build_design(self, ds):
        X = None
        regsfromds = list(self.regs)
        reg_names=None
        if len(regsfromds):
            X = np.vstack([ds.sa[reg].value for reg in regsfromds]).T
            reg_names=regsfromds
        if len(self.add_regs):
            regs=[]
            if reg_names is None:
                reg_names = []
            for reg in self.add_regs:
                regs.append(reg[1])
                reg_names.append(reg[0])
            if X is None:
                X = np.vstack(regs).T
            else:
                X = np.vstack([X.T] + regs).T
        if self.params.add_constant:
            constant = np.ones(len(ds))
            if X is None:
                X = constant[None].T
            else:
                X = np.vstack((X.T, constant)).T
            if reg_names is None:
                reg_names = ['constant']
            else:
                reg_names.append('constant')
        if X is None:
            raise ValueError("no design specified")
        return reg_names, X

    def _fit_model(self, ds, X, reg_names):
        # return the model fit instance and
        # an output dataset (something x nfeatures of input ds)
        raise NotImplementedError

    def _forward_dataset(self, ds):
        reg_names, X = self._build_design(ds)
        model, out = self._fit_model(ds, X, reg_names)
        out.fa.update(ds.fa)
        out.a.update(ds.a) # this last one might be a bit to opportunistic
        if self.params.return_design:
            if not len(out) == len(X.T):
                raise ValueError("cannot include GLM regressors as sample "
                                 "attributes (dataset probably contains "
                                 "something other than parameter estimates")
            out.sa['regressors'] = X.T
        if self.params.return_model:
            out.a['model'] = model
        return out

 
    # TODO: this is not unreasonable, forward+reverse cycle throws away residuals...
    #def _reverse_dataset(self, ds):
        # reconstruct timeseries from model fit

from mvpa2 import externals
if externals.exists('nipy'):
    from .nipy_glm import NiPyGLMMapper
    __all__.append('NiPyGLMMapper')
if externals.exists('statsmodels'):
    from .statsmodels_glm import StatsmodelsGLMMapper
    __all__.append('StatsmodelsGLMMapper')
