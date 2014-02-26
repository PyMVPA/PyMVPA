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

import numpy as np

from mvpa2.datasets import Dataset
from mvpa2.mappers.base import Mapper

class GLMMapper(Mapper):
    """
    First regressors from the dataset, then additional regressors, and a
    potential constant is added last.
    """
    def __init__(self, regs, add_regs=None, add_constant=False,
                 return_design=False, return_model=False, **kwargs):
        if not 'space' in kwargs:
            kwargs['space'] = 'regressor_names'
        # so far no separate training
        Mapper.__init__(self, auto_train=True, **kwargs)
        self.regs = list(regs)
        self.return_design = return_design
        self.return_model = return_model
        self.add_constant = add_constant
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
        if self.add_constant:
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
        if self.return_design:
            if not len(out) == len(X.T):
                raise ValueError("cannot include GLM regressors as sample "
                                 "attributes (dataset probably contains "
                                 "something other than parameter estimates")
            out.sa['regressors'] = X.T
        if self.return_model:
            out.a['model'] = model
        return out

 
    # TODO: this is not unreasonable, forward+reverse cycle throws away residuals...
    #def _reverse_dataset(self, ds):
        # reconstruct timeseries from model fit
