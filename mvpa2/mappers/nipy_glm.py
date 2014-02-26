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
if externals.exists('nipy', raise_=True):
    from nipy.modalities.fmri.glm import GeneralLinearModel

import numpy as np

from mvpa2.datasets import Dataset
from mvpa2.mappers.base import Mapper

class NiPyGLMMapper(Mapper):
    """
    First regressors from the dataset, then additional regressors, and a
    potential constant is added last.
    """
    def __init__(self, regs, glmfit_kwargs=None, return_design=False,
                 return_glmfit=False, add_regs=None, add_constant=False,
                 **kwargs):
        if not 'space' in kwargs:
            kwargs['space'] = 'regressor_names'
        Mapper.__init__(self, auto_train=True, **kwargs)
        self.regs = list(regs)
        if glmfit_kwargs is None:
            glmfit_kwargs = {}
        self.glmfit_kwargs = glmfit_kwargs
        self.return_design = return_design
        self.return_glmfit = return_glmfit
        self.add_constant = add_constant
        if add_regs is None:
            add_regs = tuple()
        self.add_regs = tuple(add_regs)

    def _forward_dataset(self, ds):
        X = None
        regs = list(self.regs)
        if len(regs):
            X = np.vstack([ds.sa[reg].value for reg in regs]).T
        if len(self.add_regs):
            regs = []
            reg_names = []
            for reg in self.add_regs:
                regs.append(reg[1])
                reg_names.append(reg[0])
            if X is None:
                X = np.vstack(regs).T
            else:
                X = np.vstack([X.T] + regs).T
            regs += reg_names
        if self.add_constant:
            constant = np.ones(len(ds))
            if X is None:
                X = constant[None].T
            else:
                X = np.vstack((X.T, constant)).T
            regs.append('constant')
        if X is None:
            raise ValueError("no design specified")
        glm = GeneralLinearModel(X)
        glm.fit(ds.samples, **self.glmfit_kwargs)
        out = Dataset(glm.get_beta(),
                        sa={self.get_space(): regs},
                        fa=ds.fa,
                        a=ds.a) # this last one might be a bit to opportunistic
        if self.return_design:
            out.sa['regressors'] = X.T
        if self.return_glmfit:
            out.a['glmfit'] = glm
        return out
 
    # TODO: this is not unreasonable, forward+reverse cycle throws away residuals...
    #def _reverse_dataset(self, ds):
        # reconstruct timeseries from model fit
