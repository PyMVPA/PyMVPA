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
    def __init__(self, regs, glmfit_kwargs=None, add_design=False,
                 add_glmfit=False, **kwargs):
        Mapper.__init__(self, auto_train=True, **kwargs)
        self.regs = regs
        if glmfit_kwargs is None:
            glmfit_kwargs = {}
        self.glmfit_kwargs = glmfit_kwargs
        self.add_design = add_design
        self.add_glmfit = add_glmfit

    def _forward_dataset(self, ds):
        X = np.vstack([ds.sa[reg].value for reg in self.regs]).T
        glm = GeneralLinearModel(X)
        glm.fit(ds.samples, **self.glmfit_kwargs)
        out = Dataset(glm.get_beta(),
                        sa={self.get_space(): self.regs},
                        fa=ds.fa,
                        a=ds.a) # this last one might be a bit to opportunistic
        if self.add_design:
            out.sa['regressors'] = X.T
        if self.add_glmfit:
            out.a['glmfit'] = glm
        return out
 
    # TODO: this is not unreasonable, forward+reverse cycle throws away residuals...
    #def _reverse_dataset(self, ds):
        # reconstruct timeseries from model fit
