# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""The general linear model (GLM)."""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.measures.base import FeaturewiseMeasure
from mvpa2.base.state import ConditionalAttribute
from mvpa2.datasets.base import Dataset

class GLM(FeaturewiseMeasure):
    """General linear model (GLM).

    Regressors can be defined in a design matrix and a linear fit of the data
    is computed univariately (i.e. indepently for each feature). This measure
    can report 'raw' parameter estimates (i.e. beta weights) of the linear
    model, as well as standardized parameters (z-stat) using an ordinary
    least squares (aka fixed-effects) approach to estimate the parameter
    estimate.

    The measure is reported in a (nfeatures x nregressors)-shaped array.
    """

    pe = ConditionalAttribute(enabled=False,
        doc="Parameter estimates (nfeatures x nparameters).")

    zstat = ConditionalAttribute(enabled=False,
        doc="Standardized parameter estimates (nfeatures x nparameters).")

    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, design, voi='pe', **kwargs):
        """
        Parameters
        ----------
        design : array (nsamples x nregressors)
          GLM design matrix.
        voi : {'pe', 'zstat'}
          Variable of interest that should be reported as feature-wise
          measure. 'beta' are the parameter estimates and 'zstat' returns
          standardized parameter estimates.
        """
        FeaturewiseMeasure.__init__(self, **kwargs)
        # store the design matrix as a such (no copying if already array)
        self._design = np.asmatrix(design)

        # what should be computed ('variable of interest')
        if not voi in ['pe', 'zstat']:
            raise ValueError, \
                  "Unknown variable of interest '%s'" % str(voi)
        self._voi = voi

        # will store the precomputed Moore-Penrose pseudo-inverse of the
        # design matrix (lazy calculation)
        self._inv_design = None
        # also store the inverse of the inner product for beta variance
        # estimation
        self._inv_ip = None


    def _call(self, dataset):
        # just for the beauty of it
        X = self._design

        # precompute transformation is not yet done
        if self._inv_design is None:
            self._inv_ip = (X.T * X).I
            self._inv_design = self._inv_ip * X.T

        # get parameter estimations for all features at once
        # (betas x features)
        betas = self._inv_design * dataset.samples

        # charge state
        self.ca.pe = pe = betas.T.A

        # if betas and no z-stats are desired return them right away
        if not self._voi == 'pe' or self.ca.is_enabled('zstat'):
            # compute residuals
            residuals = X * betas
            residuals -= dataset.samples

            # estimates of the parameter variance and compute zstats
            # assumption of mean(E) == 0 and equal variance
            # XXX next lines ignore off-diagonal elements and hence covariance
            # between regressors. The humble being writing these lines asks the
            # god of statistics for forgives, because it knows not what it does
            diag_ip = np.diag(self._inv_ip)
            # (features x betas)
            beta_vars = np.array([ r.var() * diag_ip for r in residuals.T ])
            # (parameter x feature)
            zstat = pe / np.sqrt(beta_vars)

            # charge state
            self.ca.zstat = zstat

        if self._voi == 'pe':
            # return as (beta x feature)
            result = Dataset(pe.T)
        elif self._voi == 'zstat':
            # return as (zstat x feature)
            result = Dataset(zstat.T)
        else:
            # we shall never get to this point
            raise ValueError, \
                  "Unknown variable of interest '%s'" % str(self._voi)
        result.sa['regressor'] = np.arange(len(result))
        return result
