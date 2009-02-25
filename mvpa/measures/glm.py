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

import numpy as N

from mvpa.measures.base import FeaturewiseDatasetMeasure

class GLM(FeaturewiseDatasetMeasure):
    """General linear model (GLM).

    Regressors can be defined in a design matrix and a linear fit of the
    data is computed univariately (i.e. indepently for each feature). This
    measure can report 'raw' beta weights of the linear model, as well
    as standardized parameters (z-scores) using an ordinary least squares
    (aka fixed-effects) approach to estimate the parameter estimate.

    The measure is reported in a (nfeatures x nregressors)-shaped array.
    """

    def __init__(self, design, voi='beta', **kwargs):
        """
        :Parameters:
          design: array(nsamples x nregressors)
            GLM design matrix.
          voi: 'beta' | 'zscore'
            Variable of interest that should be reported as feature-wise
            measure. 'beta' are the parameter estimates and 'zscore' returns
            standardized parameter estimates.
        """
        FeaturewiseDatasetMeasure.__init__(self, **kwargs)
        # store the design matrix as a such (no copying if already array)
        self._design = N.asmatrix(design)

        # what should be computed ('variable of interest')
        if not voi in ['beta', 'zscore']:
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

        # if betas are desired return them
        if self._voi == 'beta':
            # return as (feature x beta)
            return betas.T.A

        # compute residuals
        residuals = X * betas
        residuals -= dataset.samples

        # estimates of the parameter variance and compute zscores
        # assumption of mean(E) == 0 and equal variance
        beta_vars = residuals.var(axis=0) * self._inv_ip
        # (parameter x feature)
        zscore = betas.A / N.sqrt(beta_vars.T.A)

        if self._voi == 'zscore':
            # return as (feature x zscore)
            return zscore.T

        # we shall never get to this point
        raise ValueError, \
              "Unknown variable of interest '%s'" % str(self._voi)
