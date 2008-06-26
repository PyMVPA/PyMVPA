#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Misc. functions (in the mathematical sense)"""

__docformat__ = 'restructuredtext'

import numpy as N


def singleGammaHRF(t, A=5.4, W=5.2, K=1.0):
    """Hemodynamic response function model.

    The version consists of a single gamma function (also see
    doubleGammaHRF()).

    :Parameters:
      t: float
        Time.
      A: float
        Time to peak.
      W: float
        Full-width at half-maximum.
      K: float
        Scaling factor.
    """
    return K * (t / A) ** ((A ** 2) / (W ** 2) * 8.0 * N.log(2.0)) \
           * N.e ** ((t - A) / -((W ** 2) / A / 8.0 / N.log(2.0)))


def doubleGammaHRF(t, A1=5.4, W1=5.2, K1=1.0, A2=10.8, W2=7.35, K2=0.35):
    """Hemodynamic response function model.

    The version is using two gamm functions (also see singleGammaHRF()).

    :Parameters:
      t: float
        Time.
      A: float
        Time to peak.
      W: float
        Full-width at half-maximum.
      K: float
        Scaling factor.

    Parameters A, W and K exists individually for each of both gamma
    functions.
    """
    return singleGammaHRF(t, A1, W1, K1) - singleGammaHRF(t, A2, W2, K1)


