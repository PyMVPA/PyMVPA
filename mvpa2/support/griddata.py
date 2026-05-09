# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Provide a `griddata(x, y, z, xi, yi, interp='linear')` shim.

Historically PyMVPA preferred the standalone `griddata` package and fell
back to `matplotlib.mlab.griddata`.  Both have been gone for years
(`mlab.griddata` removed in matplotlib 3.1).  This module now wraps
`scipy.interpolate.griddata`, which has a different calling convention
(points + values + xi instead of x, y, z, xi, yi), and exposes the
historical 5-positional-arg signature so existing call sites
(`mvpa2/misc/plot/topo.py` and the docstrings) keep working unchanged.
"""

__docformat__ = 'restructuredtext'

import numpy as np
from scipy.interpolate import griddata as _scipy_griddata


def griddata(x, y, z, xi, yi, interp='linear'):
    """Interpolate scattered data onto a regular grid.

    Parameters
    ----------
    x, y : 1D array_like
      Coordinates of the input data points.
    z : 1D array_like
      Values at the input data points.
    xi, yi : 1D or 2D array_like
      Coordinates of the output grid.  When 1D, the output is computed on
      the meshgrid of (xi, yi).  When 2D (e.g. already meshgridded), the
      output is computed at those coordinates directly.
    interp : {'linear', 'nearest', 'cubic'}
      Interpolation method.  ``matplotlib``'s historical ``'nn'`` (natural
      neighbor) is not available in scipy; we map it to ``'linear'``.

    Returns
    -------
    ndarray
      Interpolated values on the output grid.
    """
    method = 'linear' if interp == 'nn' else interp
    points = np.column_stack((np.asarray(x).ravel(),
                              np.asarray(y).ravel()))
    values = np.asarray(z).ravel()
    return _scipy_griddata(points, values, (xi, yi), method=method)
