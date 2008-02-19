#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Simply functors that transform something."""

__docformat__ = 'restructuredtext'


import numpy as N


def Absolute(x):
    """Returns the elementwise absolute of any argument."""
    return N.absolute(x)


def OneMinus(x):
    """Returns elementwise '1 - x' of any argument."""
    return 1 - x


def Identity(x):
    """Return whatever it was called with."""
    return x


def FirstAxisMean(x):
    """Mean computed along the first axis."""
    return N.mean(x, axis=0)
