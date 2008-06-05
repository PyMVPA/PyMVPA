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


def SecondAxisSumOfAbs(x):
    """Sum of absolute values along the 2nd axis

    Use cases:
     - to combine multiple sensitivities to get sense about
       what features are most influential
    """
    return N.abs(x).sum(axis=1)


def SecondAxisMaxOfAbs(x):
    """Max of absolute values along the 2nd axis
    """
    return N.abs(x).max(axis=1)


def GrandMean(x):
    """Just what the name suggests."""
    return N.mean(x)


def RankOrder(x, reverse=False):
    """Rank-order by value. Highest gets 0"""
    nelements = len(x)
    indexes = N.arange(nelements)
    if not reverse:
        indexes = indexes[::-1]
    tosort = zip(x, indexes)
    tosort.sort()
    return [x[1] for x in tosort]


def ReverseRankOrder(x):
    """Convinience functor"""
    return RankOrder(x, reverse=True)
