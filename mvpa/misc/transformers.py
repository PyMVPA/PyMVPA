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


def SecondAxisMean(x):
    """Mean across 2nd axis

    Use cases:
     - to combine multiple sensitivities to get sense about
       mean sensitivity across splits
    """
    return N.mean(x, axis=1)


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


def L2Normed(x, norm=1.0, reverse=False):
    """Norm the values so that regular vector norm becomes `norm`"""
    xnorm = N.linalg.norm(x)
    return x * (norm/xnorm)

def L1Normed(x, norm=1.0, reverse=False):
    """Norm the values so that L_1 norm (sum|x|) becomes `norm`"""
    xnorm = N.sum(N.abs(x))
    return x * (norm/xnorm)

def RankOrder(x, reverse=False):
    """Rank-order by value. Highest gets 0"""

    # XXX was Yarik on drugs? please simplify this beast
    nelements = len(x)
    indexes = N.arange(nelements)
    t_indexes = indexes
    if not reverse:
        t_indexes = indexes[::-1]
    tosort = zip(x, indexes)
    tosort.sort()
    ztosort = zip(tosort, t_indexes)
    rankorder = N.empty(nelements, dtype=int)
    rankorder[ [x[0][1] for x in ztosort] ] = \
               [x[1] for x in ztosort]
    return rankorder


def ReverseRankOrder(x):
    """Convinience functor"""
    return RankOrder(x, reverse=True)
