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
    """Norm the values so that regular vector norm becomes `norm`

    More verbose: Norm that the sum of the squared elements of the
    returned vector becomes `norm`.
    """
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


class OverAxis(object):
    """Helper to apply transformer over specific axis
    """

    def __init__(self, transformer, axis=None):
        """Initialize transformer wrapper with an axis.

        :Parameters:
          transformer
            A callable to be used
          axis : None or int
            If None -- apply transformer across all the data. If some
            int -- over that axis
        """
        self.transformer = transformer
        # sanity check
        if not (axis is None or isinstance(axis, int)):
            raise ValueError, "axis must be specified by integer"
        self.axis = axis


    def __call__(self, x, *args, **kwargs):
        transformer = self.transformer
        axis = self.axis
        if axis is None:
            return transformer(x, *args, **kwargs)

        x = N.asanyarray(x)
        shape = x.shape
        if axis >= len(shape):
            raise ValueError, "Axis given in constructor %d is higher " \
                  "than dimensionality of the data of shape %s" % (axis, shape)

        # WRONG! ;-)
        #for ind in xrange(shape[axis]):
        #    results.append(transformer(x.take([ind], axis=axis),
        #                              *args, **kwargs))

        # TODO: more elegant/speedy solution?
        shape_sweep = shape[:axis] + shape[axis+1:]
        shrinker = None
        """Either transformer reduces the dimensionality of the data"""
        #results = N.empty(shape_out, dtype=x.dtype)
        for index_sweep in N.ndindex(shape_sweep):
            if axis > 0:
                index = index_sweep[:axis]
            else:
                index = ()
            index = index + (slice(None),) + index_sweep[axis:]
            x_sel = x[index]
            x_t = transformer(x_sel, *args, **kwargs)
            if shrinker is None:
                if N.isscalar(x_t) or x_t.shape == shape_sweep:
                    results = N.empty(shape_sweep, dtype=x.dtype)
                    shrinker = True
                elif x_t.shape == x_sel.shape:
                    results = N.empty(x.shape, dtype=x.dtype)
                    shrinker = False
                else:
                    raise RuntimeError, 'Not handled by OverAxis kind of transformer'

            if shrinker:
                results[index_sweep] = x_t
            else:
                results[index] = x_t

        return results
