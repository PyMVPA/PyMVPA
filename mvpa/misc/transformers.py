#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Simply functors that transform the output of another functor."""

__docformat__ = 'restructuredtext'


import numpy as N

class Transformer(object):
    """Base class for decorators to function calls"""

    def __init__(self, obj):
        """Cheap initialization."""
        self._callable = obj


class Absolute(Transformer):
    """Returns the elementwise absolute value of the value(s) that are
    returned by the wrapped object.
    """

    def __call__(self, *args, **kwargs):
        """Pass the call to the wrapper object and transform output."""
        return N.absolute(self._callable(*(args), **(kwargs)))



class OneMinus(Transformer):
    """Returns elementwise '1 - x', where x is returned by the wrapped object.
    """

    def __call__(self, *args, **kwargs):
        """Pass the call to the wrapper object and transform output."""
        # not sure what is best
        #return 1 - self.__callable(*(args), **(kwargs))

        # perhaps in-place is better
        out = self._callable(*(args), **(kwargs))
        out *= -1
        out += 1
        return out




