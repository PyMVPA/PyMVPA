# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##g
"""Parameter representation"""

__docformat__ = 'restructuredtext'

import numpy as np

if __debug__:
    from mvpa2.base import debug

class EnsureValue(object):
    """Base class for input value conversion/validation.

    These classes are also meant to be able to generate appropriate
    documentation on an appropriate parameter value.
    """
    def __init__(self):
        return

    def __call__(self, value):
        # do any necessary checks or conversions, potentially catch exceptions
        # and generate a meaningful error message
        return value

    # proposed name -- please invent a better one
    def long_description(self):
        # return meaningful docs or None
        # used as a comprehensive description in the parameter list
        return self.short_description()

    def short_description(self):
        # return meaningful docs or None
        # used as a condensed primer for the parameter lists
        return None

class EnsureDType(object):
    """Ensure that an input is of a particular data type,
    and, raises an Exception in case it is not.
    """
    # TODO extend to support numpy-like dtype specs, e.g. 'int64'
    # in addition to functors
    def __init__(self, dtype):
        """
        Parameters
        ----------
        dtype : functor
        """
        self._dtype = dtype

    def __call__(self, value):
        if hasattr(value, '__array__'):
            import numpy as np
            return np.asanyarray(value, dtype=self._dtype)
        elif hasattr(value,'__iter__'):
            return map(self._dtype, value)
        else:
            return self._dtype(value)

    def short_description(self):
        dtype_descr = str(self._dtype)
        if dtype_descr[:7] == "<type '" and dtype_descr[-2:] == "'>":
            dtype_descr = dtype_descr[7:-2]
        return dtype_descr

    def long_description(self):
        return "value must be convertible to type '%s'" % self.short_description()

class EnsureInt(EnsureDType):
    def __init__(self):
        EnsureDType.__init__(self, int)

class EnsureFloat(EnsureDType):
    def __init__(self):
        EnsureDType.__init__(self, float)

class EnsureBool(EnsureValue):
    def __call__(self, value):
        if isinstance(value, bool):
            return value
        elif value in ('0', 'no', 'off', 'disable', 'false'):
            return False
        elif value in ('1', 'yes', 'on', 'enable', 'true'):
            return True
        else:
            raise ValueError(
                    "'%s' cannot be converted into a boolean" % value)

    def long_description(self):
        return 'value must be convertible to type bool'

    def short_description(self):
        return 'bool'

class EnsureNone(EnsureValue):
    def __call__(self, value):
        if value is None:
            return None
        else:
            raise ValueError("value must be `None`")

    def short_description(self):
        return 'None'

    def long_description(self):
        return 'value must be `None`'

class EnsureChoice(EnsureValue):
    def __init__(self, *args):
        self._allowed = args

    def __call__(self, value):
        if value not in self._allowed:
            raise ValueError, "value is not one of %s" % (self._allowed, )
        return value

    def long_description(self):
        return 'value must be one of %s' % (str(self._allowed), )

    def short_description(self):
        return '{%s}' % ', '.join([str(c) for c in self._allowed])

class EnsureRange(EnsureValue):
    def __init__(self, min=None, max=None):
        self._min = min
        self._max = max

    def __call__(self, value):
        if self._min is not None:
            if value < self._min:
                raise ValueError, "value must be at least %s" % (self._min, )
        if self._max is not None:
            if value > self._max:
                raise ValueError, "value must be at most %s" % (self._max, )
        return value

    def long_description(self):
        min_str='-inf' if self._min is None else str(self._min)
        max_str='inf' if self._max is None else str(self._max)
        return 'value must be in range [%s, %s]' % (min_str, max_str)


class AltConstraints(object):
    def __init__(self, *args):
        self.constraints = [EnsureNone() if c is None else c for c in args]

    def __call__(self, value):
        e_list = []
        for c in self.constraints:
            try:
                return c(value)
            except Exception, e:
                e_list.append(e)
        raise ValueError("all alternative constraints violated")

    def long_description(self):
        cs = [c.long_description() for c in self.constraints if hasattr(c, 'long_description')]
        doc = ', or '.join(cs)
        if len(cs) > 1:
            return '(%s)' % doc
        else:
            return doc

    def short_description(self):
        cs = [c.short_description() for c in self.constraints
                    if hasattr(c, 'short_description') and not c.short_description() is None]
        doc = ' or '.join(cs)
        if len(cs) > 1:
            return '(%s)' % doc
        else:
            return doc



class Constraints(object):
    def __init__(self, *args):
        self.constraints = [EnsureNone() if c is None else c for c in args]

    def __call__(self, value):
        for c in (self.constraints):
            value = c(value)
        return value

    def long_description(self):
        cs = [c.long_description() for c in self.constraints if hasattr(c, 'long_description')]
        doc = ', and '.join(cs)
        if len(cs) > 1:
            return '(%s)' % doc
        else:
            return doc

    def short_description(self):
        cs = [c.short_description() for c in self.constraints
                    if hasattr(c, 'short_description') and not c.short_description() is None]
        doc = ' and '.join(cs)
        if len(cs) > 1:
            return '(%s)' % doc
        else:
            return doc
