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

import re
import textwrap
import numpy as np
from mvpa2.base.state import IndexedCollectable

if __debug__:
    from mvpa2.base import debug

_whitespace_re = re.compile('\n\s+|^\s+')

__all__ = [ 'Parameter', 'KernelParameter' ]

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
    def get_doc(self):
        # return meaningful docs or None
        return None

class EnsureFloat(EnsureValue):
    def __call__(self, value):
        return float(value)

    def get_doc(self):
        return 'value must be convertible to type float'

# TODO needs update
class ChoiceValidator(object):
    def __init__(self, allowed):
        self._allowed = allowed

    def __call__(self, value):
        return self.validate(value)

    def validate(self, value):
        if value not in self._allowed:
            raise ValueError, "Value is not in %s" % (self._allowed, )
        return value


# TODO needs update
class RangeValidator(object):

    def __init__(self, min=None, max=None):
        self._min = min
        self._max = max

    def __call__(self, value):
        return self.validate(value)

    def validate(self, value):
        if self._min is not None:
            if value < self._min:
                raise ValueError, "Value must be at least %s" % (self._min, )
        if self._max is not None:
            if value > self._max:
                raise ValueError, "Value must be at most %s" % (self._max, )
        return value


class Parameter(IndexedCollectable):
    """This class shall serve as a representation of a parameter.

    It might be useful if a little more information than the pure parameter
    value is required (or even only useful).

    Each parameter must have a value. However additional attributes can be
    passed to the constructor and will be stored in the object.

    Notes
    -----
    BIG ASSUMPTION: stored values are not mutable, ie nobody should do

        cls.parameter1[:] = ...

    or we wouldn't know that it was changed
    Here is a list of possible additional attributes:

    allowedtype : str
      Description of what types are allowed
    min
      Minimum value
    max
      Maximum value
    step
      Increment/decrement step size hint for optimization
    """

    def __init__(self, default, constraints=None, ro=False,  index=None,  value=None,
                 name=None, doc=None, **kwargs):
        """Specify a Parameter with a default value and arbitrary
        number of additional attributes.

        Parameters
        ----------
        constraints : callable
          A functor that takes any input value, performs checks or type
          conversions and finally returns a value that is appropriate for a
          parameter or raises an exception.
        name : str
          Name of the parameter under which it should be available in its
          respective collection.
        doc : str
          Documentation about the purpose of this parameter.
        index : int or None
          Index of parameter among the others.  Determines order of listing
          in help.  If None, order of instantiation determines the index.
        ro : bool
          Either value which will be assigned in the constructor is read-only and
          cannot be changed
        value
          Actual value of the parameter to be assigned
        """
        # XXX probably is too generic...
        # and potentially dangerous...
        # let's at least keep track of what is passed
        self._additional_props = []
        for k, v in kwargs.iteritems():
            self.__setattr__(k, v)
            self._additional_props.append(k)

        self.__default = default
        self._ro = ro
        self._constraints = constraints

        # needs to come after kwargs processing, since some debug statements
        # rely on working repr()
        # value is not passed since we need to invoke _set with init=True
        # below
        IndexedCollectable.__init__(self, index=index, # value=value,
                                    name=name, doc=doc)
        self._isset = False
        if value is None:
            self._set(self.__default, init=True)
        else:
            self._set(value, init=True)

        if __debug__:
            if 'val' in kwargs:
                raise ValueError, "'val' property name is illegal."


    def __reduce__(self):
        icr = IndexedCollectable.__reduce__(self)
        # Collect all possible additional properties which were passed
        # to the constructor
        state = dict([(k, getattr(self, k)) for k in self._additional_props])
        state['_additional_props'] = self._additional_props
        state.update(icr[2])
        res = (self.__class__, (self.__default, self._constraints, self._ro) + icr[1], state)
        #if __debug__ and 'COL_RED' in debug.active:
        #    debug('COL_RED', 'Returning %s for %s' % (res, self))
        return res


    def __str__(self):
        res = IndexedCollectable.__str__(self)
        # it is enabled but no value is assigned yet
        res += '=%s' % (self.value,)
        return res


    def __repr__(self):
        # cannot use IndexedCollectable's repr(), since the contructor
        # needs to handle the mandatory 'default' argument
        # TODO: so what? just tune it up ;)
        # TODO: think what to do with index parameter...
        s = "%s(%s, name=%s, doc=%s" % (self.__class__.__name__,
                                        self.__default,
                                        repr(self.name),
                                        repr(self.__doc__))
        plist = ["%s=%s" % (p, self.__getattribute__(p))
                    for p in self._additional_props]
        if len(plist):
            s += ', ' + ', '.join(plist)
        if self._ro:
            s += ', ro=True'
        if not self.is_default:
            s += ', value=%r' % self.value
        s += ')'
        return s


    def _paramdoc(self, indent="  ", width=70):
        """Docstring for the parameter to be used in lists of parameters

        Returns
        -------
        string or list of strings (if indent is None)
        """
        paramsdoc = '%s' % self.name
        if hasattr(paramsdoc, 'allowedtype'):
            paramsdoc += " : %s" % self.allowedtype
        paramsdoc = [paramsdoc]
        try:
            doc = self.__doc__.strip()
            if not doc.endswith('.'):
                doc += '.'
            if not self._constraints is None:
                doc += ' Constraints: %s.' % self._constraints.get_doc()
            if hasattr(self, 'choices') \
              and ((hasattr(self, 'allowedtype') and 'string' in self.allowedtype)
                   or np.all([isinstance(x, basestring) for x in self.choices])):
                choices = ', '.join(repr(x) for x in self.choices)
                doc += " [Choices: %s]" % choices
            try:
                doc += " (Default: %r)" % (self.default,)
            except:
                pass
            # Explicitly deal with multiple spaces, for some reason
            # replace_whitespace is non-effective
            doc = _whitespace_re.sub(' ', doc)
            paramsdoc += [indent + x
                          for x in textwrap.wrap(doc, width=width-len(indent),
                                                 replace_whitespace=True)]
        except Exception, e:
            pass

        return '\n'.join(paramsdoc)


    # XXX should be named reset2default? correspondingly in
    #     ParameterCollection as well
    def reset_value(self):
        """Reset value to the default"""
        #IndexedCollectable.reset(self)
        if not self.is_default and not self._ro:
            self._isset = True
            self.value = self.__default

    def _set(self, val, init=False):
        if self._constraints is not None:
            val = self._constraints(val)
        different_value = self._value != val
        isarray = isinstance(different_value, np.ndarray)
        if self._ro and not init:
            raise RuntimeError, \
                  "Attempt to set read-only parameter %s to %s" \
                  % (self.name, val)
        if (isarray and np.any(different_value)) or \
           ((not isarray) and different_value):
            if __debug__:
                debug("COL",
                      "Parameter: setting %s to %s " % (str(self), val))
            if not isarray:
                if hasattr(self, 'min') and val < self.min:
                    raise ValueError, \
                          "Minimal value for parameter %s is %s. Got %s" % \
                          (self.name, self.min, val)
                if hasattr(self, 'max') and val > self.max:
                    raise ValueError, \
                          "Maximal value for parameter %s is %s. Got %s" % \
                          (self.name, self.max, val)
                if hasattr(self, 'choices') and (not val in self.choices):
                    raise ValueError, \
                          "Valid choices for parameter %s are %s. Got %s" % \
                          (self.name, self.choices, val)
            self._value = val
            # Set 'isset' only if not called from initialization routine
            self._isset = not init #True
        elif __debug__:
            debug("COL",
                  "Parameter: not setting %s since value is the same" \
                  % (str(self)))

    @property
    def is_default(self):
        """Returns True if current value is bound to default one"""
        return self._value is self.default

    @property
    def equal_default(self):
        """Returns True if current value is equal to default one"""
        return self._value == self.__default

    def _set_default(self, value):
        wasdefault = self.is_default
        self.__default = value
        if wasdefault:
            self.reset_value()
            self._isset = False

    # incorrect behavior
    #def reset(self):
    #    """Override reset so we don't clean the flag"""
    #    pass

    default = property(fget=lambda x:x.__default, fset=_set_default)
    value = property(fget=lambda x:x._value, fset=_set)


class KernelParameter(Parameter):
    """Just that it is different beast"""
    pass
