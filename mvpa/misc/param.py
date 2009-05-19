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
from mvpa.misc.state import CollectableAttribute

if __debug__:
    from mvpa.base import debug

_whitespace_re = re.compile('\n\s+|^\s+')

__all__ = [ 'Parameter', 'KernelParameter' ]

class Parameter(CollectableAttribute):
    """This class shall serve as a representation of a parameter.

    It might be useful if a little more information than the pure parameter
    value is required (or even only useful).

    Each parameter must have a value. However additional property can be
    passed to the constructor and will be stored in the object.

    BIG ASSUMPTION: stored values are not mutable, ie nobody should do

    cls.parameter1[:] = ...

    or we wouldn't know that it was changed

    Here is a list of possible property names:

        min   - minimum value
        max   - maximum value
        step  - increment/decrement stepsize
    """

    def __init__(self, default, name=None, doc=None, index=None, **kwargs):
        """Specify a parameter by its default value and optionally an arbitrary
        number of additional parameters.

        TODO: :Parameters: for Parameter
        """
        self.__default = default

        CollectableAttribute.__init__(self, name=name, doc=doc, index=index)

        self.resetvalue()
        self._isset = False

        if __debug__:
            if kwargs.has_key('val'):
                raise ValueError, "'val' property name is illegal."

        # XXX probably is too generic...
        for k, v in kwargs.iteritems():
            self.__setattr__(k, v)


    def __str__(self):
        res = CollectableAttribute.__str__(self)
        # it is enabled but no value is assigned yet
        res += '=%s' % (self.value,)
        return res


    def doc(self, indent="  ", width=70):
        """Docstring for the parameter to be used in lists of parameters

        :Returns:
          string or list of strings (if indent is None)
        """
        paramsdoc = "  %s" % self.name
        if hasattr(paramsdoc, 'allowedtype'):
            paramsdoc += " : %s" % self.allowedtype
        paramsdoc = [paramsdoc]
        try:
            doc = self.__doc__
            if not doc.endswith('.'): doc += '.'
            try:
                doc += " (Default: %s)" % self.default
            except:
                pass
            # Explicitly deal with multiple spaces, for some reason
            # replace_whitespace is non-effective
            doc = _whitespace_re.sub(' ', doc)
            paramsdoc += ['  ' + x
                          for x in textwrap.wrap(doc, width=width-len(indent),
                                                 replace_whitespace=True)]
        except Exception, e:
            pass

        if indent is None:
            return paramsdoc
        else:
            return ('\n' + indent).join(paramsdoc)


    # XXX should be named reset2default? correspondingly in
    #     ParameterCollection as well
    def resetvalue(self):
        """Reset value to the default"""
        #CollectableAttribute.reset(self)
        if not self.isDefault:
            self._isset = True
            self.value = self.__default

    def _set(self, val):
        if self._value != val:
            if __debug__:
                debug("COL",
                      "Parameter: setting %s to %s " % (str(self), val))
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
            self._isset = True
        elif __debug__:
            debug("COL",
                  "Parameter: not setting %s since value is the same" \
                  % (str(self)))

    @property
    def isDefault(self):
        """Returns True if current value is bound to default one"""
        return self._value is self.default

    @property
    def equalDefault(self):
        """Returns True if current value is equal to default one"""
        return self._value == self.__default

    def setDefault(self, value):
        wasdefault = self.isDefault
        self.__default = value
        if wasdefault:
            self.resetvalue()
            self._isset = False

    # incorrect behavior
    #def reset(self):
    #    """Override reset so we don't clean the flag"""
    #    pass

    default = property(fget=lambda x:x.__default, fset=setDefault)
    value = property(fget=lambda x:x._value, fset=_set)

class KernelParameter(Parameter):
    """Just that it is different beast"""
    pass
