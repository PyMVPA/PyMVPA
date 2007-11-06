#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""C++-like virtual properties"""


class VProperty(object):
    """Provides "virtual" property: uses derived class's method
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=''):
        for attr in ('fget', 'fset'):
            func = locals()[attr]
            if callable(func):
                setattr(self, attr, func.func_name)
        setattr(self, '__doc__', doc)

    def __get__(self, obj=None, type=None):
        if not obj:
            return 'property'
        if self.fget:
            return getattr(obj, self.fget)()

    def __set__(self, obj, arg):
        if self.fset:
            return getattr(obj, self.fset)(arg)
