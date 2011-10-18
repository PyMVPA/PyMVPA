# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""C++-like virtual properties"""

__docformat__ = 'restructuredtext'


class VProperty(object):
    """Provides "virtual" property: uses derived class's method
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=''):
        """
        Parameters are the same as of generic `property`.
        """
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
