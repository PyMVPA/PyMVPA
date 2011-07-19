# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Misc support utility for making us compatible with liquid RPy2 atm
"""

__docformat__ = 'restructuredtext'

from mvpa2.base.externals import exists, versions

#if __debug__:
#    from mvpa2.base import debug

__all__ = []

if exists('rpy2', raise_=True):
    __all__ = [ 'Rrx', 'Rrx2' ]

    if versions['rpy2'] >= '2.1.0beta':
        def Rrx(self, x):
            return self.rx(x)
        def Rrx2(self, x):
            return self.rx2(x)
    elif versions['rpy2'] >= '2.0':
        def Rrx(self, x):
            return self.r[x]
        def Rrx2(self, x):
            return self.r[x][0]
    else:
        raise ValueError, \
              "We do not have support for rpy2 version %(rpy2)s" % versions

    Rrx.__doc__ = "Access delegator for R function ["
    Rrx2.__doc__ = "Access delegator for R function [["
