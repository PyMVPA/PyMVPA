# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helpers to unify/facilitate unittesting within PyMVPA

"""

__docformat__ = 'restructuredtext'

from mvpa.base import externals

if __debug__:
    from mvpa.base import debug
    debug('INIT', 'mvpa.testing')


if __debug__:
    debug('INIT', 'mvpa.testing end')
