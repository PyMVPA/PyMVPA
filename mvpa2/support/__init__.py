# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Import helper for PyMVPA support modules

mvpa2.support is destined to contain temporary "fixes" to external modules
(Python, scipy), which are known to be fixed in the recent releases but are not
present in comodity distributions.

Example could be copy module."""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa2.base import debug
    debug('INIT', 'mvpa2.support start')
    debug('INIT', 'mvpa2.support end')

