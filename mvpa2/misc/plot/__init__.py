# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Import helper for miscellaneous PyMVPA plotting functions (mvpa2.misc.plot)"""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa2.base import debug
    debug('INIT', 'mvpa2.misc.plot start')

from mvpa2.base import externals

if externals.exists('matplotlib'):
    from mvpa2.misc.plot.base import *

if __debug__:
    debug('INIT', 'mvpa2.misc.plot end')
