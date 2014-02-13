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

import numpy as np            # we barely can step somewhere without it
from mvpa2.base import externals
from mvpa2 import pymvpa_dataroot

if __debug__:
    from mvpa2.base import debug
    debug('INIT', 'mvpa2.testing')

from mvpa2.testing.tools import *

if __debug__:
    from mvpa2.base import debug

    _ENFORCE_CA_ENABLED = 'ENFORCE_CA_ENABLED' in debug.active
else:
    _ENFORCE_CA_ENABLED = False

from mvpa2.testing.sweep import sweepargs

if __debug__:
    debug('INIT', 'mvpa2.testing end')
