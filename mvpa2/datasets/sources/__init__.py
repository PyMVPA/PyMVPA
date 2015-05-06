# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Datasets originating from code outside of PyMVPA

Contains wrapper and adapter methods interfacing data provided by other
packages as PyMVPA datasets.
"""

__docformat__ = 'restructuredtext'

from mvpa2.base import externals
from .openfmri import *
from .native import *

if externals.exists('skl'):
    from .skl_data import *
