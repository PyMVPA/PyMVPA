#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA mappers."""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa.base import debug
    debug('INIT', 'mvpa.mappers')

from mvpa.mappers.mask import MaskMapper
from mvpa.mappers.pca import PCAMapper
from mvpa.mappers.svd import SVDMapper
from mvpa.mappers.boxcar import BoxcarMapper
from mvpa.mappers.array import DenseArrayMapper

if __debug__:
    debug('INIT', 'mvpa.mappers end')
