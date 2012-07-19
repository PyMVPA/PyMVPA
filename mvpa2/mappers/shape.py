# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Mappers that modify the shape of a dataset"""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.mappers.base import Mapper

class TransposeMapper(Mapper):
    """Swap sample and feature axes.

    This mapper swaps the sample axis (first axis) and feature axis (second
    axis) of a dataset (additional axes in multi-dimensional datasets are left
    untouched). Both, sample and feature attribute collections are also
    swapped accordingly. Neither dataset samples, not attribute collections
    are copied. Reverse mapping is supported as well. This mapper does not
    require training and a single instance can be used on different datasets
    without problems.
    """
    is_trained = True

    def __init__(self,  **kwargs):
        Mapper.__init__(self, **kwargs)

    def _swap_samples_and_feature_axes(self, ds):
        ds.samples = np.swapaxes(ds.samples, 0, 1)
        swap = ds.fa
        ds.fa = ds.sa
        ds.sa = swap
        return ds

    def _forward_dataset(self, ds):
        return self._swap_samples_and_feature_axes(ds)

    def _reverse_dataset(self, ds):
        return self._swap_samples_and_feature_axes(ds)
