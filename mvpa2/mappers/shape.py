# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Basic dataset shape modifications."""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.mappers.base import Mapper
from mvpa2.datasets import Dataset

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
        out = Dataset(np.swapaxes(ds.samples, 0, 1),
                      sa=ds.fa,
                      fa=ds.sa,
                      a=ds.a)
        return out

    def _forward_dataset(self, ds):
        return self._swap_samples_and_feature_axes(ds)

    def _reverse_dataset(self, ds):
        return self._swap_samples_and_feature_axes(ds)


class AddAxisMapper(Mapper):
    """Add an axis to a dataset at an arbitrary position.

    This mapper can be useful when there is need for aggregating multiple
    datasets, where it is often necessary or at least useful to have a
    dedicated aggregation axis.  An axis can be added at any position

    When adding an axis that causes the current sample (1st) or feature axis
    (2nd) to shift the corresponding attribute collections are modified to
    accomodate the change. This typically means also adding an axis at the
    corresponding position of the attribute arrays. A special case is, however,
    prepending an axis to the dataset, i.e. shifting both sample and feature
    axis towards the back. In this case all feature attibutes are duplicated
    to match the new number of features (formaly the number of samples).

    Examples
    --------
    >>> from mvpa2.datasets.base import Dataset
    >>> from mvpa2.mappers.shape import AddAxisMapper
    >>> ds = Dataset(np.arange(24).reshape(2,3,4))
    >>> am = AddAxisMapper(pos=1)
    >>> print am(ds).shape
    (2, 1, 3, 4)
    """
    is_trained = True

    def __init__(self, pos, **kwargs):
        """
        Parameters
        ----------
        pos : int
            Axis index to which the new axis is prepended. Negative indices are
            supported as well, but the new axis will be placed behind the given
            index. For example, a position of ``-1`` will cause an axis to be
            added behind the last axis. If ``pos`` is larger than the number of
            existing axes additional new axes will be created match the value of
            ``pos``.
        """
        Mapper.__init__(self, **kwargs)
        self._pos = pos

    def _forward_dataset(self, ds):
        pos = self._pos
        if pos < 0:
            # support negative/reverse indices
            pos = len(ds.shape) + 1 + pos
        # select all prior axes, but at most all existing axes
        slicer = [slice(None)] * min(pos, len(ds.shape))
        # and as many new axes as necessary afterwards
        slicer += [None] * max(1, pos + 1 - len(ds.shape))
        # there are two special cases that require modification of feature
        # attributes
        if pos == 0:
            # prepend an axis to all sample attributes
            out_sa = dict([(attr, ds.sa[attr].value[None]) for attr in ds.sa])
            # prepend an axis to all FAs and repeat for each previous sample
            out_fa = dict([(attr,
                            np.repeat(ds.fa[attr].value[None], len(ds), axis=0))
                for attr in ds.fa])
        elif pos == 1:
            # prepend an axis to all feature attributes
            out_fa = dict([(attr, ds.fa[attr].value[None]) for attr in ds.fa])
            out_sa = ds.sa
        else:
            out_sa = ds.sa
            out_fa = ds.fa
        out = Dataset(ds.samples.__getitem__(tuple(slicer)),
                      sa=out_sa, fa=out_fa, a=ds.a)
        return out
