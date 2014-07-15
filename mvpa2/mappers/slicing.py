# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Collection of dataset slicing procedures."""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base.node import Node
from mvpa2.mappers.base import Mapper, accepts_dataset_as_samples
from mvpa2.base.dochelpers import _str, _repr_attrs
from mvpa2.generators.splitters import mask2slice


class SliceMapper(Mapper):
    """Baseclass of Mapper that slice a Dataset in various ways.
    """
    def __init__(self, slicearg, **kwargs):
        """
        Parameters
        ----------
        slicearg
          Argument for slicing
        """
        Mapper.__init__(self, **kwargs)
        self._safe_assign_slicearg(slicearg)


    def _safe_assign_slicearg(self, slicearg):
        # convert int sliceargs into lists to prevent getting scalar values when
        # slicing
        if isinstance(slicearg, int):
            slicearg = [slicearg]
        self._slicearg = slicearg
        # if we got some sort of slicearg we assume that we are ready to go
        if not slicearg is None:
            self._set_trained()

    def __repr__(self, prefixes=[]):
        return super(SliceMapper, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['slicearg']))


    def __str__(self):
        # with slicearg it can quickly get very unreadable
        #return _str(self, str(self._slicearg))
        return _str(self)


    def _untrain(self):
        self._safe_assign_slicearg(None)
        super(SliceMapper, self)._untrain()


    def __iadd__(self, other):
        # our slicearg
        this = self._slicearg
        # if another slice mapper work on its slicearg
        if isinstance(other, SliceMapper):
            other = other._slicearg
        # catch stupid arg
        if not (isinstance(other, tuple) or isinstance(other, list) \
                or isinstance(other, np.ndarray) or isinstance(other, slice)):
            return NotImplemented
        if isinstance(this, slice):
            # we can always merge if the slicing arg can be sliced itself (i.e.
            # it is not a slice-object... unless it doesn't really slice we do
            # not want to expand slices into index lists to become mergable,
            # since that would cause cheap view-based slicing to become
            # expensive copy-based slicing
            if this == slice(None):
                # this one did nothing, just use the other and be done
                self._safe_assign_slicearg(other)
                return self
            else:
                # see comment above
                return NotImplemented
        # list or tuple are alike
        if isinstance(this, list) or isinstance(this, tuple):
            # simply convert it into an array and proceed from there
            this = np.asanyarray(this)
        if this.dtype.type is np.bool_:
            # simply convert it into an index array --prevents us from copying a
            # lot and allows for sliceargs such as [3,3,4,4,5,5]
            this = this.nonzero()[0]
        if this.dtype.char in np.typecodes['AllInteger']:
            self._safe_assign_slicearg(this[other])
            return self

        # if we get here we got something the isn't supported
        return NotImplemented

    slicearg = property(fget=lambda self:self._slicearg)


class SampleSliceMapper(SliceMapper):
    """Mapper to select a subset of samples."""
    def __init__(self, slicearg, **kwargs):
        """
        Parameters
        ----------
        slicearg : int, list(int), array(int), array(bool)
          Any slicing argument that is compatible with numpy arrays. Depending
          on the argument the mapper will perform basic slicing or
          advanced indexing (with all consequences on speed and memory
          consumption).
        """
        SliceMapper.__init__(self, slicearg, **kwargs)


    def _call(self, ds):
        # it couldn't be simpler
        return ds[self._slicearg]



class StripBoundariesSamples(Node):
    """Strip samples on boundaries defines by sample attribute values.

    A sample attribute of a dataset is scanned for consecutive blocks of
    identical values. Every change in the value is treated as a boundary
    and custom number of samples is removed prior and after this boundary.
    """
    def __init__(self, space, prestrip, poststrip, **kwargs):
        """
        Parameters
        ----------
        space : str
          name of the sample attribute that shall be used to determine the
          boundaries.
        prestrip : int
          Number of samples to be stripped prior to each boundary.
        poststrip : int
          Number of samples to be stripped after each boundary (this includes
          the boundary sample itself, i.e. the first samples with a different
          sample attribute value).
        """
        Node.__init__(self, space=space, **kwargs)
        self._prestrip = prestrip
        self._poststrip = poststrip


    def _call(self, ds):
        # attribute to detect boundaries
        battr = ds.sa[self.get_space()].value
        # filter which samples to keep
        filter_ = np.ones(battr.shape, dtype='bool')
        # determine boundary indices -- shift by one to have the new value
        # as the boundary
        bindices = (battr[:-1] != battr[1:]).nonzero()[0] + 1

        # for all boundaries
        for b in bindices:
            lower = b - self._prestrip
            upper = b + self._poststrip
            filter_[lower:upper] = False

        filter_ = mask2slice(filter_)

        return ds[filter_]
