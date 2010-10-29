# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Mappers for Dataset slicing."""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa.base.node import Node
from mvpa.mappers.base import Mapper, accepts_dataset_as_samples
from mvpa.base.dochelpers import _str
from mvpa.generators.splitters import mask2slice


class SliceMapper(Mapper):
    """Baseclass of Mapper that slice a Dataset in various ways.
    """
    def __init__(self, slicearg, **kwargs):
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


    def __str__(self):
        return _str(self, str(self._slicearg))


    def _untrain(self):
        self._slicearg = None
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



class FeatureSliceMapper(SliceMapper):
    """Mapper to select a subset of features.

    Depending on the actual slicing argument two FeatureSliceMappers can be
    merged in a number of ways: incremental selection (+=), union (&=) and
    intersection (|=).  Were the former assumes that two feature selections are
    applied subsequently, and the latter two assume that both slicings operate
    on the set of input features.

    Examples
    --------
    >>> from mvpa.datasets import *
    >>> ds = Dataset([[1,2,3,4,5]])
    >>> fs0 = FeatureSliceMapper([0,1,2,3])
    >>> fs0(ds).samples
    array([[1, 2, 3, 4]])

    Merge two incremental selections: the resulting mapper performs a selection
    that is equivalent to first applying one slicing and subsequently the next
    slicing. In this scenario the slicing argument of the second mapper is
    relative to the output feature space of the first mapper.

    >>> fs1 = FeatureSliceMapper([0,2])
    >>> fs0 += fs1
    >>> fs0(ds).samples
    array([[1, 3]])
    """
    def __init__(self, slicearg, dshape=None, oshape=None, filler=0, **kwargs):
        """
        Parameters
        ----------
        slicearg : int, list(int), array(int), array(bool)
          Any slicing argument that is compatible with numpy arrays. Depending
          on the argument the mapper will perform basic slicing or
          advanced indexing (with all consequences on speed and memory
          consumption).
        dshape : tuple
          Preseed the mappers input data shape (single sample shape).
        oshape: tuple
          Preseed the mappers output data shape (single sample shape).
        filler : optional
          Value to fill empty entries upon reverse operation
        """
        SliceMapper.__init__(self, slicearg, **kwargs)
        # store it here, might be modified later
        self._dshape = dshape
        self._oshape = oshape
        self.filler = filler


    def __repr__(self):
        s = super(FeatureSliceMapper, self).__repr__()
        return s.replace("(",
                         "(slicearg=%s, dshape=%s, "
                          % (repr(self._slicearg), repr(self._dshape)),
                         1)


    def _forward_data(self, data):
        """Map data from the original dataspace into featurespace.

        Parameters
        ----------
        data : array-like
          Either one-dimensional sample or two-dimensional samples matrix.
        """
        mdata = data[:, self._slicearg]
        # store the output shape if not set yet
        if self._oshape is None:
            self._oshape = mdata.shape[1:]
        return mdata


    def _forward_dataset(self, dataset):
        # XXX this should probably not affect the source dataset, but right now
        # init_origid is not flexible enough
        if not self.get_space() is None:
            # TODO need to do a copy first!!!
            dataset.init_origids('features', attr=self.get_space())
        # invoke super class _forward_dataset, this calls, _forward_dataset
        # and this calles _forward_data in this class
        mds = super(FeatureSliceMapper, self)._forward_dataset(dataset)
        # attribute collection needs to have a new length check
        mds.fa.set_length_check(mds.nfeatures)
        # now slice all feature attributes
        for k in mds.fa:
            mds.fa[k] = self.forward1(mds.fa[k].value)
        return mds


    def reverse1(self, data):
        # we need to reject inappropriate "single" samples to allow
        # chainmapper to properly switch to reverse() for multiple samples
        # use the fact that a single sample needs to conform to the known
        # data shape -- but may have additional appended dimensions
        if not data.shape[:len(self._oshape)] == self._oshape:
            raise ValueError("Data shape does not match training "
                             "(trained: %s; got: %s)"
                             % (self._dshape, data.shape))
        return super(FeatureSliceMapper, self).reverse1(data)


    def _reverse_data(self, data):
        """Reverse map data from featurespace into the original dataspace.

        Parameters
        ----------
        data : array-like
          Either one-dimensional sample or two-dimensional samples matrix.
        """
        if self._dshape is None:
            raise RuntimeError(
                "Cannot reverse-map data since the original data shape is "
                "unknown. Either set `dshape` in the constructor, or call "
                "train().")
        # this wouldn't preserve ndarray subclasses
        #mapped = np.zeros(data.shape[:1] + self._dshape,
        #                 dtype=data.dtype)
        # let's do it a little awkward but pass subclasses through
        # suggestions for improvements welcome
        mapped = data.copy() # make sure we own the array data
        # "guess" the shape of the final array, the following only supports
        # changes in the second axis -- the feature axis
        # this madness is necessary to support mapping of multi-dimensional
        # features
        mapped.resize(data.shape[:1] + self._dshape + data.shape[2:],
                      refcheck=False)
        mapped.fill(self.filler)
        mapped[:, self._slicearg] = data
        return mapped


    def _reverse_dataset(self, dataset):
        # invoke super class _reverse_dataset, this calls, _reverse_dataset
        # and this calles _reverse_data in this class
        mds = super(FeatureSliceMapper, self)._reverse_dataset(dataset)
        # attribute collection needs to have a new length check
        mds.fa.set_length_check(mds.nfeatures)
        # now reverse all feature attributes
        for k in mds.fa:
            mds.fa[k] = self.reverse1(mds.fa[k].value)
        return mds


    @accepts_dataset_as_samples
    def _train(self, data):
        if self._dshape is None:
            # XXX what about arrays of generic objects???
            # MH: in this case the shape will be (), which is just
            # fine since feature slicing is meaningless without features
            # the only thing we can do is kill the whole samples matrix
            self._dshape = data.shape[1:]
            # we also need to know what the output shape looks like
            # otherwise we cannot reliably say what is appropriate input
            # for reverse*()
            self._oshape = data[:, self._slicearg].shape[1:]


    def _untrain(self):
        self._dshape = None
        self._oshape = None
        super(FeatureSliceMapper, self)._untrain()



class StripBoundariesSamples(Node):
    """Strip samples on boundaries defines by sample attribute values.

    A sample attribute of a dataset is scanned for consecutive blocks if
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
