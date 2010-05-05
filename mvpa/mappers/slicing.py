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

from mvpa.mappers.base import Mapper, accepts_dataset_as_samples
from mvpa.base.dochelpers import _str


class SliceMapper(Mapper):
    """Baseclass of Mapper that slice a Dataset in various ways.
    """
    def __init__(self, slicearg, **kwargs):
        Mapper.__init__(self, **kwargs)
        # convert int sliceargs into lists to prevent getting scalar values when
        # slicing
        if isinstance(slicearg, int):
            slicearg = [slicearg]
        self._slicearg = slicearg


    def __str__(self):
        return _str(self)



class FeatureSliceMapper(SliceMapper):
    """Mapper to select a subset of features.
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
        self.__dshape = dshape
        self.__oshape = oshape
        self.filler = filler


    def __repr__(self):
        s = super(FeatureSliceMapper, self).__repr__()
        return s.replace("(",
                         "(slicearg=%s, dshape=%s, "
                          % (repr(self._slicearg), repr(self.__dshape)),
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
        if self.__oshape is None:
            self.__oshape = mdata.shape[1:]
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
        if not data.shape[:len(self.__oshape)] == self.__oshape:
            raise ValueError("Data shape does not match training "
                             "(trained: %s; got: %s)"
                             % (self.__dshape, data.shape))
        return super(FeatureSliceMapper, self).reverse1(data)


    def _reverse_data(self, data):
        """Reverse map data from featurespace into the original dataspace.

        Parameters
        ----------
        data : array-like
          Either one-dimensional sample or two-dimensional samples matrix.
        """
        if self.__dshape is None:
            raise RuntimeError(
                "Cannot reverse-map data since the original data shape is "
                "unknown. Either set `dshape` in the constructor, or call "
                "train().")
        # this wouldn't preserve ndarray subclasses
        #mapped = np.zeros(data.shape[:1] + self.__dshape,
        #                 dtype=data.dtype)
        # let's do it a little awkward but pass subclasses through
        # suggestions for improvements welcome
        mapped = data.copy() # make sure we own the array data
        # "guess" the shape of the final array, the following only supports
        # changes in the second axis -- the feature axis
        # this madness is necessary to support mapping of multi-dimensional
        # features
        mapped.resize(data.shape[:1] + self.__dshape + data.shape[2:],
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
        if self.__dshape is None:
            # XXX what about arrays of generic objects???
            # MH: in this case the shape will be (), which is just
            # fine since feature slicing is meaningless without features
            # the only thing we can do is kill the whole samples matrix
            self.__dshape = data.shape[1:]
            # we also need to know what the output shape looks like
            # otherwise we cannot reliably say what is appropriate input
            # for reverse*()
            self.__oshape = data[:, self._slicearg].shape[1:]


    def is_mergable(self, other):
        """Checks whether a mapper can be merged into this one.
        """
        if not isinstance(other, FeatureSliceMapper):
            return False
        # we can always merge if the slicing arg can be sliced itself (i.e. it
        # is not a slice-object... unless it doesn't really slice
        # we do not want to expand slices into index lists to become mergable,
        # since that would cause cheap view-based slicing to become expensive
        # copy-based slicing
        if isinstance(self._slicearg, slice) \
           and not self._slicearg == slice(None):
            return False

        return True


    def __iadd__(self, other):
        # the checker has to catch all awkward conditions
        if not self.is_mergable(other):
            raise ValueError("Mapper cannot be merged into target "
                             "(got: '%s', target: '%s')."
                             % (repr(other), repr(self)))

        # either replace non-slicing, or slice
        if isinstance(self._slicearg, slice) and self._slicearg == slice(None):
            self._slicearg = other._slicearg
            return self
        if isinstance(self._slicearg, list):
            # simply convert it into an array and proceed from there
            self._slicearg = np.asanyarray(self._slicearg)
        if self._slicearg.dtype.type is np.bool_:
            # simply convert it into an index array --prevents us from copying a
            # lot and allows for sliceargs such as [3,3,4,4,5,5]
            self._slicearg = self._slicearg.nonzero()[0]
            # do not return since it needs further processing
        if self._slicearg.dtype.char in np.typecodes['AllInteger']:
            self._slicearg = self._slicearg[other._slicearg]
            return self

        raise RuntimeError("This should not happen. Undetected condition!")
