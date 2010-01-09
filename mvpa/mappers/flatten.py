# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Data mapper"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.base.dochelpers import _str
from mvpa.mappers.base import Mapper, accepts_dataset_as_samples, \
        ChainMapper, FeatureSliceMapper
from mvpa.misc.support import isInVolume


class FlattenMapper(Mapper):
    """Reshaping mapper that flattens multidimensional arrays into 1D vectors.

    This mapper performs relatively cheap reshaping of arrays from ND into 1D
    and back upon reverse-mapping. The mapper has to be trained with a data
    array or dataset that has the first axis as the samples-separating
    dimension. Mapper training will set the particular multidimensional shape
    the mapper is transforming into 1D vector samples. The setting remains in
    place until the mapper is retrained.

    Note
    ----
    At present this mapper is only designed (and tested) to work with C-ordered
    arrays.
    """
    def __init__(self, shape=None, **kwargs):
        """
        Parameters
        ----------
        shape : tuple
          The shape of a single sample. If this argument is given the mapper
          is going to be fully configured and no training is necessary anymore.
        """
        Mapper.__init__(self, **kwargs)
        self.__origshape = None
        self.__nfeatures = None
        if not shape is None:
            self._train_with_shape(shape)


    def __repr__(self):
        s = Mapper.__repr__(self)
        m_repr = 'shape=%s' % repr(self.__origshape)
        return s.replace("(", "(%s, " % m_repr, 1)


    def __str__(self):
        return _str(self)


    @accepts_dataset_as_samples
    def _train(self, samples):
        """Train the mapper.

        Parameters
        ----------
        samples : array-like
          The first axis has to represent the samples-separating dimension. In
          case of a 1D-array each element is considered to be an individual
          element and *not* the whole array as a single sample!
        """
        self._train_with_shape(samples.shape[1:])


    def _train_with_shape(self, shape):
        """Configure the mapper with a particular sample shape.
        """
        # infer the sample shape from the data under the assumption that the
        # first axis is the samples-separating dimension
        self.__origshape = shape
        # total number of features in a sample
        self.__nfeatures = N.prod(self.__origshape)


    def _forward_data(self, data):
        # this method always gets data where the first axis is the samples axis!
        # local binding
        nsamples = data.shape[0]
        sshape = data.shape[1:]
        oshape = self.__origshape
        nfeatures = self.__nfeatures

        if oshape is None:
            raise RuntimeError("FlattenMapper needs to be trained before it "
                               "can be used.")

        # input matches the shape of a single sample
        if sshape == oshape:
            return data.reshape(nsamples, -1)
        # the first part of the shape matches (e.g. some additional axes present)
        elif sshape[:len(oshape)] == oshape:
            # flatten the pieces the mapper knows about and preserve the rest
            return data.reshape((nsamples, -1) + sshape[len(oshape):])

        raise ValueError("FlattenMapper has not been trained for data "
                         "shape '%s' (known only '%s')."
                         % (str(sshape), str(oshape)))


    def _forward_dataset(self, dataset):
        # invoke super class _forward_dataset, this calls, _forward_dataset
        # and this calls _forward_data in this class
        mds = super(FlattenMapper, self)._forward_dataset(dataset)
        # attribute collection needs to have a new length check
        mds.fa.set_length_check(mds.nfeatures)
        # now flatten all feature attributes
        for k in mds.fa:
            mds.fa[k] = self.forward1(mds.fa[k].value)

        # if there is no inspace return immediately
        if self.get_inspace() is None:
            return mds
        # otherwise create the coordinates as feature attributes
        else:
            mds.fa[self.get_inspace()] = \
                list(N.ndindex(dataset.samples[0].shape))
            return mds


    def _reverse_data(self, data):
        # this method always gets data where the first axis is the samples axis!
        # local binding
        nsamples = data.shape[0]
        sshape = data.shape[1:]
        oshape = self.__origshape
        nfeatures = self.__nfeatures
        return data.reshape((nsamples,) + sshape[:-1] + oshape)

        # XXX anything we cannot reverse-map?
        raise ValueError("FlattenMapper has not been trained for data "
                         "with shape '%s', but '%s'."
                         % (str(dshape[1:]), (nfeatures,)))


    def _reverse_dataset(self, dataset):
        # invoke super class _reverse_dataset, this calls, _reverse_dataset
        # and this calles _reverse_data in this class
        mds = super(FlattenMapper, self)._reverse_dataset(dataset)
        # attribute collection needs to have a new length check
        mds.fa.set_length_check(mds.nfeatures)
        # now unflatten all feature attributes
        inspace = self.get_inspace()
        for k in mds.fa:
            # reverse map all attributes, but not the inspace indices, since the
            # did not come through this mapper and make not sense in inspace
            if k != inspace:
                mds.fa[k] = self.reverse1(mds.fa[k].value)
        # wipe out the inspace attribute -- needs to be done after the loop to
        # not change the size of the dict
        if inspace and inspace in mds.fa:
            del mds.fa[inspace]
        return mds


    def is_valid_outid(self, id):
        """Checks for a valid output id for this (trained) mapper).

        If the mapper is not trained any id is invalid.
        """
        # untrained -- all is invalid
        if self.__nfeatures is None:
            return False
        return id >= 0 and id < self.__nfeatures


    def is_valid_inid(self, id):
        """Checks for a valid output id for this (trained) mapper).

        If the mapper is not trained any id is invalid.
        """
        # untrained -- all is invalid
        if self.__nfeatures is None:
            return False
        # check for proper coordinate (also handle the case of 1d coords given
        # as scalars
        if N.isscalar(id):
            # scalar id but multiple dimensions -> wrong
            if len(self.__origshape) > 1:
                return False
            # scalar id and scalar data -> id must be 0
            elif not len(self.__origshape):
                return id == 0
            # otherwise no flattening is done and inid == outid
            else:
                return self.is_valid_outid(id)
        if len(id) == len(self.__origshape):
            return isInVolume(id, self.__origshape)

        return False


    def get_insize(self):
        """Return the (flattened) size of input space vectors."""
        return self.__nfeatures


    def get_outsize(self):
        """Return the size of output space vectors."""
        return self.__nfeatures



def mask_mapper(mask=None, shape=None, inspace=None):
    """
    Parameters
    ----------
    mask : array
      an array in the original dataspace and its nonzero elements are
      used to define the features included in the dataset. alternatively,
      the `shape` argument can be used to define the array dimensions.
    shape : tuple
      The shape of the array to be mapped. If `shape` is provided instead
      of `mask`, a full mask (all True) of the desired shape is
      constructed. If `shape` is specified in addition to `mask`, the
      provided mask is extended to have the same number of dimensions.
    """
    if mask is None:
        if shape is None:
            raise ValueError, \
                  "Either `shape` or `mask` have to be specified."
        else:
            # make full dataspace mask if nothing else is provided
            mask = N.ones(shape, dtype='bool')
    else:
        if not shape is None:
            # expand mask to span all dimensions but first one
            # necessary e.g. if only one slice from timeseries of volumes is
            # requested.
            mask = N.asanyarray(mask, ndmin=len(shape))
            # check for compatibility
            if not shape == mask.shape:
                raise ValueError, \
                    "The mask dataspace shape %s is not " \
                    "compatible with the provided shape %s." \
                    % (mask.shape, shape)

    fm = FlattenMapper(shape=mask.shape, inspace=inspace)
    flatmask = fm.forward1(mask)
    mapper = ChainMapper([fm, FeatureSliceMapper(flatmask,
                                                 dshape=flatmask.shape)])
    return mapper
