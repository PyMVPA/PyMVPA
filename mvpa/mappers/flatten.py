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

from mvpa.mappers.base import Mapper, accepts_dataset_as_samples
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
        self.__coord_helper = None
        if not shape is None:
            self._train_with_shape(shape)


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
        if N.isfortran(samples):
            raise ValueError("FlattenMapper currently works for C-ordered "
                             "arrays only.")
        self._train_with_shape(samples.shape[1:])


    def _train_with_shape(self, shape):
        """Configure the mapper with a particular sample shape.
        """
        # infer the sample shape from the data under the assumption that the
        # first axis is the samples-separating dimension
        self.__origshape = shape
        # total number of features in a sample
        self.__nfeatures = N.prod(self.__origshape)
        # compute a vector that aids computing in coords into feature ids
        # this whole thing only works for C-ordered arrays
        self.__coord_helper = \
                N.array([N.prod(self.__origshape[i:])
                            for i in range(1, len(self.__origshape))]
                        + [1])


    def _forward_data(self, data):
        if N.isfortran(data):
            raise ValueError("FlattenMapper currently works for C-ordered "
                             "arrays only.")

        # local binding
        dshape = data.shape
        oshape = self.__origshape
        nfeatures = self.__nfeatures

        if oshape is None:
            raise RuntimeError("FlattenMapper needs to be trained before it "
                               "can be used.")

        # if data is a single samples
        if dshape == oshape:
            return data.reshape(-1)
        # multiple samples
        elif dshape[1:] == oshape:
            return data.reshape(dshape[0], -1)

        raise ValueError("FlattenMapper has not been trained for data "
                         "shape '%s' (known only '%s')."
                         % (str(dshape), str(oshape)))

    def _forward_dataset(self, dataset):
        # invoke super class _forward_dataset, this calls, _forward_dataset
        # and this calles _forward_data in this class
        mds = super(FlattenMapper, self)._forward_dataset(dataset)
        # if there is no inspace return immediately
        if self.get_inspace() is None:
            return mds
        # otherwise create the coordinates as feature attributes
        else:
            mds.fa[self.get_inspace()] = \
            N.transpose(N.isfinite(dataset.samples[0]).nonzero())
            return mds

    def _reverse_data(self, data):
        if N.isfortran(data):
            raise ValueError("FlattenMapper currently works for C-ordered "
                             "arrays only.")
        # local binding
        dshape = data.shape
        oshape = self.__origshape
        nfeatures = self.__nfeatures

        # if data is a single samples
        if len(data) == nfeatures:
            return data.reshape(oshape)
        # multiple samples
        elif dshape[1:] == (nfeatures,):
            return data.reshape((dshape[0],) + oshape)

        raise ValueError("FlattenMapper has not been trained for data "
                         "shape '%s'."% str(dshape))


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


    def _get_outids(self, in_ids):
        # this whole thing only works for C-ordered arrays
        return [N.sum(self.__coord_helper * N.asanyarray(in_id))
                    for in_id in in_ids]


    def get_insize(self):
        """Return the (flattened) size of input space vectors."""
        return self.__nfeatures


    def get_outsize(self):
        """Return the size of output space vectors."""
        return self.__nfeatures



