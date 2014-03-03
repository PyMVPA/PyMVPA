# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Flatten multi-dimensional samples"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base.dochelpers import _str, _repr_attrs
from mvpa2.mappers.base import Mapper, accepts_dataset_as_samples, \
        ChainMapper
from mvpa2.featsel.base import StaticFeatureSelection
from mvpa2.misc.support import is_in_volume

if __debug__:
    from mvpa2.base import debug

class FlattenMapper(Mapper):
    """Reshaping mapper that flattens multidimensional arrays into 1D vectors.

    This mapper performs relatively cheap reshaping of arrays from ND into 1D
    and back upon reverse-mapping. The mapper has to be trained with a data
    array or dataset that has the first axis as the samples-separating
    dimension. Mapper training will set the particular multidimensional shape
    the mapper is transforming into 1D vector samples. The setting remains in
    place until the mapper is retrained.

    Notes
    -----
    At present this mapper is only designed (and tested) to work with C-ordered
    arrays.
    """
    def __init__(self, shape=None, maxdims=None, **kwargs):
        """
        Parameters
        ----------
        shape : tuple
          The shape of a single sample. If this argument is given the mapper
          is going to be fully configured and no training is necessary anymore.
        maxdims : int or None
          The maximum number of dimensions to flatten (starting with the first).
          If None, all axes will be flattened.
        """
        # by default auto train
        kwargs['auto_train'] = kwargs.get('auto_train', True)
        Mapper.__init__(self, **kwargs)
        self.__origshape = None         # pylint pacifier
        self.__maxdims = maxdims
        if not shape is None:
            self._train_with_shape(shape)

    def __repr__(self, prefixes=[]):
        return super(FlattenMapper, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['shape', 'maxdims']))

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
        # flag the mapper as trained
        self._set_trained()


    def _forward_data(self, data):
        # this method always gets data where the first axis is the samples axis!
        # local binding
        nsamples = data.shape[0]
        sshape = data.shape[1:]
        oshape = self.__origshape

        if oshape is None:
            raise RuntimeError("FlattenMapper needs to be trained before it "
                               "can be used.")
        # at least the first feature axis has to match match
        if oshape[0] != sshape[0]:
            raise ValueError("FlattenMapper has not been trained for data "
                             "shape '%s' (known only '%s')."
                             % (str(sshape), str(oshape)))
        ## input matches the shape of a single sample
        #if sshape == oshape:
        #    return data.reshape(nsamples, -1)
        ## the first part of the shape matches (e.g. some additional axes present)
        #elif sshape[:len(oshape)] == oshape:
        if not self.__maxdims is None:
            maxdim = min(len(oshape), self.__maxdims)
        else:
            maxdim = len(oshape)
        # flatten the pieces the mapper knows about and preserve the rest
        return data.reshape((nsamples, -1) + sshape[maxdim:])



    def _forward_dataset(self, dataset):
        # invoke super class _forward_dataset, this calls, _forward_dataset
        # and this calls _forward_data in this class
        mds = super(FlattenMapper, self)._forward_dataset(dataset)
        # attribute collection needs to have a new length check
        mds.fa.set_length_check(mds.nfeatures)
        # we need to duplicate all existing feature attribute, as each original
        # feature is now spread across the new feature axis
        # take all "additional" axes after the actual feature axis and count
        # elements a sample -- if not axis exists this will be 1
        for k in dataset.fa:
            if __debug__:
                debug('MAP_', "Forward-mapping fa '%s'." % k)
            attr = dataset.fa[k].value
            # the maximmum number of axis to flatten in the attr
            if not self.__maxdims is None:
                maxdim = min(len(self.__origshape), self.__maxdims)
            else:
                maxdim = len(self.__origshape)
            multiplier = mds.nfeatures \
                    / np.prod(attr.shape[:maxdim])
            if __debug__:
                debug('MAP_', "Broadcasting fa '%s' %s %d times"
                        % (k, attr.shape, multiplier))
            # broadcast as many times as necessary to get 'matching dimensions'
            bced = np.repeat(attr, multiplier, axis=0)
            # now reshape as many dimensions as the mapper knows about
            mds.fa[k] = bced.reshape((-1,) + bced.shape[maxdim:])

        # if there is no inspace return immediately
        if self.get_space() is None:
            return mds
        # otherwise create the coordinates as feature attributes
        else:
            mds.fa[self.get_space()] = \
                list(np.ndindex(dataset.samples[0].shape))
            return mds


    def _reverse_data(self, data):
        # this method always gets data where the first axis is the samples axis!
        # local binding
        nsamples = data.shape[0]
        sshape = data.shape[1:]
        oshape = self.__origshape
        return data.reshape((nsamples,) + oshape + sshape[1:])


    def _reverse_dataset(self, dataset):
        # invoke super class _reverse_dataset, this calls, _reverse_dataset
        # and this calles _reverse_data in this class
        mds = super(FlattenMapper, self)._reverse_dataset(dataset)
        # attribute collection needs to have a new length check
        mds.fa.set_length_check(mds.nfeatures)
        # now unflatten all feature attributes
        inspace = self.get_space()
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

    shape = property(fget=lambda self:self.__origshape)
    maxdims = property(fget=lambda self:self.__maxdims)

class ProductFlattenMapper(FlattenMapper):
    """Reshaping mapper that flattens multidimensional arrays and
    preserves information for each dimension in feature attributes

    Notes
    -----
    This class' name contains 'product' because it maps feature
    attributes in a cartesian-product way."""

    def __init__(self, factor_names, **kwargs):
        '''
        Parameters
        ----------
        factor_names: iterable
            The names and values for each dimension. If the dataset to
            be flattened is shaped ns X nf1 x nf2 x ... x nfN, then
            factor_names should have a length of N. Furthermore
            when applied to a dataset ds, it should have each
            of the factor names factor_names[K] as an attribute and the value
            of this attribute should have nfK values.
            Applying this mapper to such a dataset yields a new dataset
            with size ns X (nf1 * nf2 * ... * nfN) with
            feature attributes nameK and nameKindices for each nameK
            in the factor names.
        reverse_set_fa: bool
        '''
        kwargs['auto_train'] = kwargs.get('auto_train', True)

        # make sure the factor names and values are properly set
        factor_names = list(factor_names)
        space = '_'.join(factor_names) + '_indices'

        FlattenMapper.__init__(self, space=space, **kwargs)

        self._factor_names = factor_names
        self._factor_values = None

    def __repr__(self, prefixes=[]):
        return super(ProductFlattenMapper, self).__repr__(
                        prefixes=prefixes
                        + _repr_attrs(self, ['factor_names']))

    @property
    def factor_names(self):
        return self._factor_names

    def _train(self, ds):
        super(ProductFlattenMapper, self)._train(ds)
        self._factor_values = []
        for nm in self._factor_names:
            if not nm in ds.a.keys():
                raise KeyError("Missing attribute: %s" % nm)
            self._factor_values.append(ds.a[nm].value)


    def _untrain(self):
        self._factor_values = None

    def _check_factor_name_values(self, ds):
        ### currently unished...
        if self._factor_values is None:
            raise ValueError("Dataset is not trained")
        for nm, value in zip(*(self._factor_names, self._factor_values)):
            if any(ds.a[nm].value != value):
                raise ValueError("Mismatch for attribute %s: %s != %s" %
                                        (nm, value, ds.a[nm].value))


    def _forward_dataset(self, dataset):
        self._train(dataset)

        mds = super(ProductFlattenMapper, self)._forward_dataset(dataset)

        oshape = self.shape

        factor_names_values = zip(*(self._factor_names, self._factor_values))
        # now map all the factor names and values to feature attributes
        for i, (name, value) in enumerate(factor_names_values):
            # keep track of both the value itself and the indices
            for repr, postfix in ((value, None),
                                  (np.arange(len(value)), '_indices')):

                nshape = [1] + list(oshape) # full shape with one sample
                nshape[i + 1] = 1 # dimension of current factor

                # shape for repr with 1 value at all dimensions except the
                # current one. In other words nshapa and ushape complement
                # each other.
                ushape = [1] * len(nshape)
                ushape[i + 1] = len(value)

                # reshape and tile
                repr_rs = np.reshape(np.asarray(repr), ushape)
                repr_arr = np.tile(repr_rs, nshape)

                # ensure that values have the proper shape
                if repr_arr.shape[1:] != oshape:
                    raise ValueError("Shape mismatch: %s != %s - this should"
                                    " not happen" % ((repr_arr.shape,), (oshape,)))

                # flatten the attributes
                repr_flat = self.forward(repr_arr)
                # assigne as feature attribute
                fa_label = name if postfix is None else name + postfix

                mds.fa[fa_label] = repr_flat.ravel()

            del mds.a[name]

        return mds

    def _reverse_dataset(self, dataset):
        #self._train(dataset)
        factor_names_values = zip(*(self._factor_names, self._factor_values))

        mds = super(ProductFlattenMapper, self)._reverse_dataset(dataset)
        postfix = '_indices'
        for name, _ in factor_names_values:
            label = name + postfix
            if label in mds.fa:
                del mds.fa[label]

        for nm, values in factor_names_values: #:self._get_reversed_factor_name_values(mds):
            if nm in mds.a.keys() and any(mds.a[nm].value != values):
                raise ValueError("name clash for %s" % nm)
            del mds.fa[nm]
            mds.a[nm] = values

        return mds


    def _get_reversed_factor_name_values(self, reversed_dataset):
        '''Helper function to trace back the original names and values
        after the mapper reversed a dataset.

        Parameters
        ----------
        reversed_dataset: Dataset
            The instance that was reversed using this instance

        Returns
        -------
        factor_names_values: iterable
            The names and values for each dimension. If the reversed_dataset
            is shaped ns X nf1 x nf2 x ... x nfN, then
            factor_names_values will have a length of N. Furthermore
            the K-th element in factor_names_values is a tuple
            (nameK, valueK) where nameK is a string and valueK has
            length nfK.

        Notes
        -----
        It is not guaranteed that the order of valueK matches the
        corresponding element in self.factor_names_values that was
        supplied to the __init__ method
        '''

        output_factor_names = []
        factor_names_values = zip(*(self._factor_names, self._factor_values))
        for dim, (name, values) in enumerate(factor_names_values):
            vs = reversed_dataset.fa[name].value

            if dim > 0:
                vs = vs.swapaxes(0, dim)

            n = vs.shape[0]
            print values
            print n

            print len(values)

            assert(n == len(values))

            unq_vs = []
            for i in xrange(n):
                v = vs[i, ...]
                unq_v = np.unique(v)
                assert(unq_v.size == 1)
                assert(unq_v in values)
                unq_vs.append(unq_v)

            output_factor_names.append((name, np.asarray(unq_vs).ravel()))

        return output_factor_names


def mask_mapper(mask=None, shape=None, space=None):
    """Factory method to create a chain of Flatten+StaticFeatureSelection Mappers

    Parameters
    ----------
    mask : None or array
      an array in the original dataspace and its nonzero elements are
      used to define the features included in the dataset. Alternatively,
      the `shape` argument can be used to define the array dimensions.
    shape : None or tuple
      The shape of the array to be mapped. If `shape` is provided instead
      of `mask`, a full mask (all True) of the desired shape is
      constructed. If `shape` is specified in addition to `mask`, the
      provided mask is extended to have the same number of dimensions.
    inspace
      Provided to `FlattenMapper`
    """
    if mask is None:
        if shape is None:
            raise ValueError, \
                  "Either `shape` or `mask` have to be specified."
        else:
            # make full dataspace mask if nothing else is provided
            mask = np.ones(shape, dtype='bool')
    else:
        if not shape is None:
            # expand mask to span all dimensions but first one
            # necessary e.g. if only one slice from timeseries of volumes is
            # requested.
            mask = np.array(mask, copy=False, subok=True, ndmin=len(shape))
            # check for compatibility
            if not shape == mask.shape:
                raise ValueError, \
                    "The mask dataspace shape %s is not " \
                    "compatible with the provided shape %s." \
                    % (mask.shape, shape)

    fm = FlattenMapper(shape=mask.shape, space=space)
    flatmask = fm.forward1(mask)
    mapper = ChainMapper([fm,
                          StaticFeatureSelection(
                              flatmask,
                              dshape=flatmask.shape,
                              oshape=(len(flatmask.nonzero()[0]),))])
    return mapper
