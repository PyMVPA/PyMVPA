# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Data normalization by Z-Scoring."""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import warning
from mvpa2.base.dochelpers import _str, borrowkwargs, _repr_attrs
from mvpa2.mappers.base import accepts_dataset_as_samples, Mapper
from mvpa2.datasets.base import Dataset
from mvpa2.datasets.miscfx import get_nsamples_per_attr, get_samples_by_attr
from mvpa2.support import copy


class ZScoreMapper(Mapper):
    """Mapper to normalize features (Z-scoring).

    Z-scoring can be done chunk-wise (with independent mean and standard
    deviation per chunk) or on the full data. It is possible to specify
    a sample attribute, unique value of which would then be used to determine
    the chunks.

    By default, Z-scoring parameters (mean and standard deviation) are
    estimated from the data (either chunk-wise or globally). However, it is
    also possible to define fixed parameters (again a global setting or
    per-chunk definitions), or to select a specific subset of samples from
    which these parameters should be estimated.

    If necessary, data is upcasted into a configurable datatype to prevent
    information loss.

    Notes
    -----

    It should be mentioned that the mapper can be used for forward-mapping
    of datasets without prior training (it will auto-train itself
    upon first use). It is, however, not possible to map plain data arrays
    without prior training. Also, for obvious reasons, it is also not possible
    to perform chunk-wise Z-scoring of plain data arrays.

    Reverse-mapping is currently not implemented.
    """
    def __init__(self, params=None, param_est=None, chunks_attr='chunks',
                 dtype='float64', **kwargs):
        """
        Parameters
        ----------
        params : None or tuple(mean, std) or dict
          Fixed Z-Scoring parameters (mean, standard deviation). If provided,
          no parameters are estimated from the data. It is possible to specify
          individual parameters for each chunk by passing a dictionary with the
          chunk ids as keys and the parameter tuples as values. If None,
          parameters will be estimated from the training data.
        param_est : None or tuple(attrname, attrvalues)
          Limits the choice of samples used for automatic parameter estimation
          to a specific subset identified by a set of a given sample attribute
          values.  The tuple should have the name of that sample
          attribute as the first element, and a sequence of attribute values
          as the second element. If None, all samples will be used for parameter
          estimation.
        chunks_attr : str or None
          If provided, it specifies the name of a samples attribute in the
          training data, unique values of which will be used to identify chunks of
          samples, and to perform individual Z-scoring within them.
        dtype : Numpy dtype, optional
          Target dtype that is used for upcasting, in case integer data is to be
          Z-scored.
        """
        Mapper.__init__(self, **kwargs)

        self.__chunks_attr = chunks_attr
        self.__params = params
        self.__param_est = param_est
        self.__params_dict = None
        self.__dtype = dtype

        # secret switch to perform in-place z-scoring
        self._secret_inplace_zscore = False


    def __repr__(self, prefixes=[]):
        return super(ZScoreMapper, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['params', 'param_est', 'chunks_attr'])
            + _repr_attrs(self, ['dtype'], default='float64'))


    def __str__(self):
        return _str(self)


    def _train(self, ds):
        # local binding
        chunks_attr = self.__chunks_attr
        params = self.__params
        param_est = self.__param_est

        # populate a dictionary with tuples of (mean,std) for all chunks, or
        # a global value that is is used for the whole data
        if not params is None:
            # we got mean and std already
            if not isinstance(params, dict):
                # turn into dict, otherwise assume that we have parameters per
                # chunk
                params = {'__all__': params}
        else:
            # no parameters given, need to estimate
            if not param_est is None:
                est_attr, est_attr_values = param_est
                # which samples to use for estimation
                est_ids = set(get_samples_by_attr(ds, est_attr,
                                                  est_attr_values))
            else:
                est_ids = slice(None)

            # now we can either do it one for all, or per chunk
            if not chunks_attr is None:
                # per chunk estimate
                params = {}
                for c in ds.sa[chunks_attr].unique:
                    slicer = np.where(ds.sa[chunks_attr].value == c)[0]
                    if not isinstance(est_ids, slice):
                        slicer = list(est_ids.intersection(set(slicer)))
                    params[c] = self._compute_params(ds.samples[slicer])
            else:
                # global estimate
                params = {'__all__': self._compute_params(ds.samples[est_ids])}


        self.__params_dict = params


    def _forward_dataset(self, ds):
        # local binding
        chunks_attr = self.__chunks_attr
        dtype = self.__dtype

        if __debug__ and not chunks_attr is None:
            nsamples_per_chunk = get_nsamples_per_attr(ds, chunks_attr)
            min_nsamples_per_chunk = np.min(nsamples_per_chunk.values())
            if min_nsamples_per_chunk in range(3, 6):
                warning("Z-scoring chunk-wise having a chunk with only "
                        "%d samples is 'discouraged'. "
                        "You have chunks with following number of samples: %s"
                        % (min_nsamples_per_chunk, nsamples_per_chunk,))
            if min_nsamples_per_chunk <= 2:
                warning("Z-scoring chunk-wise having a chunk with less "
                        "than three samples will set features in these "
                        "samples to either zero (with 1 sample in a chunk) "
                        "or -1/+1 (with 2 samples in a chunk). "
                        "You have chunks with following number of samples: %s"
                        % (nsamples_per_chunk,))

        params = self.__params_dict
        if params is None:
            raise RuntimeError, \
                  "ZScoreMapper needs to be trained before call to forward"

        if self._secret_inplace_zscore:
            mds = ds
        else:
            # shallow copy to put the new stuff in
            mds = ds.copy(deep=False)
            # but deepcopy the samples since _zscore would modify inplace
            mds.samples = mds.samples.copy()

        # cast the data to float, since in-place operations below do not upcast!
        if np.issubdtype(mds.samples.dtype, np.integer):
            mds.samples = mds.samples.astype(dtype)

        if '__all__' in params:
            # we have a global parameter set
            mds.samples = self._zscore(mds.samples, *params['__all__'])
        else:
            # per chunk z-scoring
            for c in mds.sa[chunks_attr].unique:
                if not c in params:
                    raise RuntimeError(
                        "%s has no parameters for chunk '%s'. It probably "
                        "wasn't present in the training dataset!?"
                        % (self.__class__.__name__, c))
                slicer = np.where(mds.sa[chunks_attr].value == c)[0]
                mds.samples[slicer] = self._zscore(mds.samples[slicer],
                                                   *params[c])

        return mds


    def _forward_data(self, data):
        if self.__chunks_attr is not None:
            raise RuntimeError(
                "%s cannot do chunk-wise Z-scoring of plain data "
                "since it has to be parameterized with chunks_attr." % self)
        if self.__param_est is not None:
            raise RuntimeError("%s cannot do Z-scoring with estimating "
                               "parameters on some attributes of plain"
                               "data." % self)

        params = self.__params_dict
        if params is None:
            raise RuntimeError, \
                  "ZScoreMapper needs to be trained before call to forward"

        # mappers should not modify the input data
        # cast the data to float, since in-place operations below to not upcast!
        if np.issubdtype(data.dtype, np.integer):
            if self._secret_inplace_zscore:
                raise TypeError(
                    "Cannot perform inplace z-scoring since data is of integer "
                    "type. Please convert to float before calling zscore")
            mdata = data.astype(self.__dtype)
        elif self._secret_inplace_zscore:
            mdata = data
        else:
            # do not call .copy() directly, since it might not be an array
            mdata = copy.deepcopy(data)

        self._zscore(mdata, *params['__all__'])
        return mdata


    def _compute_params(self, samples):
        return (np.mean(samples, axis=0), np.std(samples, axis=0))


    def _zscore(self, samples, mean, std):
        # de-mean
        if np.isscalar(mean) or samples.shape[1] == len(mean):
            mean = np.asanyarray(mean)  # assure array
            samples -= mean
        else:
            raise RuntimeError("mean should be a per-feature vector. Got: %r"
                               % (mean,))

        # scale
        if np.isscalar(std):
            if std == 0:
                samples[:] = 0
            else:
                samples /= std
        else:
            std = np.asanyarray(std)
            if samples.shape[1] != len(std):
                raise RuntimeError("std should be a per-feature vector.")
            else:
                # check for invariant features
                std_nz = std != 0
                samples[:, std_nz] /= np.asanyarray(std)[std_nz]
        return samples

    params = property(fget=lambda self:self.__params)
    param_est = property(fget=lambda self:self.__param_est)
    chunks_attr = property(fget=lambda self:self.__chunks_attr)
    dtype = property(fget=lambda self:self.__dtype)


@borrowkwargs(ZScoreMapper, '__init__')
def zscore(ds, **kwargs):
    """In-place Z-scoring of a `Dataset` or `ndarray`.

    This function behaves identical to `ZScoreMapper`. The only difference is
    that the actual Z-scoring is done in-place -- potentially causing a
    significant reduction of memory demands.

    Parameters
    ----------
    ds : Dataset or ndarray
      The data that will be Z-scored in-place.
    **kwargs
      For all other arguments, please see the documentation of `ZScoreMapper`.
    """
    zm = ZScoreMapper(**kwargs)
    zm._secret_inplace_zscore = True
    # train
    if isinstance(ds, Dataset):
        zm.train(ds)
    else:
        zm.train(Dataset(ds))
    # map
    mapped = zm.forward(ds)
    # and append the mapper to the dataset
    if isinstance(mapped, Dataset):
        mapped._append_mapper(zm)
