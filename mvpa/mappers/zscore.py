# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Mapper for data normalization by Z-Scoring."""

__docformat__ = 'restructuredtext'

from sets import Set
import numpy as N

from mvpa.base.dochelpers import _str, borrowkwargs
from mvpa.mappers.base import accepts_dataset_as_samples, Mapper
from mvpa.datasets.miscfx import get_nsamples_per_attr, get_samples_by_attr


class ZScoreMapper(Mapper):
    """Mapper to normalize features (Z-scoring).

    Z-scoring can be done chunk-wise (with independent mean and standard
    deviation per chunk) or on the full data. It is possible to specify
    an sample attribute, whos unique value are used to determine the chunks.

    By default, the Z-scoring parameters mean and standard deviation are
    estimated from the data (either chunk-wise or globally). However, it is
    also possible to define fixed parameters (again a global setting or
    per-chunk definitions), or select a specific subset of samples from which
    these parameters should be estimated.

    If necessary, data is upcasted into a configurable datatype to prevent
    information loss.

    Notes
    -----

    It should be mentioned that the mapper can be used for
    forward-mapping of datasets without prior train (it will auto-train itself
    upon first use). It is, however, not possible to map plain data arrays
    without prior training. Also, for obvious reasons, it is also not possible
    to perform chunk-wise Z-scoring of plain data arrays.

    Reverse-mapping is currently not implemented.
    """
    def __init__(self, params=None, param_est=None, chunks='chunks',
                 dtype='float64', inspace=None):
        """
        Parameters
        ----------
        params : tuple(mean, std) or dict or None
          Fixed Z-Scoring parameters (mean, standard deviation). If provided
          no parameters are estimated from the data. It is possible to specify
          individual parameters for each chunk by passing a dictionary with the
          chunk ids as keys and the parameter tuples as values. If None,
          parameters will be estimated from the training data.
        param_est : tuple(attrname, attrvalues) or None
          Limited the samples used for automatic parameter estimation to a
          specific subset identified by a set of value of a given sample
          attribute. The to be passed tuple has the name of that sample
          attribute as the first element, and a sequence of atrribute values
          as the second element. If None, all samples will be used for parameter
          estimation
        chunks : str or None
          If provided, it specifies the name of a samples attribute in the
          training data, whos unique values will be sued to identify chunks of
          samples, and perform individual Z-scoring on them.
        dtype : Numpy dtype
          Target dtype that is used for upcasting, in case integer data is
          Z-scored.
        inspace : None
          Currently, this argument has no effect.
        """
        Mapper.__init__(self, inspace=inspace)

        self.__chunks = chunks
        self.__params= params
        self.__param_est = param_est
        self.__params_dict = None
        self.__dtype = dtype

        # secret switch to perform in-place detrending
        self._secret_inplace_detrend = False


    def _train(self, ds):
        # local binding
        chunks = self.__chunks
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
                est_ids = Set(get_samples_by_attr(ds, est_attr,
                                                  est_attr_values))
            else:
                est_ids = slice(None)

            # now we can either do it one for all, or per chunk
            if not chunks is None:
                # per chunk estimate
                params = {}
                for c in ds.sa[chunks].unique:
                    slicer = N.where(ds.sa[chunks].value == c)[0]
                    if not isinstance(est_ids, slice):
                        slicer = list(est_ids.intersection(Set(slicer)))
                    params[c] = self._compute_params(ds.samples[slicer])
            else:
                # global estimate
                params = {'__all__': self._compute_params(ds.samples[est_ids])}


        self.__params_dict = params


    def _forward_dataset(self, ds):
        # local binding
        chunks = self.__chunks
        dtype = self.__dtype

        if __debug__ and not chunks is None \
          and N.array(get_nsamples_per_attr(ds, chunks).values()).min() <= 2:
            warning("Z-scoring chunk-wise and one chunk with less than three "
                    "samples will set features in these samples to either zero "
                    "(with 1 sample in a chunk) "
                    "or -1/+1 (with 2 samples in a chunk).")

        # auto-train the mapper if not yet done
        if self.__params_dict is None:
            self.train(ds)
        params = self.__params_dict

        if self._secret_inplace_detrend:
            mds = ds
        else:
            # shallow copy to put the new stuff in
            mds = ds.copy(deep=False)

        # cast the data to float, since in-place operations below to not upcast!
        if N.issubdtype(mds.samples.dtype, N.integer):
            mds.samples = mds.samples.astype(dtype)

        if '__all__' in params:
            # we have a global parameter set
            mds.samples = self._zscore(mds.samples, *params['__all__'])
        else:
            # per chunk z-scoring
            for c in mds.sa[chunks].unique:
                if not c in params:
                    raise RuntimeError(
                        "%s has no parameters for chunk '%s'. It probably "
                        "wasn't present in the training dataset!?"
                        % (self.__class__.__name__, c))
                slicer = N.where(mds.sa[chunks].value == c)[0]
                mds.samples[slicer] = self._zscore(mds.samples[slicer],
                                                   *params[c])

        return mds


    def _forward_data(self, data):
        if not self.__chunks is None:
            raise RuntimeError("%s cannot do chunk-wise Z-scoring of plain "
                               "data." % self.__class__.__name__)

        params = self.__params_dict

        if params is None:
            raise RuntimeError("%s needs to be trained before use."
                               % self.__class__.__name__)

        # mappers should not modify the input data
        # cast the data to float, since in-place operations below to not upcast!
        if N.issubdtype(data.dtype, N.integer):
            mdata = data.astype(self.__dtype)
        else:
            mdata = data.copy()

        self._zscore(mdata, *params['__all__'])
        return mdata


    def _compute_params(self, samples):
        return (samples.mean(axis=0), samples.std(axis=0))


    def _zscore(self, samples, mean, std):
        # de-mean
        if N.isscalar(mean) or samples.shape[1] == len(mean):
            samples -= mean
        else:
            raise RuntimeError("mean should be a per-feature vector. Got: %s")

        # scale
        if N.isscalar(std):
            if std == 0:
                samples[:] = 0
            else:
                samples /= std
        else:
            if samples.shape[1] != len(std):
                raise RuntimeError("std should be a per-feature vector.")
            else:
                # check for invariant features
                samples[:, std != 0] /= std[std != 0]
        return samples



@borrowkwargs(ZScoreMapper, '__init__')
def zscore(ds, **kwargs):
    """In-place Z-scoring of a `Dataset`.

    This function behaves identical to `ZScoreMapper`. The only difference is
    that the actual Z-scoring is done in-place -- potentially causing a
    significant reduction of the memory demands.

    Parameters
    ----------
    ds : Dataset
      The dataset that will be Z-scored in-place.
    **kwargs
      For all other arguments, please see the documentation of `ZScoreMapper`.
    """
    zm = ZScoreMapper(**kwargs)
    zm._secret_inplace_detrend = True
    # map
    mapped = zm(ds)
    # and append the mapper to the dataset
    mapped._append_mapper(zm)
    return mapped
