# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Spectral filtering and FFT-based resampling."""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import externals

if externals.exists('scipy', raise_=True):
    from scipy.signal import resample
    from mvpa2.support.scipy.signal import filtfilt

from mvpa2.base import warning
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import EnsureChoice, EnsureInt, EnsureNone
from mvpa2.base.dochelpers import _str, borrowkwargs
from mvpa2.mappers.base import Mapper
from mvpa2.datasets import Dataset
from mvpa2.base.dataset import vstack
from mvpa2.generators.splitters import Splitter

class FFTResampleMapper(Mapper):
    """Mapper for FFT-based resampling.

    Can do per-chunk.

    Supports positional information of samples and outputs them as sample
    attribute. however, only meaningful for data with equally spaced sampling
    points.

    Pretty much Mapper frontend for scipy.signal.resample

    """
    def __init__(self, num, window=None, chunks_attr=None, position_attr=None,
                 attr_strategy='remove', **kwargs):
        """
        Parameters
        ----------
        num : int
          Number of output samples. If operating on chunks, this is the number
          of samples per chunk.
        window : str or float or tuple
          Passed to scipy.signal.resample
        chunks_attr : str or None
          If not None, this samples attribute defines chunks that will be
          resampled individually.
        position_attr : str
          A samples attribute with positional information that is passed
          to scipy.signal.resample. If not None, the output dataset will
          also contain a sample attribute of this name, with updated
          positional information (this is, however, only meaningful for
          equally spaced samples).
        attr_strategy : {'remove', 'sample', 'resample'}
          Strategy to process sample attributes during mapping. 'remove' will
          cause all sample attributes to be removed. 'sample' will pick orginal
          attribute values matching the new resampling frequency (e.g. every
          10th), and 'resample' will also apply the actual data resampling
          procedure to the attributes as well (which might not be possible, e.g.
          for literal attributes).
        """
        Mapper.__init__(self, **kwargs)

        self.__num = num
        self.__window_args = window
        self.__chunks_attr = chunks_attr
        self.__position_attr = position_attr
        self.__attr_strategy = attr_strategy


    def __repr__(self):
        s = super(FFTResamplemapper, self).__repr__()
        return s.replace("(",
                         "(chunks_attr=%s, "
                          % (repr(self.__chunks_attr),),
                         1)


    def __str__(self):
        return _str(self, chunks_attr=self.__chunks_attr)


    def _forward_data(self, data):
        # we cannot have position information without a dataset
        return resample(data, self.__num, t=None, window=self.__window_args)


    def _forward_dataset(self, ds):
        if self.__chunks_attr is None:
            return self._forward_dataset_helper(ds)
        else:
            # strip down dataset to speedup local processing
            if self.__attr_strategy == 'remove':
                keep_sa = []
            else:
                keep_sa = None
            proc_ds = ds.copy(deep=False, sa=keep_sa, fa=[], a=[])
            # process all chunks individually
            # use a customsplitter to speed-up splitting
            spl = Splitter(self.__chunks_attr)
            dses = [self._forward_dataset_helper(d)
                        for d in spl.generate(proc_ds)]
            # and merge them again
            mds = vstack(dses)
            # put back attributes
            mds.fa.update(ds.fa)
            mds.a.update(ds.a)
            return mds


    def _forward_dataset_helper(self, ds):
        # local binding
        num = self.__num

        pos = None
        if self.__position_attr is not None:
            # we know something about sample position
            pos = ds.sa[self.__position_attr].value
            rsamples, pos = resample(ds.samples, self.__num, t=pos,
                                     window=self.__window_args)
        else:
            # we know nothing about samples position
            rsamples = resample(ds.samples, self.__num, t=None,
                                window=self.__window_args)
        # new dataset that reuses that feature and dataset attributes of the
        # source
        mds = Dataset(rsamples, fa=ds.fa, a=ds.a)

        # the tricky part is what to do with the samples attributes, since their
        # number has changes
        if self.__attr_strategy == 'remove':
            # nothing to be done
            pass
        elif self.__attr_strategy == 'sample':
            step = int(len(ds) / num)
            sa = dict([(k, ds.sa[k].value[0::step][:num]) for k in ds.sa])
            mds.sa.update(sa)
        elif self.__attr_strategy == 'resample':
            # resample the attributes themselves
            sa = {}
            for k in ds.sa:
                v = ds.sa[k].value
                if pos is None:
                    sa[k] = resample(v, self.__num, t=None,
                                     window=self.__window_args)
                else:
                    if k == self.__position_attr:
                        # position attr will be handled separately at the end
                        continue
                    sa[k] = resample(v, self.__num, t=pos,
                                     window=self.__window_args)[0]
            # inject them all
            mds.sa.update(sa)
        else:
            raise ValueError("Unkown attribute handling strategy '%s'."
                             % self.__attr_strategy)

        if pos is not None:
            # we got the new sample positions and can store them
            mds.sa[self.__position_attr] = pos
        return mds


@borrowkwargs(FFTResampleMapper, '__init__')
def fft_resample(ds, num, **kwargs):
    """FFT-based resampling.

    Parameters
    ----------
    ds : Dataset
    **kwargs
      For all other arguments, please see the documentation of
      FFTResampleMapper.
    """
    dm = FFTResampleMapper(num, **kwargs)
    return dm.forward(ds)


class IIRFilterMapper(Mapper):
    """Mapper using IIR filters for data transformation.

    This mapper is able to perform any IIR-based low-pass, high-pass, or
    band-pass frequency filtering. This is a front-end for SciPy's filtfilt(),
    hence its usage looks almost exactly identical, and any of SciPy's IIR
    filters can be used with this mapper:

    >>> from scipy import signal
    >>> b, a = signal.butter(8, 0.125)
    >>> mapper = IIRFilterMapper(b, a, padlen=150)

    """

    axis = Parameter(0, constraints='int',
            doc="""The axis of `x` to which the filter is applied. By default
            the filter is applied to all features along the samples axis""")

    padtype = Parameter('odd', 
            constraints=EnsureChoice('odd', 'even', 'constant') | EnsureNone(),
            doc="""Must be 'odd', 'even', 'constant', or None.  This determines
            the type of extension to use for the padded signal to which the
            filter is applied.  If `padtype` is None, no padding is used.  The
            default is 'odd'""")

    padlen = Parameter(None, constraints=EnsureInt() | EnsureNone(),
            doc="""The number of elements by which to extend `x` at both ends
            of `axis` before applying the filter. This value must be less than
            `x.shape[axis]-1`.  `padlen=0` implies no padding. The default
            value is 3*max(len(a),len(b))""")

    def __init__(self, b, a, **kwargs):
        """
        All constructor parameters are analogs of filtfilt() or are passed
        on to the Mapper base class.

        Parameters
        ----------
        b : (N,) array_like
            The numerator coefficient vector of the filter.
        a : (N,) array_like
            The denominator coefficient vector of the filter.  If a[0]
            is not 1, then both a and b are normalized by a[0].
        """
        Mapper.__init__(self, auto_train=True, **kwargs)
        self.__iir_num = b
        self.__iir_denom = a

    def _forward_data(self, data):
        params = self.params
        try:
            mapped = filtfilt(self.__iir_num,
                              self.__iir_denom,
                              data,
                              axis=params.axis,
                              padtype=params.padtype,
                              padlen=params.padlen)
        except TypeError:
            # we have an ancient scipy, do manually
            # but is will only support 2d arrays
            if params.axis == 0:
                data = data.T
            if params.axis > 1:
                raise ValueError("this version of scipy does not "
                                 "support nd-arrays for filtfilt()")
            if not (params['padlen'].is_default and params['padtype'].is_default):
                warning("this version of scipy.signal.filtfilt() does not "
                        "support `padlen` and `padtype` arguments -- ignoring "
                        "them")
            mapped = [filtfilt(self.__iir_num,
                               self.__iir_denom,
                               x)
                    for x in data]
            mapped = np.array(mapped)
            if params.axis == 0:
                mapped = mapped.T
        return mapped


@borrowkwargs(IIRFilterMapper, '__init__')
def iir_filter(ds, *args, **kwargs):
    """IIR-based frequency filtering.

    Parameters
    ----------
    ds : Dataset
    **kwargs
      For all other arguments, please see the documentation of
      IIRFilterMapper.
    """
    dm = IIRFilterMapper(*args, **kwargs)
    return dm.forward(ds)
