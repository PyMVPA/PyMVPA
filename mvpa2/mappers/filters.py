# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Filtering mappers."""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import externals
if externals.exists('scipy', raise_=True):
    from scipy.signal import resample
    if externals.versions['scipy'] >= '0.11':
        from scipy.signal import filtfilt
    else:
        from scipy.signal._arraytools import axis_reverse, axis_slice, odd_ext
        from scipy.signal import lfilter, lfilter_zi
        # Taken from scipy 0.11, all version before are broken see
        # https://bugs.debian.org/736185
        def filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None):
            """
            A forward-backward filter.

            This function applies a linear filter twice, once forward
            and once backwards.  The combined filter has linear phase.

            Before applying the filter, the function can pad the data along the
            given axis in one of three ways: odd, even or constant.  The odd
            and even extensions have the corresponding symmetry about the end point
            of the data.  The constant extension extends the data with the values
            at end points.  On both the forward and backwards passes, the
            initial condition of the filter is found by using `lfilter_zi` and
            scaling it by the end point of the extended data.

            Parameters
            ----------
            b : (N,) array_like
                The numerator coefficient vector of the filter.
            a : (N,) array_like
                The denominator coefficient vector of the filter.  If a[0]
                is not 1, then both a and b are normalized by a[0].
            x : array_like
                The array of data to be filtered.
            axis : int, optional
                The axis of `x` to which the filter is applied.
                Default is -1.
            padtype : str or None, optional
                Must be 'odd', 'even', 'constant', or None.  This determines the
                type of extension to use for the padded signal to which the filter
                is applied.  If `padtype` is None, no padding is used.  The default
                is 'odd'.
            padlen : int or None, optional
                The number of elements by which to extend `x` at both ends of
                `axis` before applying the filter. This value must be less than
                `x.shape[axis]-1`.  `padlen=0` implies no padding.
                The default value is 3*max(len(a),len(b)).

            Returns
            -------
            y : ndarray
                The filtered output, an array of type numpy.float64 with the same
                shape as `x`.

            See Also
            --------
            lfilter_zi, lfilter

            Examples
            --------
            First we create a one second signal that is the sum of two pure sine
            waves, with frequencies 5 Hz and 250 Hz, sampled at 2000 Hz.

            >>> t = np.linspace(0, 1.0, 2001)
            >>> xlow = np.sin(2 * np.pi * 5 * t)
            >>> xhigh = np.sin(2 * np.pi * 250 * t)
            >>> x = xlow + xhigh

            Now create a lowpass Butterworth filter with a cutoff of 0.125 times
            the Nyquist rate, or 125 Hz, and apply it to x with filtfilt.  The
            result should be approximately xlow, with no phase shift.

            >>> from scipy import signal
            >>> b, a = signal.butter(8, 0.125)
            >>> y = signal.filtfilt(b, a, x, padlen=150)
            >>> np.abs(y - xlow).max()
            9.1086182074789912e-06

            We get a fairly clean result for this artificial example because
            the odd extension is exact, and with the moderately long padding,
            the filter's transients have dissipated by the time the actual data
            is reached.  In general, transient effects at the edges are
            unavoidable.

            """

            if padtype not in ['even', 'odd', 'constant', None]:
                raise ValueError(("Unknown value '%s' given to padtype.  padtype must "
                                 "be 'even', 'odd', 'constant', or None.") %
                                    padtype)

            b = np.asarray(b)
            a = np.asarray(a)
            x = np.asarray(x)

            ntaps = max(len(a), len(b))

            if padtype is None:
                padlen = 0

            if padlen is None:
                # Original padding; preserved for backwards compatibility.
                edge = ntaps * 3
            else:
                edge = padlen

            # x's 'axis' dimension must be bigger than edge.
            if x.shape[axis] <= edge:
                raise ValueError("The length of the input vector x must be at least "
                                 "padlen, which is %d." % edge)

            if padtype is not None and edge > 0:
                # Make an extension of length `edge` at each
                # end of the input array.
                if padtype == 'even':
                    ext = even_ext(x, edge, axis=axis)
                elif padtype == 'odd':
                    ext = odd_ext(x, edge, axis=axis)
                else:
                    ext = const_ext(x, edge, axis=axis)
            else:
                ext = x

            # Get the steady state of the filter's step response.
            zi = lfilter_zi(b, a)

            # Reshape zi and create x0 so that zi*x0 broadcasts
            # to the correct value for the 'zi' keyword argument
            # to lfilter.
            zi_shape = [1] * x.ndim
            zi_shape[axis] = zi.size
            zi = np.reshape(zi, zi_shape)
            x0 = axis_slice(ext, stop=1, axis=axis)

            # Forward filter.
            (y, zf) = lfilter(b, a, ext, axis=axis, zi=zi * x0)

            # Backward filter.
            # Create y0 so zi*y0 broadcasts appropriately.
            y0 = axis_slice(y, start=-1, axis=axis)
            (y, zf) = lfilter(b, a, axis_reverse(y, axis=axis), axis=axis, zi=zi * y0)

            # Reverse y.
            y = axis_reverse(y, axis=axis)

            if edge > 0:
                # Slice the actual signal from the extended signal.
                y = axis_slice(y, start=edge, stop=-edge, axis=axis)

            return y

from mvpa2.base import warning
from mvpa2.base.param import Parameter
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
        if not self.__position_attr is None:
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

        if not pos is None:
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

    axis = Parameter(0, allowedtype='int',
            doc="""The axis of `x` to which the filter is applied. By default
            the filter is applied to all features along the samples axis""")

    padtype = Parameter('odd', allowedtype="{'odd', 'even', 'constant', None}",
            doc="""Must be 'odd', 'even', 'constant', or None.  This determines
            the type of extension to use for the padded signal to which the
            filter is applied.  If `padtype` is None, no padding is used.  The
            default is 'odd'""")

    padlen = Parameter(None, allowedtype="int or None",
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
