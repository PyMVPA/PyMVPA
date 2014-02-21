# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Some functionality which was fixed in later versions of SciPy -- signal processing"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import externals
if externals.exists('scipy', raise_=True):
    if externals.versions['scipy'] >= '0.11':
        from scipy.signal import filtfilt
    else:
        if externals.versions['scipy'] >= '0.10':
            from scipy.signal._arraytools import axis_reverse, axis_slice, odd_ext, const_ext
        else:
            from ._arraytools import axis_reverse, axis_slice, odd_ext, const_ext

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
            >>> y = filtfilt(b, a, x, padlen=150)
            >>> print('%.5g' % np.abs(y - xlow).max())
            9.1086e-06

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
