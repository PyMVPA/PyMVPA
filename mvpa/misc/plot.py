#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Misc. plotting helpers."""

__docformat__ = 'restructuredtext'

import pylab as P
import numpy as N


def errLinePlot(data, errtype='ste', curves=None, linestyle='--', fmt='o'):
    """Make a line plot with errorbars on the data points.

    :Parameters:
      data: sequence of sequences
        First axis separates samples and second axis will appear as
        x-axis in the plot.
      errtype: 'ste' | 'std'
        Type of error value to be computed per datapoint.
          'ste': standard error of the mean
          'std': standard deviation
      curves: None | ndarrayb
        Each *row* of the array is plotted as an additional curve. The
        curves might have a different sampling frequency (i.e. number of
        samples) than the data array, as it will be scaled (along
        x-axis) to the range of the data points.
      linestyle: str
        matplotlib linestyle argument. Applied to either the additional
        curve or a the line connecting the datapoints. Set to 'None' to
        disable the line completely.
      fmt: str
        matplotlib plot style argument to be applied to the data points
        and errorbars.


    :Example:

      Make dataset with 20 samples from a full sinus wave period,
      computed 100 times with individual noise pattern.

        >>> x = N.linspace(0, N.pi * 2, 20)
        >>> data = N.vstack([N.sin(x)] * 30)
        >>> data += N.random.normal(size=data.shape)

      Now, plot mean data points with error bars, plus a high-res
      version of the original sinus wave.

        >>> errLinePlot(data, curves=N.sin(N.linspace(0, N.pi * 2, 200)))
        >>> #P.show()
    """
    data = N.asanyarray(data)

    if len(data.shape) < 2:
        data = N.atleast_2d(data)

    # compute mean signal course
    md = data.mean(axis=0)

    # compute matching datapoint locations on x-axis
    x = N.arange(len(md))

    # plot highres line if present
    if curves is not None:
        curves = N.array(curves, ndmin=2).T

        # scales line array to same range as datapoints
        P.plot(N.linspace(0, len(md), len(curves)), curves, linestyle=linestyle)
        # no line between data points
        linestyle='None'

    # compute error per datapoint
    if errtype == 'ste':
        err = data.std(axis=0) / N.sqrt(len(data))
    elif errtype == 'std':
        err = data.std(axis=0)
    else:
        raise ValueError, "Unknown error type '%s'" % errtype

    # plot datapoints with error bars
    P.errorbar(x, md, err, fmt=fmt, linestyle=linestyle)
