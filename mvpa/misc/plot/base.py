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

from mvpa.datasets.splitter import NFoldSplitter
from mvpa.clfs.distance import squared_euclidean_distance



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
        xaxis = N.linspace(0, len(md), len(curves))

        # Since older matplotlib versions cannot plot multiple plots
        # for the same axis, lets plot each column separately
        for c in xrange(curves.shape[1]):
            # scales line array to same range as datapoints
            P.plot(xaxis, curves[:, c], linestyle=linestyle)
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


def plotFeatureHist(dataset, xlim=None, noticks=True, perchunk=False,
                    **kwargs):
    """Plot histograms of feature values for each labels.

    :Parameters:
      dataset: Dataset
      xlim: None | 2-tuple
        Common x-axis limits for all histograms.
      noticks: boolean
        If True, no axis ticks will be plotted. This is useful to save
        space in large plots.
      perchunk: boolean
        If True, one histogramm will be plotted per each label and each
        chunk, resulting is a histogram grid (labels x chunks).
      **kwargs:
        Any additional arguments are passed to matplotlib's hist().
    """
    lsplit = NFoldSplitter(1, attr='labels')
    csplit = NFoldSplitter(1, attr='chunks')

    nrows = len(dataset.uniquelabels)
    ncols = len(dataset.uniquechunks)

    def doplot(data):
        P.hist(data, **kwargs)

        if xlim is not None:
            P.xlim(xlim)

        if noticks:
            P.yticks([])
            P.xticks([])

    fig = 1

    # for all labels
    for row, (ignore, ds) in enumerate(lsplit(dataset)):
        if perchunk:
            for col, (alsoignore, d) in enumerate(csplit(ds)):

                P.subplot(nrows, ncols, fig)
                doplot(d.samples)

                if row == 0:
                    P.title('C:' + str(d.uniquechunks[0]))
                if col == 0:
                    P.ylabel('L:' + str(d.uniquelabels[0]))

                fig += 1
        else:
            P.subplot(1, nrows, fig)
            doplot(ds.samples)

            P.title('L:' + str(ds.uniquelabels[0]))

            fig += 1


def plotSamplesDistance(dataset, sortbyattr=None):
    """Plot the euclidean distances between all samples of a dataset.

    :Parameters:
      dataset: Dataset
        Providing the samples.
      sortbyattr: None | str
        If None, the samples distances will be in the same order as their
        appearance in the dataset. Alternatively, the name of a samples
        attribute can be given, which wil then be used to sort/group the
        samples, e.g. to investigate the similarity samples by label or by
        chunks.
    """
    if sortbyattr is not None:
        slicer = []
        for attr in dataset.__getattribute__('unique' + sortbyattr):
            slicer += \
                dataset.__getattribute__('idsby' + sortbyattr)(attr).tolist()
        samples = dataset.samples[slicer]
    else:
        samples = dataset.samples

    ed = N.sqrt(squared_euclidean_distance(samples))

    P.imshow(ed)
    P.colorbar()
