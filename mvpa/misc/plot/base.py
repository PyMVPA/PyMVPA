# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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

from mvpa.datasets.splitters import NFoldSplitter
from mvpa.clfs.distance import squared_euclidean_distance



def plotErrLine(data, x=None, errtype='ste', curves=None, linestyle='--',
                fmt='o', perc_sigchg=False, baseline=None):
    """Make a line plot with errorbars on the data points.

    :Parameters:
      data: sequence of sequences
        First axis separates samples and second axis will appear as
        x-axis in the plot.
      x: sequence
        Value to be used as 'x-values' corresponding to the elements of
        the 2nd axis id `data`. If `None`, a sequence of ascending integers
        will be generated.
      errtype: 'ste' | 'std'
        Type of error value to be computed per datapoint.
          'ste': standard error of the mean
          'std': standard deviation
      curves: None | list of tuple(x, y)
        Each tuple represents an additional curve, with x and y coordinates of
        each point on the curve.
      linestyle: str
        matplotlib linestyle argument. Applied to either the additional
        curve or a the line connecting the datapoints. Set to 'None' to
        disable the line completely.
      fmt: str
        matplotlib plot style argument to be applied to the data points
        and errorbars.
      perc_sigchg: bool
        If `True` the plot will show percent signal changes relative to a
        baseline.
      baseline: float | None
        Baseline used for converting values into percent signal changes.
        If `None` and `perc_sigchg` is `True`, the absolute of the mean of the
        first feature (i.e. [:,0]) will be used as a baseline.


    :Example:

      Make dataset with 20 samples from a full sinus wave period,
      computed 100 times with individual noise pattern.

        >>> x = N.linspace(0, N.pi * 2, 20)
        >>> data = N.vstack([N.sin(x)] * 30)
        >>> data += N.random.normal(size=data.shape)

      Now, plot mean data points with error bars, plus a high-res
      version of the original sinus wave.

        >>> x = N.linspace(0, N.pi * 2, 200)
        >>> plotErrLine(data, curves=[(x, N.sin(x))])
        >>> #P.show()
    """
    data = N.asanyarray(data)

    if len(data.shape) < 2:
        data = N.atleast_2d(data)

    # compute mean signal course
    md = data.mean(axis=0)

    if baseline is None:
        baseline = N.abs(md[0])

    if perc_sigchg:
        md /= baseline
        md -= 1.0
        md *= 100.0
        # not in-place to keep original data intact
        data = data / baseline
        data *= 100.0

    # compute matching datapoint locations on x-axis
    if x is None:
        x = N.arange(len(md))
    else:
        if not len(md) == len(x):
            raise ValueError, "The length of `x` (%i) has to match the 2nd " \
                              "axis of the data array (%i)" % (len(x), len(md))

    # plot highres line if present
    if curves is not None:
        for c in curves:
            xc, yc = c
            # scales line array to same range as datapoints
            P.plot(xc, yc, linestyle=linestyle)

        # no line between data points
        linestyle = 'None'

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
                doplot(d.samples.ravel())

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


def plotBars(data, labels=None, title=None, ylim=None, ylabel=None,
               width=0.2, offset=0.2, color='0.6', distance=1.0,
               yerr='ste', **kwargs):
    """Make bar plots with automatically computed error bars.

    Candlestick plot (multiple interleaved barplots) can be done,
    by calling this function multiple time with appropriatly modified
    `offset` argument.

    :Parameters:
      data: array (nbars x nobservations) | other sequence type
        Source data for the barplot. Error measure is computed along the
        second axis.
      labels: list | None
        If not None, a label from this list is placed on each bar.
      title: str
        An optional title of the barplot.
      ylim: 2-tuple
        Y-axis range.
      ylabel: str
        An optional label for the y-axis.
      width: float
        Width of a bar. The value should be in a reasonable relation to
        `distance`.
      offset: float
        Constant offset of all bar along the x-axis. Can be used to create
        candlestick plots.
      color: matplotlib color spec
        Color of the bars.
      distance: float
        Distance of two adjacent bars.
      yerr: 'ste' | 'std' | None
        Type of error for the errorbars. If `None` no errorbars are plotted.
      **kwargs:
        Any additional arguments are passed to matplotlib's `bar()` function.
    """
    # determine location of bars
    xlocations = (N.arange(len(data)) * distance) + offset

    if yerr == 'ste':
        yerr = [N.std(d) / N.sqrt(len(d)) for d in data]
    elif yerr == 'std':
        yerr = [N.std(d) for d in data]
    else:
        # if something that we do not know just pass on
        pass

    # plot bars
    plot = P.bar(xlocations,
                 [N.mean(d) for d in data],
                 yerr=yerr,
                 width=width,
                 color=color,
                 ecolor='black',
                 **kwargs)

    if ylim:
        P.ylim(*(ylim))
    if title:
        P.title(title)

    if labels:
        P.xticks(xlocations + width / 2, labels)

    if ylabel:
        P.ylabel(ylabel)

    # leave some space after last bar
    P.xlim(0, xlocations[-1] + width + offset)

    return plot


def inverseCmap(cmap_name):
    """Create a new colormap from the named colormap, where it got reversed

    """
    import matplotlib._cm as _cm
    import matplotlib as mpl
    try:
        cmap_data = eval('_cm._%s_data' % cmap_name)
    except:
        raise ValueError, "Cannot obtain data for the colormap %s" % cmap_name
    new_data = dict( [(k, [(v[i][0], v[-(i+1)][1], v[-(i+1)][2])
                           for i in xrange(len(v))])
                      for k,v in cmap_data.iteritems()] )
    return mpl.colors.LinearSegmentedColormap('%s_rev' % cmap_name,
                                              new_data, _cm.LUTSIZE)


def plotDatasetChunks(ds, clf_labels=None):
    """Quick plot to see chunk sctructure in dataset with 2 features

    if clf_labels is provided for the predicted labels, then
    incorrectly labeled samples will have 'x' in them
    """
    if ds.nfeatures != 2:
        raise ValueError, "Can plot only in 2D, ie for datasets with 2 features"
    if P.matplotlib.get_backend() == 'TkAgg':
        P.ioff()
    if clf_labels is not None and len(clf_labels) != ds.nsamples:
        clf_labels = None
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
    labels = ds.uniquelabels
    labels_map = dict(zip(labels, colors[:len(labels)]))
    for chunk in ds.uniquechunks:
        chunk_text = str(chunk)
        ids = ds.where(chunks=chunk)
        ds_chunk = ds[ids]
        for i in xrange(ds_chunk.nsamples):
            s = ds_chunk.samples[i]
            l = ds_chunk.labels[i]
            format = ''
            if clf_labels != None:
                if clf_labels[i] != ds_chunk.labels[i]:
                    P.plot([s[0]], [s[1]], 'x' + labels_map[l])
            P.text(s[0], s[1], chunk_text, color=labels_map[l],
                   horizontalalignment='center',
                   verticalalignment='center',
                   )
    dss = ds.samples
    P.axis((1.1 * N.min(dss[:, 0]),
            1.1 * N.max(dss[:, 1]),
            1.1 * N.max(dss[:, 0]),
            1.1 * N.min(dss[:, 1])))
    P.draw()
    P.ion()

