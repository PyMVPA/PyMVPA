# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Visualization of datasets"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import externals

from mvpa2.base.node import ChainNode
from mvpa2.base.dataset import is_datasetlike
from mvpa2.generators.splitters import Splitter
from mvpa2.generators.partition import NFoldPartitioner

def hist(dataset, xgroup_attr=None, ygroup_attr=None,
         xlim=None, ylim=None,
         noticks=False,
         **kwargs):
    """Compute and draw feature histograms (for groups of samples)

    This is a convenience wrapper around matplotlib's hist() function.  It
    supports it entire API, but data is taken from an input dataset.  In
    addition, feature histograms for groups of dataset samples can be drawn as
    an array of subplots. Using ``xgroup_attr`` and ``ygroup_attr`` up to two
    sample attributes can be selected and samples groups are defined by their
    unique values. For example, plotting histograms for all combinations of
    ``targets`` and ``chunks`` attribute values in a dataset is done by this
    code:

    >>> from mvpa2.viz import hist
    >>> from mvpa2.misc.data_generators import normal_feature_dataset
    >>> ds = normal_feature_dataset(10, 3, 10, 5)
    >>> plots = hist(ds, ygroup_attr='targets', xgroup_attr='chunks',
    ...              noticks=None, xlim=(-.5,.5), normed=True)
    >>> len(plots)
    15

    This function can also be used with plain arrays, in which case it will
    fall back on the behavior of matplotlib's hist() and additional
    functionality is not available.

    Parameters
    ----------
    dataset : Dataset or array
    xgroup_attr : string, optional
      Name of a samples attribute to be used as targets
    ygroup_attr : None or string, optional
      If a string, a histogram will be plotted per each target and each
      chunk (as defined in sa named `chunks_attr`), resulting is a
      histogram grid (targets x chunks).
    xlim : None or 2-tuple, optional
      Common x-axis limits for all histograms.
    ylim : None or 2-tuple, optional
      Common y-axis limits for all histograms.
    noticks : bool or None, optional
      If True, no axis ticks will be plotted. If False, each histogram subplot
      will have its own ticks. If None, only the outer subplots will
      have ticks. This is useful to save space in large plots, but should be
      combined with ``xlim`` and ``ylim`` arguments in order to ensure equal
      axes across subplots.
    **kwargs
      Any additional arguments are passed to matplotlib's hist().

    Returns
    -------
    list
      List of figure handlers for all generated subplots.
    """
    externals.exists("pylab", raise_=True)
    import pylab as pl

    xgroup = {'attr': xgroup_attr}
    ygroup = {'attr': ygroup_attr}
    for grp in (xgroup, ygroup):
        if grp['attr'] is not None and is_datasetlike(dataset):
            grp['split'] = ChainNode([NFoldPartitioner(1, attr=grp['attr']),
                                      Splitter('partitions', attr_values=[2])])
            grp['gen'] = lambda s, x: s.generate(x)
            grp['npanels'] = len(dataset.sa[grp['attr']].unique)
        else:
            grp['split'] = None
            grp['gen'] = lambda s, x: [x]
            grp['npanels'] = 1

    fig = 1
    nrows = ygroup['npanels']
    ncols = xgroup['npanels']
    subplots = []
    # for all labels
    for row, ds in enumerate(ygroup['gen'](ygroup['split'], dataset)):
        for col, d in enumerate(xgroup['gen'](xgroup['split'], ds)):
            ax = pl.subplot(nrows, ncols, fig)
            if is_datasetlike(d):
                data = d.samples
            else:
                data = d
            ax.hist(data.ravel(), **kwargs)
            if xlim is not None:
                pl.xlim(xlim)

            if noticks is True or (noticks is None and row < nrows - 1):
                pl.xticks([])
            if noticks is True or (noticks is None and col > 0):
                pl.yticks([])

            if ncols > 1 and row == 0:
                pl.title(str(d.sa[xgroup['attr']].unique[0]))
            if nrows > 1 and col == 0:
                pl.ylabel(str(d.sa[ygroup['attr']].unique[0]))
            fig += 1
            subplots.append(ax)
    return subplots


def matshow(matrix, xlabel_attr=None, ylabel_attr=None, numbers=None,
            **kwargs):
    """Enhanced version of matplotlib's matshow().

    This version is able to handle datasets, and label axis according to
    dataset attribute values.

    >>> from mvpa2.viz import matshow
    >>> from mvpa2.misc.data_generators import normal_feature_dataset
    >>> ds = normal_feature_dataset(10, 2, 18, 5)
    >>> im = matshow(ds, ylabel_attr='targets', xlabel_attr='chunks',
    ...               numbers='%.0f')

    Parameters
    ----------
    matrix : 2D array
      The matrix that is to be plotted as an image. If 'matrix' is of
      type Dataset the function tries to plot the corresponding samples.
    xlabel_attr : str or None
      If not 'None' matrix is treated as a Dataset and labels are
      extracted from the sample attribute named 'xlabel_attr'.
      The labels are used as the 'x_tick_lables' of the image.
    ylabel_attr : str or None
      If not 'None' matrix is treated as a Dataset and labels are
      extracted from the feature attribute named 'ylabel_attr'.
      The labels are used as the 'y_tick_lables' of the image.
    numbers : dict, str or None
      If not 'None' plots matrix values as text inside the image.
      If a string is provided, then this string is used as format string.
      In case that a dictionary is provided, the dictionary gets passed
      on to the text command, and, '%d' is used to format the values.
    **kwargs
      Additional parameters passed on to matshow().

    Returns
    -------
    matplotlib.AxesImage
      Handler for the created image.
    """

    externals.exists("pylab", raise_=True)
    import pylab as pl

    if numbers is not None:
        if isinstance(numbers, str):
            numbers_format = numbers
            numbers_alpha = None
            numbers_kwargs_ = {}
        elif isinstance(numbers, dict):
            numbers_format = '%d'
            numbers_alpha = numbers.pop('numbers_alpha', None)
            numbers_kwargs_ = numbers
        else:
            raise TypeError("The argument to keyword 'numbers' must be "
                            "either of type string or type dict")

    _xlabel = None
    _ylabel = None

    # check if dataset 'is' a confusion matrix
    if is_datasetlike(matrix):
        if xlabel_attr is not None and ylabel_attr is not None:
            _xlabel = matrix.get_attr(xlabel_attr)[0].value  # LookupError
            _ylabel = matrix.get_attr(ylabel_attr)[0].value  # if it's not there

    matrix = np.asanyarray(matrix)

    fig = pl.gcf()
    ax = pl.gca()
    im = ax.matshow(matrix, **kwargs)

    # customize labels if _xlabel  and _ylabel are set
    if _xlabel is not None and _ylabel is not None:
        xlabels = [item.get_text() for item in ax.get_xticklabels()]
        xlabels[1:-1] = _xlabel
        ax.set_xticklabels(xlabels)
        pl.xlabel(xlabel_attr)
        ylabels = [item.get_text() for item in ax.get_yticklabels()]
        ylabels[1:-1] = _ylabel
        ax.set_yticklabels(ylabels)
        pl.ylabel(ylabel_attr)

    # colorbar customization for discrete matrix
    # code taken from old ConfusionMatrix.plot()
    # TODO: colorbar should be discrete as well
    cb_kwargs_ = {}
    maxv = np.max(matrix)
    if ('int' in matrix.dtype.name) and (maxv > 0):
        boundaries = np.linspace(0, maxv, np.min((maxv, 10)), True)
        cb_kwargs_['format'] = '%d'
        cb_kwargs_['ticks'] = boundaries

    cb = pl.colorbar(im, **cb_kwargs_)

    # plot matrix values inside the image if number is set
    if numbers is not None:
        colors = [im.to_rgba(0), im.to_rgba(maxv)]
        for i, cas in enumerate(matrix):
            for j, v in enumerate(cas):
                numbers_kwargs_['color'] = colors[int(v<maxv/2)]
                # code taken from old ConfusionMatrix.plot()
                if numbers_alpha is None:
                    alpha = 1.0
                else:
                    # scale non-linearly w.r.t. value
                    alpha = 1 - np.array(1 - np.float(v)/maxv) ** numbers_alpha
                pl.text(j, i, numbers_format % v,
                        alpha=alpha, **numbers_kwargs_)

    return im
