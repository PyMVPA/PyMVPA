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

import pylab as pl
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
