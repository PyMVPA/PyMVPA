# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


__docformat__ = 'restructuredtext'


import numpy as np

from mvpa2.base import externals, types


def imshow(matrix, xlabel_attr=None, ylabel_attr=None, numbers=None,
           **kwargs):
    """Plot a matrix by calling matshow() from matplotlib.

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
    if types.is_datasetlike(matrix):
        if xlabel_attr is not None and ylabel_attr is not None:
            _xlabel = matrix.get_attr(xlabel_attr)[0].value  # LookupError
            _ylabel = matrix.get_attr(ylabel_attr)[0].value  # if it's not there
            if not np.array_equal(_xlabel, _ylabel):
                raise ValueError, "Elements in %s and %s " \
                                  "do not match" % (xlabel_attr, ylabel_attr)

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
