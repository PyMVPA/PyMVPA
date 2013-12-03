# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Multi-purpose dataset container with support for attributes."""

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
    numbers : str or None
      If not 'None' plots matrix values as text inside the image using
      'numbers' as format string.   
    **kwargs
      Additional parameters passed on to matshow().

    Returns
    -------
    fig, im, cb
      Handlers to the created figure, image and colorbar, respectively.
    """
    
    externals.exists("pylab", raise_=True)
    import pylab as pl

    _xlabel = None
    _ylabel = None

    # check if dataset 'is' a confusion matrix 
    if types.is_datasetlike(matrix):
        if xlabel_attr is not None and ylabel_attr is not None:           
            _xlabel = matrix.get_attr(xlabel_attr)[0].value  # LookupError
            _ylabel = matrix.get_attr(ylabel_attr)[0].value  # if it's not there
            if not np.equal(_xlabel, _ylabel):
                raise ValueError, "Elements in %s and $s " \
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
        numbers_kwargs_ = {'fontsize': 14,
                           'horizontalalignment': 'center',
                           'verticalalignment': 'center'}
        colors = [im.to_rgba(0), im.to_rgba(maxv)]
        for i, cas in enumerate(matrix):
            for j, v in enumerate(cas):            
                numbers_kwargs_['color'] = colors[int(v<maxv/2)]
                pl.text(j, i, numbers % v, alpha=2.0, **numbers_kwargs_)
    
    pl.draw()
    
    return fig, im, cb