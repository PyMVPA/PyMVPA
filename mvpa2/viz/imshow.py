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

from mvpa2.datasets import Dataset
from mvpa2.base import externals



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
    
    fig = pl.gcf()
    ax = pl.gca()
    im = ax.matshow(matrix)               
    cb = pl.colorbar(im) 

    # plot matrix values inside the image   
    if numbers is not None:
        numbers_kwargs_ = {'fontsize': 14,
                           'horizontalalignment': 'center',
                           'verticalalignment': 'center'}
        maxv = float(np.max(matrix))
        colors = [im.to_rgba(0), im.to_rgba(maxv)]
        for i, cas in enumerate(matrix):
            for j, v in enumerate(cas):            
                numbers_kwargs_['color'] = colors[int(v<maxv/2)]
                pl.text(j, i, numbers % v, alpha=2.0, **numbers_kwargs_)
                
    pl.draw()
    
    return fig, im, cb