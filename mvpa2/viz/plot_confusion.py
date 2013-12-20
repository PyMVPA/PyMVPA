# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

__docformat__ = 'restructuredtext'


from imshow import matshow
import pylab as pl
import numpy as np
from mvpa2.datasets import Dataset

def plot_confusion(ds, labels=None, numbers=False, 
                   numbers_alpha=None, xlabels_vertical=True):
    """Plot a confusiom matrix by calling viz.matshow().
    
    Parameters                                                                                                                  
    ---------- 
    ds : confusion matrix (Dataset)
      The matrix that is to be plotted as an image. The Dataset must 
      have a sample attribute 'predictions' and a feature attribute
      'targets'. This is the way transerror.Confusion() constructs
      a confusion matrix. 
    labels : str or None
      If not 'None' uses labels to reorder and/or subsetting the
      matrix for plotting.             
    xlabels_vertical : bool
      If set to 'True' (default) xlabels are plot vertically.
    """  
    
    if labels is not None:
        # construct a permutation vector from the argument 'labels'
        # ValueError is raised if 'labels' is not a proper subset
        full_labels = list(ds.sa['predictions'].value)   
        p = [full_labels.index(item) for item in labels]
        
        # construct a new matrix that yields the result of permutation
        # not the most efficient way (in terms of complexity) to do it
        # but perhaps the one with the least amount of typing
        cm = ds.samples
        cm[:np.size(p),:]=cm[p,:]
        cm[:,:np.size(p)]=cm[:,p]
        cm = cm[:np.size(p),:np.size(p)]
        
        # make the constructed matrix a Dataset that matshow() can handle
        # indicates that the current interface is rather clumsy
        ds_cm = Dataset(cm)
        ds_cm.sa['predictions'] = labels
        ds_cm.fa['targets'] = labels
    else:
        ds_cm = ds

    _numbers = None
    if numbers:
        _numbers = {'fontsize': 14,
                    'horizontalalignment': 'center',
                    'verticalalignment': 'center'}
        if numbers_alpha is not None:
            _numbers['numbers_alpha']=numbers_alpha              
        
    im = matshow(ds_cm, 
                 xlabel_attr='predictions', 
                 ylabel_attr='targets',
                 numbers = _numbers)
                                                  
    if xlabels_vertical:
        pl.setp(pl.getp(im.axes, 'xticklabels'), rotation='vertical')

    pl.show()
    



       
