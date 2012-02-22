# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""iSmooth - An intelligent smoothing measure.

"""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.measures.base import Measure
from mvpa2.datasets.base import Dataset
from mvpa2.mappers.zscore import zscore
import copy
from numpy.linalg import LinAlgError


class iSmooth(Measure):
    """iSmooth smooths a Dataset 
       using searchlight
       params:
       model = 'regression' or 'correlation'
                correlation code is commented for speed(?)
       cthresh = minimum variance threshold to change timeseries
                    default=0.10
               if -1 is given, then the weight is defaulted 
               to 4mm FWMH Gaussian kernel weight 
               for 3mm voxels, which is 0.241275
       '"""

    def __init__(self, model='regression', cthresh=0.10):
        Measure.__init__(self)
        self.cthresh = cthresh
        self.model = model


    def __call__(self, dataset):
        #if self.model == 'correlation':
        #    orig_ds = copy.deepcopy(dataset)
        #    zscore(orig_ds, chunks_attr=None)
        #    ref_ts = orig_ds[:,orig_ds.fa.roi_seed].samples
        #    corrs = np.mat(ref_ts).T*np.mat(orig_ds.samples)/orig_ds.nsamples
        #    corrs[np.isnan(corrs)] = 0
        #    corrs[abs(corrs)<self.cthresh] = 0
        #    corrs = corrs/np.sum(corrs)
        #    return Dataset(np.asarray(np.mat(orig_ds.samples)*corrs.T))
        #elif self.model == 'regression':
            X = np.mat(dataset[:, dataset.fa.roi_seed!=True].samples)
            y = np.mat(dataset[:, dataset.fa.roi_seed==True].samples)
            try:
                Xi = np.linalg.pinv(X, 1e-5)
                r = y.T*X*Xi*y
                r = r[0,0]**2
            except LinAlgError:
                r = -1000
            if r >= self.cthresh:
                if self.cthresh>=0:
                    ym = (y + r*(X*Xi*y))/(1+r)
                else:
                    ym = (0.241275*y + 0.758725*(X*Xi*y))
                return Dataset(np.asarray(ym))
            else:
                return Dataset(np.asarray(y))
