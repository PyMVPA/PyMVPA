# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PLS is not yet implemented
"""
__docformat__ = 'restructuredtext'


import numpy as np

from mvpa.measures.base import FeaturewiseMeasure

if __debug__:
    from mvpa.base import debug


class PLS(FeaturewiseMeasure):
    def __init__(self, num_permutations=200, num_bootstraps=100, **kwargs):
        raise NotImplemented, 'PLS was not yet implemented fully'

        # init base classes first
        FeaturewiseMeasure.__init__(self, **kwargs)

        # save the args for the analysis
        self.num_permutations = num_permutations
        self.num_bootstraps = num_bootstraps
        
    def _calc_pls(self,mat,labels):
        # take mean within condition(label) and concat to make a
        # condition by features matrix
        X = []
        for ul in np.unique(labels):
            X.append(mat[labels==ul].mean(axis=0))
        X = np.asarray(X)
        
        # center each condition by subtracting the grand mean
        X -= X.mean(axis=1)[:,np.newaxis].repeat(X.shape[1],axis=1)
        
        # run SVD (checking to transpose if necessary)
        U,s,Vh = np.linalg.svd(X, full_matrices=0)

        # run procrust to reorder if necessary

    def _procrust():
        pass

    def _call(self,dataset):
        
        # 
        pass

class TaskPLS(PLS):
    pass
