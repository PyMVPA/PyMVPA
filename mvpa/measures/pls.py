#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.measures.base import FeaturewiseDatasetMeasure

if __debug__:
    from mvpa.base import debug


class PLS(FeaturewiseDatasetMeasure):
    def __init__(self, num_permutations=200, num_bootstraps=100, **kwargs):
        # init base classes first
        FeaturewiseDatasetMeasure.__init__(self, **kwargs)

        # save the args for the analysis
        self.num_permutations = num_permutations
        self.num_bootstraps = num_bootstraps
        
    def _calc_pls(self,mat):
        # take mean within condition and concat to make a condition by
        # features matrix

        # center each condition by subtracting the grand mean

        # run SVD (checking to transpose if necessary)
        pass

    def _procrust():
        pass

    def _call(self,dataset):
        
        # 
        pass

class TaskPLS(PLS):
    pass
