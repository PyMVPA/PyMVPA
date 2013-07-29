# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Measure of consistency between dissimilarity matrices across chunks."""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.measures.base import Measure
from mvpa2.datasets.base import Dataset
from scipy.spatial.distance import *

class dsm(Measure):
    """`Measure` returns the lower triangle of the n x n disimilarity matrix
    defined as the pairwise distances between all samples in the dataset, and
    where n is the number of samples.
    """
    
    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, pairwise_metric='correlation', chunks_attr=None,
                square=False, **kwargs):
        """Initialize

        Parameters
        ----------
        pairwise_metric: Distance metric to use for calculating pairwise vector distances.
            See scipy.spatial.distance.pdist for all possible metrics.
            (Default = 'correlation', i.e. one minus Pearson correlation)
        chunks_attr: Chunks attribute (default = None). Indicates the samples attribute
            to use to parition dataset returning one dissimilarity matrix per chunk
        """
        Measure.__init__(self, **kwargs)
        self.pairwise_metric = pairwise_metric

    def _call(self,dataset):
        return Dataset(pdist(dataset.samples,metric=self.pairwise_metric))



class SimCorr(Measure):
    """`Measure` that calculates the average correlation across chunk (for now)
    in pairwise dissimilarity matrices defined over the samples in each chunk

    This can be a a measure of consitency in similarity structure across runs
    within individuals, or across individuals if the target dataset is made from
    several subjects in standard space and where the chunks attribute code for 
    subject identity.

    @author: Andy 
    """
    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, pairwise_metric='correlation', **kwargs):
        """Initialize

        Parameters
        ----------
        pairwise_metric: Distance metric to use for calculating pairwise vector distances.
        See spatial.distance.pdist for all possible metrics.
        (Default = 'correlation', i.e. one minus Pearson correlation)

        To Do:
        Another possible param is 'consistency_metric' ... Right now this takes the measure
        of consistency between dissimilarity matrices to be Pearson correlation
        This could be changed to make possible spearman correlation or another 
        metric such as the "Rv" coefficient...  (ac)
        """
        # init base classes first
        Measure.__init__(self, **kwargs)

        self.pairwise_metric = pairwise_metric


    def _call(self, dataset):
        """Computes the average correlation in similarity structure acros chunks."""
        
        nchunks = len(np.unique(dataset.chunks))
        if nchunks < 2:
            raise StandardError("This measure calculates simiarity consitency across "
                                "chunks and is not meaningful for datasets with only "
                                "one chunk:")
        sims = None
    
        for chunk in np.unique(dataset.chunks):
            ds = dataset[dataset.chunks==chunk,:]
            dsm = pdist(ds.samples,self.pairwise_metric)
            #print dsm.shape
            if sims is None:
                sims = dsm
            else:
                sims = np.vstack((sims,dsm))
        corrmat = np.corrcoef(sims)
        
        return Dataset(squareform(corrmat,checks=False))
    

