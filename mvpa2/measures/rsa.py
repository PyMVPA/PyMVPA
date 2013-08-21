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
from scipy.spatial.distance import pdist, squareform

class DissimilarityMatrixMeasure(Measure):
    """Dissimilarity Matrix `Measure` returns the lower triangle of the n x n disimilarity matrix
    defined as the pairwise distances between all samples in the dataset, and
    where n is the number of samples.
    """
    
    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, pairwise_metric='correlation', center_data=False,
                    chunks_attr=None, square=False, **kwargs):
        """Initialize

        Parameters
        ----------
        pairwise_metric: Distance metric to use for calculating pairwise vector distances.
            See scipy.spatial.distance.pdist for all possible metrics.
            (Default = 'correlation', i.e. one minus Pearson correlation)
        chunks_attr: Chunks attribute (default = None). Indicates the samples attribute
            to use to partition dataset returning one dissimilarity matrix per chunk
        """
        Measure.__init__(self, **kwargs)
        self.pairwise_metric = pairwise_metric
        self.center_data = center_data
        self.chunks_attr = chunks_attr
        self.square = square

    def _call(self,ds):
        chunks_attr = self.chunks_attr
        dsm = None
        if chunks_attr is None:
            chunks = np.zeros((len(ds.samples)))
        else:
            chunks = ds.sa[chunks_attr]
        for chunk in np.unique(chunks):
            dset = ds[chunks==chunk,:]
            print ds.shape
            if self.center_data:
                ds.samples = ds.samples - np.mean(ds.samples,0)
            pd = pdist(ds.samples,metric=self.pairwise_metric)
            if self.square:
                pd = squareform(pd)
            if dsm is None:
                dsm = pd
            else:
                dsm = np.vstack((dsm,pd))

        return Dataset(dsm) 



class DissimilarityConsistencyMeasure(Measure):
    """Dissimilarity Conistency `Measure` calculates the average
    correlation across chunks in pairwise dissimilarity matrices defined over the
    samples in each chunk.

    This measures the consistency in similarity structure across runs
    within individuals, or across individuals if the target dataset is made from
    several subjects in some common space and where the sample attribute
    specified as the chunks_attr codes for subject identity.

    @author: ACC Aug 2013
    """
    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, chunks_attr='chunks', pairwise_metric='correlation', 
                    consistency_metric='pearson', **kwargs):
        """Initialize

        Parameters
        ----------
        chunks_attr:        Chunks attribute to use for chunking dataset. Can be any
                            samples attribute specified in the dataset.sa dict.
                            (Default: 'chunks')

        pairwise_metric:    Distance metric to use for calculating dissimilarity
                            matrices from the set of samples in each chunk specified.
                            See spatial.distance.pdist for all possible metrics.
                            (Default = 'correlation', i.e. one minus Pearson correlation)

        consistency_metric: Correlation measure to use for the correlation
                            between dissimilarity matrices. Options are
                            'pearson' (default) or 'spearman'

        Returns
        -------
        Dataset:    Contains an array of the pairwise correlations between the
                    DSMs defined for each chunk of the dataset. Length of array
                    will be N(N-1)/2 for N chunks.

        To Do:
        Another metric for consistency metric could be the "Rv" coefficient...  (ac)
        """
        # init base classes first
        Measure.__init__(self, **kwargs)

        self.pairwise_metric = pairwise_metric
        self.consistency_metric = consistency_metric
        self.chunks_attr = chunks_attr

    def _call(self, dataset):
        """Computes the average correlation in similarity structure across chunks."""
        
        chunks_attr = self.chunks_attr
        nchunks = len(np.unique(dataset.sa[chunks_attr]))
        if nchunks < 2:
            raise StandardError("This measure calculates similarity consistency across "
                                "chunks and is not meaningful for datasets with only "
                                "one chunk:")
        sims = None
    
        for chunk in np.unique(dataset.sa[chunks_attr]):
            ds = dataset[dataset.sa[chunks_attr]==chunk,:]
            dsm = pdist(ds.samples,self.pairwise_metric)
            #print dsm.shape
            if sims is None:
                sims = dsm
            else:
                sims = np.vstack((sims,dsm))

        if self.consistency_metric=='spearman':
            sims = np.apply_along_axis(stats.rankdata, 1, sims)
        corrmat = np.corrcoef(sims)
        
        return Dataset(squareform(corrmat,checks=False))
    

