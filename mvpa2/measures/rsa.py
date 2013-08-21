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
    """
    Dissimilarity Matrix `Measure` returns the lower triangle of the n x n
    disimilarity matrix defined as the pairwise distances between samples in
    the dataset, and where n is the number of samples.  
    """
    
    is_trained = True # Indicate that this measure is always trained.

    def __init__(self, pairwise_metric='correlation', center_data=True,
                    square=False, **kwargs):
        """Initialize

        Parameters
        ----------

        pairwise_metric :   String. Distance metric to use for calculating 
                            pairwise vector distances for dissimilarity matrix 
                            (DSM).  See scipy.spatial.distance.pdist for all 
                            possible metrics.  (Default ='correlation', i.e. one 
                            minus Pearson correlation) 
        center_data :       boolean. (Optional. Default = True) If True then center 
                            each column of the data matrix by subtracing the column 
                            mean from each element  (by chunk if chunks_attr 
                            specified). This is recommended especially when using 
                            pairwise_metric = 'correlation'.  
        square :            boolean. (Optional.  Default = False) If True return 
                            the square distance matrices, if False, returns the 
                            flattened lower triangle.
    
        Returns
        -------
        Dataset :           Contains a row vector of pairwise distances between
                            all samples if square = False; square dissimilarty
                            matrix if square = True.
        """

        Measure.__init__(self, **kwargs) 
        self.pairwise_metric = pairwise_metric
        self.center_data = center_data 
        self.square = square

    def _call(self,ds):
       
        data = ds.samples
        # center data if specified
        if self.center_data:
            data = data - np.mean(data,0)
        
        # get dsm 
        dsm = pdist(data,metric=self.pairwise_metric)
        
        # if square return value make dsm square 
        if self.square:
            dsm = squareform(dsm)
        else:
            dsm = dsm.reshape((1,-1))
        return Dataset(dsm) 


class DissimilarityConsistencyMeasure(Measure):
    """
    Dissimilarity Consistency `Measure` calculates the correlations across
    chunks for pairwise dissimilarity matrices defined over the samples in each
    chunk.

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
    

