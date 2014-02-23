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
from scipy.stats import rankdata, pearsonr

class DissimilarityMatrixMeasure(Measure):
    """
    Dissimilarity Matrix `Measure` returns the lower triangle of the n x n
    disimilarity matrix defined as the pairwise distances between samples in
    the dataset, and where n is the number of samples.  
    """
    
    is_trained = True # Indicate that this measure is always trained.

    def __init__(self, pairwise_metric='correlation', center_data=False,
                    square=False, **kwargs):
        """Initialize

        Parameters
        ----------

        pairwise_metric :   String. Distance metric to use for calculating 
                            pairwise vector distances for dissimilarity matrix 
                            (DSM).  See scipy.spatial.distance.pdist for all 
                            possible metrics.  (Default ='correlation', i.e. one 
                            minus Pearson correlation) 
        center_data :       boolean. (Optional. Default = False) If True then center 
                            each column of the data matrix by subtracing the column 
                            mean from each element  (by chunk if chunks_attr 
                            specified). This is recommended especially when using 
                            pairwise_metric = 'correlation'.  
        square :            boolean. (Optional.  Default = False) If True return 
                            the square distance matrices, if False, returns the 
                            flattened lower triangle.
    
        Returns
        -------
        Dataset :           Contains a column vector of length = n(n-1)/2 of pairwise 
                            distances between all samples if square = False; square 
                            dissimilarty matrix if square = True.
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
            # add some attributes
            ds = Dataset(squareform(dsm))
        else:
            # add some attributes
            ds = Dataset(dsm) 
        return ds


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
                    consistency_metric='pearson', center_data=False, **kwargs):
        """Initialize

        Parameters
        ----------
        chunks_attr :       Chunks attribute to use for chunking dataset. Can be any
                            samples attribute specified in the dataset.sa dict.
                            (Default: 'chunks')
        pairwise_metric :   Distance metric to use for calculating dissimilarity
                            matrices from the set of samples in each chunk specified.
                            See spatial.distance.pdist for all possible metrics.
                            (Default = 'correlation', i.e. one minus Pearson correlation)
        consistency_metric: Correlation measure to use for the correlation
                            between dissimilarity matrices. Options are
                            'pearson' (default) or 'spearman'
        center_data :       boolean. (Optional. Default = False) If True then center 
                            each column of the data matrix by subtracing the column 
                            mean from each element  (by chunk if chunks_attr 
                            specified). This is recommended especially when using 
                            pairwise_metric = 'correlation'.  

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
        self.center_data = center_data

    def _call(self, dataset):
        """Computes the average correlation in similarity structure across chunks."""
        
        chunks_attr = self.chunks_attr
        nchunks = len(np.unique(dataset.sa[chunks_attr]))
        if nchunks < 2:
            raise StandardError("This measure calculates similarity consistency across "
                                "chunks and is not meaningful for datasets with only "
                                "one chunk:")
        dsms = None
    
        for chunk in np.unique(dataset.sa[chunks_attr]):
            data = dataset.samples[dataset.sa[chunks_attr]==chunk,:]
            if self.center_data:
                data = data - np.mean(data,0)
            dsm = pdist(data,self.pairwise_metric)
            #print dsm.shape
            if dsms is None:
                dsms = dsm
            else:
                dsms = np.vstack((dsms,dsm))

        if self.consistency_metric=='spearman':
            dsms = np.apply_along_axis(rankdata, 1, dsms)
        corrmat = np.corrcoef(dsms)
        
        return Dataset(squareform(corrmat,checks=False))
    
class TargetDissimilarityCorrelationMeasure(Measure):
    """
    Target dissimilarity correlation `Measure`. Computes the correlation between
    the dissimilarity matrix defined over the pairwise distances between the
    samples of dataset and the target dissimilarity matrix.
    """
    
    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, target_dsm, pairwise_metric='correlation', 
                    comparison_metric='pearson', center_data = False, 
                    corrcoef_only = False, **kwargs):
        """
        Initialize

        Parameters
        ----------
        dataset :           Dataset with N samples such that corresponding dissimilarity
                            matrix has N*(N-1)/2 unique pairwise distances
        target_dsm :        numpy array, length N*(N-1)/2. Target dissimilarity matrix
        pairwise_metric :   To be used by pdist to calculate dataset DSM
                            Default: 'correlation', 
                            see scipy.spatial.distance.pdist for other metric options.
        comparison_metric : To be used for comparing dataset dsm with target dsm
                            Default: 'pearson'. Options: 'pearson' or 'spearman'
        center_data :       Center data by subtracting mean column values from
                            columns prior to calculating dataset dsm. 
                            Default: False
        corrcoef_only :     If true, return only the correlation coefficient
                            (rho), otherwise return rho and probability, p. 
                            Default: False
        Returns
        -------
        Dataset :           Dataset contains the correlation coefficient (rho) only or
                            rho plus p, when corrcoef_only is set to false.
        """
        # init base classes first
        Measure.__init__(self, **kwargs)
        if comparison_metric not in ['spearman','pearson']:
            raise Exception("comparison_metric %s is not in "
                            "['spearman','pearson']" % comparison_metric)
        self.target_dsm = target_dsm
        if comparison_metric == 'spearman':
            self.target_dsm = rankdata(target_dsm)
        self.pairwise_metric = pairwise_metric
        self.comparison_metric = comparison_metric
        self.center_data = center_data
        self.corrcoef_only = corrcoef_only

    def _call(self,dataset):
        data = dataset.samples
        if self.center_data:
            data = data - np.mean(data,0)
        dsm = pdist(data,self.pairwise_metric)
        if self.comparison_metric=='spearman':
            dsm = rankdata(dsm)
        rho, p = pearsonr(dsm,self.target_dsm)
        if self.corrcoef_only:
            return Dataset(np.array([rho,]))
        else:
            return Dataset(np.array([rho,p]))


class DissimilarityMultiRegressMeasure(Measure):
    """
    This is a dataset measure that computes the multiple linear regression
    between a specified set of model or predictor DSMs and the DSM derived from
    the dataset.

    Note: this transforms all dissimilarity data to ranks before calculating the
    multiple regression.
    """

    is_trained = True # Indicate that this measure is always trained.

    def __init__(self, X, pairwise_metric='correlation', center_data=False,**kwargs):
        """Initialize

        Parameters
        ----------
        X               :   The n X m predictor matrix with n the number of
                            predictors, including a row of all ones for a
                            constant term, and m is the length of the flattened
                            upper triangle of the dissimilarity mattrices. Thus
                            if the dataset in question has j rows, then the
                            value of m = j(j-1)/2.

        pairwise_metric :   String. Distance metric to use for calculating 
                            pairwise vector distances for dissimilarity matrix 
                            (DSM).  See scipy.spatial.distance.pdist for all 
                            possible metrics.  (Default ='correlation', i.e. one 
                            minus Pearson correlation) 
        center_data :       boolean. (Optional. Default = False) If True then center 
                            each column of the data matrix by subtracing the column 
                            mean from each element  (by chunk if chunks_attr 
                            specified). This is recommended especially when using 
                            pairwise_metric = 'correlation'.  
        square :            boolean. (Optional.  Default = False) If True return 
                            the square distance matrices, if False, returns the 
                            flattened lower triangle.
    
        """

        Measure.__init__(self, **kwargs)
        (m,n) = X.shape
        for i in range(m):
            X[i,:] = rankdata(X[i,:])

        self.pairwise_metric = pairwise_metric
        self.center_data = center_data

    def _call(self,ds):

        X = self.X

        data = ds.samples
        # center data if specified
        if self.center_data:
            data = data - np.mean(data,0)

        # get dsm 
        dsm = pdist(data,metric=self.pairwise_metric)
        dsm = rankdata(dsm)
        y = np.mat(dsm.reshape((1,-1)))

        b = y * X.T * np.linalg.inv(X*X.T)
        y_hat = np.dot(b,X)
        r = np.corrcoef(y_hat,y)[1,0]
        rsq = r*r
        b = b[0,:].reshape((-1,1))
        b = np.vstack((rsq,b))
        b[np.isnan(b)] = 0


        return Dataset(b)
           








