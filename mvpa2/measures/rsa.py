# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Representational (dis)similarity analysis"""

__docformat__ = 'restructuredtext'

from itertools import combinations
import numpy as np
from mvpa2.measures.base import Measure
from mvpa2.datasets.base import Dataset
from mvpa2.base import externals
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import EnsureChoice

if externals.exists('scipy', raise_=True):
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import rankdata, pearsonr

class PDist(Measure):
    """Compute dissimiliarity matrix for samples in a dataset

    This `Measure` returns the upper triangle of the n x n disimilarity matrix
    defined as the pairwise distances between samples in the dataset, and where
    n is the number of samples.
    """

    is_trained = True # Indicate that this measure is always trained.

    pairwise_metric = Parameter('correlation', constraints='str', doc="""\
          Distance metric to use for calculating pairwise vector distances for
          dissimilarity matrix (DSM).  See scipy.spatial.distance.pdist for
          all possible metrics.""")

    center_data = Parameter(False, constraints='bool', doc="""\
          If True then center each column of the data matrix by subtracing the
          column mean from each element. This is recommended especially when
          using pairwise_metric='correlation'.""")

    square = Parameter(False, constraints='bool', doc="""\
          If True return the square distance matrix, if False, returns the
          flattened upper triangle.""")

    def __init__(self, **kwargs):
        """
        Returns
        -------
        Dataset
          If square is False, contains a column vector of length = n(n-1)/2 of
          pairwise distances between all samples. A sample attribute ``pairs``
          identifies the indices of input samples for each individual pair.
          If square is True, the dataset contains a square dissimilarty matrix
          and the entire sample attributes collection of the input dataset.
        """

        Measure.__init__(self, **kwargs)

    def _call(self,ds):

        data = ds.samples
        # center data if specified
        if self.params.center_data:
            data = data - np.mean(data,0)

        # get dsm
        dsm = pdist(data,metric=self.params.pairwise_metric)

        # if square return value make dsm square
        if self.params.square:
            # re-add the sample attributes -- should still be valid
            out = Dataset(squareform(dsm),
                          sa=ds.sa)
        else:
            # add some attributes
            out = Dataset(dsm,
                          sa=dict(pairs=list(combinations(range(len(ds)), 2))))
        return out


class PDistConsistency(Measure):
    """Calculate the correlations of PDist measures across chunks

    This measures the consistency in similarity structure across runs
    within individuals, or across individuals if the target dataset is made from
    several subjects in some common space and where the sample attribute
    specified as the chunks_attr codes for subject identity.

    @author: ACC Aug 2013
    """
    is_trained = True
    """Indicate that this measure is always trained."""

    chunks_attr = Parameter('chunks', constraints='str', doc="""\
          Chunks attribute to use for chunking dataset. Can be any samples
          attribute.""")

    pairwise_metric = Parameter('correlation', constraints='str', doc="""\
          Distance metric to use for calculating dissimilarity matrices from
          the set of samples in each chunk specified. See
          spatial.distance.pdist for all possible metrics.""")

    consistency_metric = Parameter('pearson',
                                   constraints=EnsureChoice('pearson',
                                                            'spearman'),
                                   doc="""\
          Correlation measure to use for the correlation between dissimilarity
          matrices.""")

    center_data = Parameter(False, constraints='bool', doc="""\
          If True then center each column of the data matrix by subtracing the
          column mean from each element. This is recommended especially when
          using pairwise_metric='correlation'.""")

    square = Parameter(False, constraints='bool', doc="""\
          If True return the square distance matrix, if False, returns the
          flattened upper triangle.""")


    def __init__(self, **kwargs):
        """
        Returns
        -------
        Dataset
          Contains the pairwise correlations between the DSMs
          computed from each chunk of the input dataset. If square is False,
          this is a column vector of length N(N-1)/2 for N chunks. If square
          is True, this is a square matrix of size NxN for N chunks.
        """
        # TODO: Another metric for consistency metric could be the "Rv"
        # coefficient...  (ac)
        # init base classes first
        Measure.__init__(self, **kwargs)

    def _call(self, dataset):
        """Computes the average correlation in similarity structure across chunks."""

        chunks_attr = self.params.chunks_attr
        nchunks = len(dataset.sa[chunks_attr].unique)
        if nchunks < 2:
            raise StandardError("This measure calculates similarity consistency across "
                                "chunks and is not meaningful for datasets with only "
                                "one chunk:")
        dsms = []
        chunks = []
        for chunk in dataset.sa[chunks_attr].unique:
            data = np.atleast_2d(
                    dataset.samples[dataset.sa[chunks_attr].value == chunk,:])
            if self.params.center_data:
                data = data - np.mean(data,0)
            dsm = pdist(data, self.params.pairwise_metric)
            dsms.append(dsm)
            chunks.append(chunk)
        dsms = np.vstack(dsms)

        if self.params.consistency_metric=='spearman':
            dsms = np.apply_along_axis(rankdata, 1, dsms)
        corrmat = np.corrcoef(dsms)
        if self.params.square:
            ds = Dataset(corrmat, sa={self.params.chunks_attr: chunks})
        else:
            ds = Dataset(squareform(corrmat,checks=False),
                         sa=dict(pairs=list(combinations(chunks, 2))))
        return ds

class PDistTargetSimilarity(Measure):
    """Calculate the correlations of PDist measures with a target

    Target dissimilarity correlation `Measure`. Computes the correlation between
    the dissimilarity matrix defined over the pairwise distances between the
    samples of dataset and the target dissimilarity matrix.
    """

    is_trained = True
    """Indicate that this measure is always trained."""

    pairwise_metric = Parameter('correlation', constraints='str', doc="""\
          Distance metric to use for calculating pairwise vector distances for
          dissimilarity matrix (DSM).  See scipy.spatial.distance.pdist for
          all possible metrics.""")

    comparison_metric = Parameter('pearson',
                                   constraints=EnsureChoice('pearson',
                                                            'spearman'),
                                   doc="""\
          Similarity measure to be used for comparing dataset DSM with the
          target DSM.""")

    center_data = Parameter(False, constraints='bool', doc="""\
          If True then center each column of the data matrix by subtracing the
          column mean from each element. This is recommended especially when
          using pairwise_metric='correlation'.""")

    corrcoef_only = Parameter(False, constraints='bool', doc="""\
          If True, return only the correlation coefficient (rho), otherwise
          return rho and probability, p.""")

    def __init__(self, target_dsm, **kwargs):
        """
        Parameters
        ----------
        target_dsm : array (length N*(N-1)/2)
          Target dissimilarity matrix

        Returns
        -------
        Dataset
          If ``corrcoef_only`` is True, contains one feature: the correlation
          coefficient (rho); or otherwise two-fetaures: rho plus p.
        """
        # init base classes first
        Measure.__init__(self, **kwargs)
        self.target_dsm = target_dsm
        if self.params.comparison_metric == 'spearman':
            self.target_dsm = rankdata(target_dsm)

    def _call(self,dataset):
        data = dataset.samples
        if self.params.center_data:
            data = data - np.mean(data,0)
        dsm = pdist(data,self.params.pairwise_metric)
        if self.params.comparison_metric=='spearman':
            dsm = rankdata(dsm)
        rho, p = pearsonr(dsm,self.target_dsm)
        if self.params.corrcoef_only:
            return Dataset([rho], fa={'metrics': ['rho']})
        else:
            return Dataset([[rho,p]], fa={'metrics': ['rho', 'p']})
