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

from itertools import combinations, product
import numpy as np
from mvpa2.measures.base import Measure
from mvpa2.datasets.base import Dataset
from mvpa2.base import externals
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import EnsureChoice
from mvpa2.mappers.fx import mean_group_sample

if externals.exists('scipy', raise_=True):
    from scipy.spatial.distance import pdist, squareform, cdist
    from scipy.stats import rankdata, pearsonr


class CDist(Measure):
    """Compute cross-validated dissimiliarity matrix for samples in a dataset

    This `Measure` can be trained on part of the dataset (for example,
    a partition) and called on another partition. It can be used in
    cross-validation to generate cross-validated RSA.
    Returns flattened dissimilarity values.
    """
    pairwise_metric = Parameter('correlation', constraints='str',
            doc="""Distance metric to use for calculating pairwise vector distances for
            dissimilarity matrix (DSM).  See scipy.spatial.distance.cdist for
            all possible metrics.""")

    pairwise_metric_kwargs = Parameter({},
            doc="""kwargs dictionary passed to cdist. For example,
            if `pairwise_metric='mahalanobis'`, `pairwise_metric_kwargs`
            might contain the inverse of the covariance matrix.""")

    sattr = Parameter(['targets'],
            doc="""List of sample attributes whose unique values will be used to
            identify the samples groups. Typically your category labels or targets.""")

    def __init__(self, **kwargs):
        Measure.__init__(self, **kwargs)
        self._train_ds = None

    def _prepare_ds(self, ds):
        if self.params.sattr is not None:
            mgs = mean_group_sample(attrs=self.params.sattr)
            ds_ = mgs(ds)
        else:
            ds_ = ds.copy()
        return ds_

    def _train(self, ds):
        self._train_ds = self._prepare_ds(ds)
        self.is_trained = True

    def _call(self, ds):
        test_ds = self._prepare_ds(ds)
        if test_ds.nsamples != self._train_ds.nsamples:
            raise ValueError('Datasets should have same sample size for dissimilarity, '\
                             'nsamples for train: %d, test: %d'%(self._train_ds.nsamples,
                                                                 test_ds.nsamples))
        # Call actual distance metric
        distds = cdist(self._train_ds.samples, test_ds.samples,
                       metric=self.params.pairwise_metric,
                       **self.params.pairwise_metric_kwargs)
        # Make target pairs
        sa_dict = dict()
        for k in self._train_ds.sa:
            if k in test_ds.sa:
                sa_dict[k] = list(product(self._train_ds.sa.get(k).value,
                                                   test_ds.sa.get(k).value))

        distds = Dataset(samples=distds.ravel()[:, None], sa=sa_dict)
        return distds


class PDist(Measure):
    """Compute dissimiliarity matrix for samples in a dataset

    This `Measure` returns the upper triangle of the n x n disimilarity matrix
    defined as the pairwise distances between samples in the dataset, and where
    n is the number of samples.
    """

    is_trained = True  # Indicate that this measure is always trained.

    pairwise_metric = Parameter('correlation', constraints='str', doc="""\
          Distance metric to use for calculating pairwise vector distances for
          dissimilarity matrix (DSM).  See scipy.spatial.distance.pdist for
          all possible metrics.""")

    center_data = Parameter(False, constraints='bool', doc="""\
          If True then center each column of the data matrix by subtracting the
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

        super(PDist, self).__init__(**kwargs)

    def _call(self, ds):

        data = ds.samples
        # center data if specified
        if self.params.center_data:
            data = data - np.mean(data, 0)

        # get dsm
        dsm = pdist(data, metric=self.params.pairwise_metric)

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
          If True then center each column of the data matrix by subtracting the
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
        super(PDistConsistency, self).__init__(**kwargs)

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
                dataset.samples[dataset.sa[chunks_attr].value == chunk, :])
            if self.params.center_data:
                data = data - np.mean(data, 0)
            dsm = pdist(data, self.params.pairwise_metric)
            dsms.append(dsm)
            chunks.append(chunk)
        dsms = np.vstack(dsms)

        if self.params.consistency_metric == 'spearman':
            dsms = np.apply_along_axis(rankdata, 1, dsms)
        corrmat = np.corrcoef(dsms)
        if self.params.square:
            ds = Dataset(corrmat, sa={self.params.chunks_attr: chunks})
        else:
            ds = Dataset(squareform(corrmat, checks=False),
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
          If True then center each column of the data matrix by subtracting the
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
          coefficient (rho); or otherwise two-features: rho plus p.
        """
        # init base classes first
        super(PDistTargetSimilarity, self).__init__(**kwargs)
        self.target_dsm = target_dsm
        if self.params.comparison_metric == 'spearman':
            self.target_dsm = rankdata(target_dsm)

    def _call(self, dataset):
        data = dataset.samples
        if self.params.center_data:
            data = data - np.mean(data, 0)
        dsm = pdist(data, self.params.pairwise_metric)
        if self.params.comparison_metric == 'spearman':
            dsm = rankdata(dsm)
        rho, p = pearsonr(dsm, self.target_dsm)
        if self.params.corrcoef_only:
            return Dataset([rho], fa={'metrics': ['rho']})
        else:
            return Dataset([[rho, p]], fa={'metrics': ['rho', 'p']})


class Regression(Measure):
    """
    Given a dataset, compute regularized regression (Ridge or Lasso) on the
    computed neural dissimilarity matrix using an arbitrary number of predictors
    (model dissimilarity matrices).

    Requires scikit-learn
    """

    is_trained = True
    """Indicate that this measure is always trained."""

    # copied from PDist class XXX: ok or pass it in kwargs?
    pairwise_metric = Parameter('correlation', constraints='str', doc="""\
          Distance metric to use for calculating pairwise vector distances for
          dissimilarity matrix (DSM).  See scipy.spatial.distance.pdist for
          all possible metrics.""")

    center_data = Parameter(False, constraints='bool', doc="""\
          If True then center each column of the data matrix by subtracting the
          column mean from each element. This is recommended especially when
          using pairwise_metric='correlation'.""")

    method = Parameter('ridge', constraints=EnsureChoice('ridge', 'lasso'),
                       doc='Compute Ridge (l2) or Lasso (l1) regression')

    alpha = Parameter(1.0, constraints='float', doc='alpha parameter for lasso'
                                                    'regression')

    fit_intercept = Parameter(True, constraints='bool', doc='whether to fit the'
                                                            'intercept')

    rank_data = Parameter(True, constraints='bool', doc='whether to rank the neural dsm and the '
                                                        'predictor dsms before running the regression model')

    normalize = Parameter(False, constraints='bool', doc='if True the predictors and neural dsm will be'
                                                        'normalized (z-scored) prior to the regression (and after '
                                                        'the data ranking, if rank_data=True)')


    def __init__(self, predictors, keep_pairs=None, **kwargs):
        """
        Parameters
        ----------
        predictors : array (N*(N-1)/2, n_predictors)
            array containing the upper triangular matrix in vector form of the
            predictor Dissimilarity Matrices. Each column is a predictor dsm.

        keep_pairs : None or list or array
            indices in range(N*(N-1)/2) to keep before running the regression.
            All other elements will be removed. If None, the regression is run
            on the entire DSM.

        Returns
        -------
        Dataset
            a dataset with n_predictors samples and one feature. If fit_intercept
            is True, the last sample is the intercept.
        """
        super(Regression, self).__init__(**kwargs)

        if len(predictors.shape) == 1:
            raise ValueError('predictors have shape {0}. Make sure the array '
                             'is at least 2d and transposed correctly'.format(predictors.shape))
        self.predictors = predictors
        self.keep_pairs = keep_pairs

    def _call(self, dataset):
        externals.exists('skl', raise_=True)
        from sklearn.linear_model import Lasso, Ridge
        from sklearn.preprocessing import scale

        # first run PDist
        compute_dsm = PDist(pairwise_metric=self.params.pairwise_metric,
                            center_data=self.params.center_data)
        dsm = compute_dsm(dataset)
        dsm_samples = dsm.samples

        if self.params.rank_data:
            dsm_samples = rankdata(dsm_samples)
            predictors = np.apply_along_axis(rankdata, 0, self.predictors)
        else:
            predictors = self.predictors

        if self.params.normalize:
            predictors = scale(predictors, axis=0)
            dsm_samples = scale(dsm_samples, axis=0)

        # keep only the item we want
        if self.keep_pairs is not None:
            dsm_samples = dsm_samples[self.keep_pairs]
            predictors = predictors[self.keep_pairs, :]

        # check that predictors and samples have the correct dimensions
        if dsm_samples.shape[0] != predictors.shape[0]:
            raise ValueError('computed dsm has {0} rows, while predictors have'
                             '{1} rows. Check that predictors have the right'
                             'shape'.format(dsm_samples.shape[0],
                                            predictors.shape[0]))

        # now fit the regression
        if self.params.method == 'lasso':
            reg = Lasso
        elif self.params.method == 'ridge':
            reg = Ridge
        else:
            raise ValueError('I do not know method {0}'.format(self.params.method))
        reg_ = reg(alpha=self.params.alpha, fit_intercept=self.params.fit_intercept)
        reg_.fit(predictors, dsm_samples)

        coefs = reg_.coef_.reshape(-1, 1)

        sa = ['coef' + str(i) for i in range(len(coefs))]

        if self.params.fit_intercept:
            coefs = np.vstack((coefs, reg_.intercept_))
            sa += ['intercept']

        return Dataset(coefs, sa={'coefs': sa})
