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

if __debug__:
    from mvpa2.base import debug

from itertools import combinations, combinations_with_replacement, product
import numpy as np
import tempfile, os
import time

import mvpa2
from mvpa2.base.progress import ProgressBar
from mvpa2.base.types import is_datasetlike
from mvpa2.measures.base import Measure
from mvpa2.measures.searchlight import BaseSearchlight, Searchlight
from mvpa2.datasets.base import Dataset
from mvpa2.base import externals
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import EnsureChoice
from mvpa2.base.dataset import hstack
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.generators.splitters import Splitter

if externals.exists('scipy', raise_=True):
    from scipy.spatial.distance import pdist, squareform, cdist
    from scipy.stats import rankdata, pearsonr
    import scipy.sparse


class CrossNobis(Measure):

    """Compute cross-validated mahalanobis distance for samples in a dataset

    This `Measure` can be trained on part of the dataset (for example,
    a partition) and called on another partition. It can be used in
    cross-validation to generate cross-validated RSA.
    Returns flattened dissimilarity values.
    """
    
    def __init__(self, **kwargs):
        Measure.__init__(self, **kwargs)
        self._train_ds = None
        
    def _train(self, ds):
        self._train_ds = None

        targets_sa_name = self.get_space()
        targets_sa = ds.sa[targets_sa_name]
        self.ulabels = ulabels = targets_sa.unique
        
        targets_sort_idx = np.argsort(targets_sa.value)

        self._train_pairs = Dataset(
            [ds.samples[i]-ds.samples[j] for ii,i in enumerate(targets_sort_idx) for j in targets_sort_idx[ii+1:]])
        self._train_pairs.targets = np.asarray(
            [(targets_sa[i], targets_sa[j]) for ii,i in enumerate(targets_sort_idx) for j in targets_sort_idx[ii+1:] ])
        
        self.is_trained = True

    def _call(self, ds):
        n_ulabels = len(self.ulabels)
        pair_dists = np.zeros((n_ulabels,n_ulabels))
        pair_counts = np.zeros((n_ulabels,n_ulabels), dtype=np.int)
        
        targets_sa_name = self.get_space()
        targets_sa = ds.sa[targets_sa_name]
        ulabels = targets_sa.unique
        label2index = dict((l, il) for il, l in enumerate(ulabels))

        if np.any(ulabels != self.ulabels):
            raise ValueError('Datasets should have same targets for dissimilarity.')
        
        targets_sort_idx = np.argsort(targets_sa.value)        

        test_pairs = Dataset(
            [ds.samples[i]-ds.samples[j] for ii,i in enumerate(targets_sort_idx) for j in targets_sort_idx[ii+1:]])
        test_pairs.targets = np.asarray(
            [(targets_sa[i], targets_sa[j]) for ii,i in enumerate(targets_sort_idx) for j in targets_sort_idx[ii+1:] ])
        
        for ii in range(self._train_pairs.nsamples):
            pair1 = self._train_pairs.targets[ii]
            for jj in range(test_pairs.nsamples):
                pair2 = test_pairs.targets[jj]
                if np.all(pair1 == pair2):
                    a,b = label2index[pair1[0]], label2index[pair1[1]]
                    pair_dists[a,b] += self._train_pairs.samples[ii].dot(test_pairs.samples[jj])/ds.nfeatures
                    pair_counts[a,b] += 1

        pair_dists /= pair_counts
        
        distds = Dataset(pair_dists, 
                         sa=dict(targets=ulabels),
                         fa=dict(targets=ulabels))

        return distds


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
        dsms = np.hstack(dsms)

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
            coefs = vstack((coefs, reg_.intercept_))
            sa += ['intercept']

        return Dataset(coefs, sa={'coefs': sa})


class CrossNobisSearchlight(Searchlight):

    def __init__(self, generator, queryengine,
                 splitter=None,
                 **kwargs):
        """Initialize the base class for "naive" searchlight classifiers

        Parameters
        ----------
        generator : `Generator`
          Some `Generator` to prepare partitions for cross-validation.
          It must not change "targets", thus e.g. no AttributePermutator's
        splitter : Splitter, optional
          Which will be used to split partitioned datasets.  If None specified
          then standard one operating on partitions will be used
        """

        # init base class first
        Searchlight.__init__(self, None, queryengine, **kwargs)

        self._generator = generator
        self._splitter = splitter

        self.__all_pairs = None

    def _untrain(self):
        super(CrossNobisSearchlight, self)._untrain()
        self.__all_pairs = None    

    def _sl_call(self, dataset, roi_ids, nproc, dataset_residuals=None):
        """Call to CrossNobisSearchlight
        """
        
        # Local bindings
        generator = self._generator
        qe = self.queryengine

        if __debug__:
            time_start = time.time()

        targets_sa_name = self.get_space()
        targets_sa = dataset.sa[targets_sa_name]

        if __debug__:
            debug_slc_ = 'SLC_' in debug.active

        # get the dataset information into easy vars
        X = dataset.samples
        if len(X.shape) != 2:
            raise ValueError(
                  'Unlike a classifier, %s (for now) operates on already'
                  'flattened datasets' % (self.__class__.__name__))
        labels = targets_sa.value
        self._ulabels = ulabels = targets_sa.unique
        nlabels = len(ulabels)
        label2index = dict((l, il) for il, l in enumerate(ulabels))
        labels_numeric = np.array([label2index[l] for l in labels])
        self._ulabels_numeric = [label2index[l] for l in ulabels]
        # set the feature dimensions
        nsamples = len(X)
        nrois = len(roi_ids)

        # 1. Query generator for the splits we will have
        if __debug__:
            debug('SLC',
                  'Phase 1. Initializing partitions using %s on %s'
                  % (generator, dataset))

        # Lets just create a dummy ds which will store for us actual sample
        # indicies
        # XXX we could make it even more lightweight I guess...
        dataset_indicies = Dataset(np.arange(nsamples), sa=dataset.sa)

        splitter = Splitter(attr=generator.get_space(), attr_values=[1, 2]) \
            if self._splitter is None \
            else self._splitter

        partitions = list(generator.generate(dataset_indicies)) \
            if generator \
            else [dataset_indicies]

        if __debug__:
            for p in partitions:
                assert(p.shape[1] == 1)
                if not (np.all(p.sa[targets_sa_name].value == labels[p.samples[:, 0]])):
                    raise NotImplementedError(
                        "%s does not yet support partitioners altering the targets "
                        "(e.g. permutators)" % self.__class__)

        nsplits = len(partitions)
        # ATM we need to keep the splits instead since they are used
        # in two places in the code: step 2 and 5
        # We care only about training and testing partitions (i.e. first two)
        self._splits = list(tuple(splitter.generate(ds_))[:2] for ds_ in partitions)
        del partitions                    # not used any longer
        self._nsplits = len(self._splits)

        self._all_pairs = dict()
        self._all_pairs_targets = dict()
        self._splits_idx = []


        # precompute all the intra-chunk pairs difference used in the cross-validation
        if __debug__:
            debug('SLC',
                  'Phase 2. Precompute all samples paired differences per split')

        for splits2 in self._splits:
            self._splits_idx.append([])
            for split in splits2:
                self._splits_idx[-1].append([])
                for ii,i in enumerate(split.samples[:,0]):
                    ti = labels_numeric[i]
                    for j in split.samples[:ii,0]:
                        tj = labels_numeric[j]
                        pair = (i,j)
                        self._splits_idx[-1][-1].append(pair)
                        if pair in self._all_pairs:
                            continue
                        # order the targets to always fill the lower triangle of the dissimilarity matrix
                        if ti<tj:
                            pair_targ = (tj,ti)
                            dif = dataset.samples[j] - dataset.samples[i]
                        else:
                            pair_targ = (ti,tj)
                            dif = dataset.samples[i] - dataset.samples[j]
                        self._all_pairs[pair] = dif
                        self._all_pairs_targets[pair] = pair_targ

        # estimate the residual covariance from the training sets only
        self._splits_cov = None
        if dataset_residuals is not None:
            if __debug__:
                debug('SLC',
                      'Phase 2b. Precompute the feature covariance per split')
            externals.exists('skl', raise_=True) 
            from sklearn.covariance import ledoit_wolf_shrinkage
               
            dataset_indicies = Dataset(np.arange(dataset_residuals.nsamples), sa=dataset_residuals.sa)
            partitions = list(generator.generate(dataset_indicies)) \
                         if generator \
                            else [dataset_indicies]
            
            train_sets = [list(splitter.generate(ds_))[0] for ds_ in partitions]
            self._splits_cov = []
            self._splits_cov2 = []
            self._splits_cov_nsamples = []
            #self._splits_cov_shrinkage = []

            if __debug__:
                debug('SLC',
                      'Phase 2b1. Compute neighborhoods')
            sl_ext_conn = set()
            for f in roi_ids:
                neighs = sorted(self._queryengine[f])
                #sl_ext_conn.update([(n1,n2) for ni,n1 in enumerate(neighs) for n2 in neighs[:ni+1]])
                sl_ext_conn.update(combinations_with_replacement(neighs,2))
                #sl_ext_conn.update(product(neighs,neighs))
            sl_ext_conn = np.sort(np.array(list(sl_ext_conn),dtype=[('row',np.uint),('col',np.uint)]))
            self._sl_ext_conn = sl_ext_conn.view(np.uint).reshape(-1,2).T.copy()
            del sl_ext_conn
            
            blocksize = int(1e5)
            for split_idx, train_idx in enumerate(train_sets):
                
                if __debug__:
                    debug('SLC',
                          'Phase 2b2. Compute covariances, split %d/%d'%(split_idx, len(train_sets)),cr=True)
                cov_tmp = np.empty(self._sl_ext_conn.shape[1])
                cov_tmp2 = np.empty(self._sl_ext_conn.shape[1])
                
                resid = dataset_residuals.samples[train_idx.samples.ravel()]
                resid2 = resid**2
                nsamp = train_idx.nsamples

                for i in range(int(len(cov_tmp)/blocksize+1)):
                    slz = slice(i*blocksize,(i+1)*blocksize)
                    cov_tmp[slz] = np.einsum('ij, ij->j',
                                             resid[:,self._sl_ext_conn[0,slz]],
                                             resid[:,self._sl_ext_conn[1,slz]])
                    cov_tmp2[slz] = np.einsum('ij, ij->j',
                                             resid2[:,self._sl_ext_conn[0,slz]],
                                             resid2[:,self._sl_ext_conn[1,slz]])
                cov_tmp /= nsamp
                cov_tmp2 /= nsamp
                self._splits_cov.append(cov_tmp)
                self._splits_cov2.append(cov_tmp2)
                self._splits_cov_nsamples.append(nsamp)
                #self._splits_cov_shrinkage.append(ledoit_wolf_shrinkage(train_ds.samples))
            if __debug__:
                debug('SLC','')


        if nproc is not None and nproc > 1:
            # split all target ROIs centers into `nproc` equally sized blocks
            nproc_needed = min(len(roi_ids), nproc)
            nblocks = nproc_needed \
                      if self.nblocks is None else self.nblocks
            roi_blocks = np.array_split(roi_ids, nblocks)

            # the next block sets up the infrastructure for parallel computing
            # this can easily be changed into a ParallelPython loop, if we
            # decide to have a PP job server in PyMVPA
            import pprocess
            p_results = pprocess.Map(limit=nproc_needed)
            if __debug__:
                debug('SLC', "Starting off %s child processes for nblocks=%i"
                      % (nproc_needed, nblocks))
            compute = p_results.manage(
                        pprocess.MakeParallel(self._proc_block))
            for iblock, block in enumerate(roi_blocks):
                # should we maybe deepcopy the measure to have a unique and
                # independent one per process?
                seed = mvpa2.get_random_seed()
                compute(block, dataset, seed=seed)
        else:
            # otherwise collect the results in an 1-item list
            p_results = [self._proc_block(roi_ids, dataset  )]

        # Finally collect and possibly process results
        # p_results here is either a generator from pprocess.Map or a list.
        # In case of a generator it allows to process results as they become
        # available
        
        result_ds = hstack([pr for pr in p_results])
        return result_ds

    def _proc_block(self, block, ds, seed=None) :
        """Little helper to capture the parts of the computation that can be
        parallelized

        Parameters
        ----------
        seed
          RNG seed.  Should be provided e.g. in child process invocations
          to guarantee that they all seed differently to not keep generating
          the same sequencies due to reusing the same copy of numpy's RNG
        block
          Critical for generating non-colliding temp filenames in case
          of hdf5 backend.  Otherwise RNGs of different processes might
          collide in their temporary file names leading to problems.
        """
        if seed is not None:
            mvpa2.seed(seed)
        if __debug__:
            debug_slc_ = 'SLC_' in debug.active
            debug('SLC',
                  "Starting computing block for %i elements" % len(block))
            start_time = time.time()

        ulabels = self._ulabels_numeric
        nlabels = len(ulabels)
        n_pair_targets = nlabels*(nlabels-1)/2+nlabels

        res = np.zeros(n_pair_targets)
        counts = np.zeros_like(res, dtype=np.uint)

        target_pairs = [(ul1,ul2) for uli, ul1 in enumerate(self._ulabels) for ul2 in self._ulabels[:uli]]

        results = Dataset(np.empty((n_pair_targets*self._nsplits, len(block))),
                          sa=dict(targets=target_pairs),
                          fa=ds[:,block].fa.copy())
        store_roi_feature_ids = self.ca.is_enabled('roi_feature_ids')
        if store_roi_feature_ids:
            results.fa['roi_feature_ids'] = np.zeros(results.nfeatures, dtype=np.object)
        store_roi_sizes = self.ca.is_enabled('roi_sizes')
        if store_roi_sizes:
            results.fa['roi_sizes'] = np.zeros(results.nfeatures, dtype=np.uint)
        store_roi_center_ids = self.ca.is_enabled('roi_center_ids')
        if store_roi_center_ids:
            results.fa['roi_center_ids'] = block


        # put rois around all features in the dataset and compute the
        # measure within them
        bar = ProgressBar()

        if self._splits_cov is not None:
            store_roi_shrinkage = self.ca.is_enabled('roi_shrinkage')
            if store_roi_shrinkage:
                results.fa['roi_shrinkage'] = np.zeros((results.nfeatures,self._nsplits), dtype=np.float)
            cov_mask = np.empty(self._sl_ext_conn.shape[1], dtype=np.bool)

        for i, f in enumerate(block):
            # retrieve the feature ids of all features in the ROI from the query
            # engine
            roi_specs = self._queryengine[f]

            if is_datasetlike(roi_specs):
                # TODO: unittest
                assert(len(roi_specs) == 1)
                roi_fids = roi_specs.samples[0]
            else:
                roi_fids = roi_specs
            roi_fids = np.sort(roi_fids)

            n_fids = len(roi_fids)

            if store_roi_feature_ids:
                results.fa.roi_feature_ids[i] = roi_fids
            if store_roi_sizes:
                results.fa.roi_sizes[i] = n_fids

            if __debug__ and  debug_slc_:
                debug('SLC_', 'For %r query returned roi_specs %r'
                      % (f, roi_specs))

            if n_fids<1:
                results[:,i] = 0
                continue

            if self._splits_cov is not None:
                cov_mask.fill(False)
                # rows are sorted, optimize sparse matrix slicing
                #for l,r in zip(*[np.searchsorted(self._sl_ext_conn[0],roi_fids,s) for s in ['left','right']]):
                for l,r in zip(*np.searchsorted(self._sl_ext_conn[0],[roi_fids,roi_fids+1],'left')):
                    cov_mask[l:r] = True
                cov_mask[cov_mask] = np.any(self._sl_ext_conn[1,cov_mask,np.newaxis] == roi_fids,-1)
                cov_mask_idx = np.argwhere(cov_mask).flatten()
                triu_idx = np.triu_indices(n_fids)
                cov = np.empty((n_fids, n_fids))
                cov2 = np.empty((n_fids, n_fids))
                cov_shrink = np.empty((n_fids, n_fids))
                inv_cov = np.empty((n_fids, n_fids))

            for spi, split2_idx, split2 in zip(range(len(self._splits)), self._splits_idx, self._splits):
                res.fill(0)
                counts.fill(0)
                if self._splits_cov is not None:
                    cov[triu_idx] = self._splits_cov[spi][cov_mask_idx]
                    cov2[triu_idx] = self._splits_cov2[spi][cov_mask_idx]
                    cov[triu_idx[::-1]] = cov[triu_idx]
                    cov2[triu_idx[::-1]] = cov2[triu_idx]
                    # ledoit wolf shrinkage
                    mu = np.sum(np.trace(cov))/n_fids
                    delta_ = cov.copy()
                    delta_.flat[::n_fids+1] -= mu
                    delta = (delta_ ** 2).sum() / n_fids
                    beta_ = 1. / (n_fids * self._splits_cov_nsamples[spi]) * np.sum(cov2 - cov ** 2)
                    beta = min(beta_, delta)
                    shrinkage = beta / delta
                    if store_roi_shrinkage:
                        results.fa.shrinkages[i,spi] = shrinkage
                    
                    cov_shrink[:] = cov*(1-shrinkage)
                    cov_shrink.flat[::n_fids+1] += mu*shrinkage
                    inv_cov[:] = np.linalg.inv(cov_shrink)

                for pair_train in split2_idx[0]:
                    target_train = self._all_pairs_targets[pair_train]
                    vec_train = self._all_pairs[pair_train][roi_fids]
                    t1,t2 = target_train
                    res_idx = (t1*(t1-1)/2+t1) + t2

                    all_test_vecs = np.asarray([self._all_pairs[pair_test][roi_fids] \
                                                for pair_test in split2_idx[1] if self._all_pairs_targets[pair_test]])
                    if self._splits_cov is not None:
                        res[res_idx] += vec_train.dot(inv_cov).dot(all_test_vecs.T).sum()/n_fids
                    else:
                        res[res_idx] += vec_train.dot(all_test_vecs).sum()/n_fids
                    counts[res_idx] += len(all_test_vecs)
                    
                results.samples[spi*n_pair_targets:(spi+1)*n_pair_targets,i] = res/counts
            
            if self._splits_cov is not None:
                del cov, cov2, delta_, cov_shrink, inv_cov, cov_mask_idx, roi_fids
            
            if __debug__:
                msg = 'ROI %i (%i/%i), %i features' % \
                            (f + 1, i + 1, len(block), n_fids)
                debug('SLC', bar(float(i + 1) / len(block), msg), cr=True)

        if __debug__:
            # just to get to new line
            debug('SLC', '')

        return results
