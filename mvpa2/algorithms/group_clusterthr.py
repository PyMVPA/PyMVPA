# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Cluster thresholding algorithm for a group-level searchlight analysis"""

__docformat__ = 'restructuredtext'

__all__ = ['GroupClusterThreshold', 'get_thresholding_map',
           'get_cluster_sizes', 'get_cluster_pvals']

if __debug__:
    from mvpa2.base import debug

import random
from collections import Counter

import numpy as np

from scipy.ndimage import measurements
from scipy.sparse import dok_matrix

from mvpa2.mappers.base import IdentityMapper, _verified_reverse1
from mvpa2.datasets import Dataset
from mvpa2.base.learner import Learner
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import \
    EnsureInt, EnsureFloat, EnsureRange, EnsureChoice
from mvpa2.mappers.fx import mean_sample

from mvpa2.support.due import due, Doi


class GroupClusterThreshold(Learner):
    """Statistical evaluation of group-level average accuracy maps

    This algorithm can be used to perform cluster-thresholding of
    searchlight-based group analyses. It implements a two-stage procedure that
    uses the results of within-subject permutation analyses, estimates a per
    feature cluster forming threshold (via bootstrap), and uses the thresholded
    bootstrap samples to estimate the distribution of cluster sizes in
    group-average accuracy maps under the NULL hypothesis, as described in [1]_.

    Note: this class implements a modified version of that algorithm. The
    present implementation differs in, at least, four aspects from the
    description in that paper.

    1) Cluster p-values refer to the probability of observing a particular
       cluster size or a larger one (original paper: probability to observe a
       larger cluster only).  Consequently, probabilities reported by this
       implementation will have a tendency to be higher in comparison.

    2) Clusters found in the original (unpermuted) accuracy map are always
       included in the NULL distribution estimate of cluster sizes. This
       provides an explicit lower bound for probabilities, as there will
       always be at least one observed cluster for every cluster size found
       in the original accuracy map. Consequently, it is impossible to get a
       probability of zero for clusters of any size (see [2] for more
       information).

    3) Bootstrap accuracy maps that contain no clusters are counted in a
       dedicated size-zero bin in the NULL distribution of cluster sizes.
       This change yields reliable cluster-probabilities even for very low
       featurewise threshold probabilities, where (some portion) of the
       bootstrap accuracy maps do not contain any clusters.

    4) The method for FWE-correction used by the original authors is not
       provided. Instead, a range of alternatives implemented by the
       statsmodels package are available.

    Moreover, this implementation minimizes the required memory demands and
    allows for computing large numbers of bootstrap samples without
    significant increase in memory demand (CPU time trade-off).

    Instances of this class must be trained before than can be used to
    threshold accuracy maps. The training dataset must match the following
    criteria:

    1) For every subject in the group, it must contain multiple accuracy maps
       that are the result of a within-subject classification analysis
       based on permuted class labels. One map must corresponds to one fixed
       permutation for all features in the map, as described in [1]_. The
       original authors recommend 100 accuracy maps per subject for a typical
       searchlight analysis.

    2) It must contain a sample attribute indicating which sample is
       associated with which subject, because bootstrapping average accuracy
       maps is implemented by randomly drawing one map from each subject.
       The name of the attribute can be configured via the ``chunk_attr``
       parameter.

    After training, an instance can be called with a dataset to perform
    threshold and statistical evaluation. Unless a single-sample dataset
    is passed, all samples in the input dataset will be averaged prior
    thresholding.

    Returns
    -------
    Dataset
      This is a shallow copy of the input dataset (after a potential
      averaging), hence contains the same data and attributes. In addition it
      includes the following attributes:

      ``fa.featurewise_thresh``
        Vector with feature-wise cluster-forming thresholds.

      ``fa.clusters_featurewise_thresh``
        Vector with labels for clusters after thresholding the input data
        with the desired feature-wise probability. Each unique non-zero
        element corresponds to an individual super-threshold cluster. Cluster
        values are sorted by cluster size (number of features). The largest
        cluster is always labeled with ``1``.

      ``fa.clusters_fwe_thresh``
        Vector with labels for super-threshold clusters after correction for
        multiple comparisons. The attribute is derived from
        ``fa.clusters_featurewise_thresh`` by removing all clusters that
        do not pass the threshold when controlling for the family-wise error
        rate.

      ``a.clusterstats``
        Record array with information on all detected clusters. The array is
        sorted according to cluster size, starting with the largest cluster
        in terms of number of features. The array contains the fields ``size``
        (number of features comprising the cluster), ``mean``, ``median``,
        min``, ``max``, ``std`` (respective descriptive statistics for all
        clusters), and ``prob_raw`` (probability of observing the cluster of a
        this size or larger under the NULL hypothesis). If correction for
        multiple comparisons is enabled an additional field ``prob_corrected``
        (probability after correction) is added.

      ``a.clusterlocations``
        Record array with information on the location of all detected clusters.
        The array is sorted according to cluster size (same order as
        ``a.clusterstats``. The array contains the fields ``max``
        (feature coordinate of the maximum score within the cluster, and
        ``center_of_mass`` (coordinate of the center of mass; weighted by
        the feature values within the cluster.

    References
    ----------
    .. [1] Johannes Stelzer, Yi Chen and Robert Turner (2013). Statistical
       inference and multiple testing correction in classification-based
       multi-voxel pattern analysis (MVPA): Random permutations and cluster
       size control. NeuroImage, 65, 69--82.
    .. [2] Smyth, G. K., & Phipson, B. (2010). Permutation P-values Should
       Never Be Zero: Calculating Exact P-values When Permutations Are
       Randomly Drawn. Statistical Applications in Genetics and Molecular
       Biology, 9, 1--12.
    """

    n_bootstrap = Parameter(
        100000, constraints=EnsureInt() & EnsureRange(min=1),
        doc="""Number of bootstrap samples to be generated from the training
            dataset. For each sample, an average map will be computed from a
            set of randomly drawn samples (one from each chunk). Bootstrap
            samples will be used to estimate a featurewise NULL distribution of
            accuracy values for initial thresholding, and to estimate the NULL
            distribution of cluster sizes under the NULL hypothesis. A larger
            number of bootstrap samples reduces the lower bound of
            probabilities, which may be beneficial for multiple comparison
            correction.""")

    feature_thresh_prob = Parameter(
        0.001, constraints=EnsureFloat() & EnsureRange(min=0.0, max=1.0),
        doc="""Feature-wise probability threshold. The value corresponding
            to this probability in the NULL distribution of accuracies will
            be used as threshold for cluster forming. Given that the NULL
            distribution is estimated per feature, the actual threshold value
            will vary across features yielding a threshold vector. The number
            of bootstrap samples need to be adequate for a desired probability.
            A ``ValueError`` is raised otherwise.""")

    chunk_attr = Parameter(
        'chunks',
        doc="""Name of the attribute indicating the individual chunks from
            which a single sample each is drawn for averaging into a bootstrap
            sample.""")

    fwe_rate = Parameter(
        0.05, constraints=EnsureFloat() & EnsureRange(min=0.0, max=1.0),
        doc="""Family-wise error rate for multiple comparison correction
            of cluster size probabilities.""")

    multicomp_correction = Parameter(
        'fdr_bh', constraints=EnsureChoice('bonferroni', 'sidak', 'holm-sidak',
                                           'holm', 'simes-hochberg', 'hommel',
                                           'fdr_bh', 'fdr_by', None),
        doc="""Strategy for multiple comparison correction of cluster
            probabilities. All methods supported by statsmodels' ``multitest``
            are available. In addition, ``None`` can be specified to disable
            correction.""")

    n_blocks = Parameter(
        1, constraints=EnsureInt() & EnsureRange(min=1),
        doc="""Number of segments used to compute the feature-wise NULL
            distributions. This parameter determines the peak memory demand.
            In case of a single segment a matrix of size
            (n_bootstrap x nfeatures) will be allocated. Increasing the number
            of segments reduces the peak memory demand by that roughly factor.
            """)

    n_proc = Parameter(
        1, constraints=EnsureInt() & EnsureRange(min=1),
        doc="""Number of parallel processes to use for computation.
            Requires `joblib` external module.""")

    def __init__(self, **kwargs):
        # force disable auto-train: would make no sense
        Learner.__init__(self, auto_train=False, **kwargs)
        if 1. / (self.params.n_bootstrap + 1) > self.params.feature_thresh_prob:
            raise ValueError('number of bootstrap samples is insufficient for'
                             ' the desired threshold probability')
        self.untrain()

    def _untrain(self):
        self._thrmap = None
        self._null_cluster_sizes = None

    @due.dcite(
        Doi("10.1016/j.neuroimage.2012.09.063"),
        description="Statistical assessment of (searchlight) MVPA results",
        tags=['implementation'])
    def _train(self, ds):
        # shortcuts
        chunk_attr = self.params.chunk_attr
        #
        # Step 0: bootstrap maps by drawing one for each chunk and average them
        # (do N iterations)
        # this could take a lot of memory, hence instead of computing the maps
        # we compute the source maps they can be computed from and then (re)build
        # the matrix of bootstrapped maps either row-wise or column-wise (as
        # needed) to save memory by a factor of (close to) `n_bootstrap`
        # which samples belong to which chunk
        chunk_samples = dict([(c, np.where(ds.sa[chunk_attr].value == c)[0])
                              for c in ds.sa[chunk_attr].unique])
        # pre-built the bootstrap combinations
        bcombos = [[random.sample(v, 1)[0] for v in chunk_samples.values()]
                   for i in xrange(self.params.n_bootstrap)]
        bcombos = np.array(bcombos, dtype=int)
        #
        # Step 1: find the per-feature threshold that corresponds to some p
        # in the NULL
        segwidth = ds.nfeatures / self.params.n_blocks
        # speed things up by operating on an array not a dataset
        ds_samples = ds.samples
        if __debug__:
            debug('GCTHR',
                  'Compute per-feature thresholds in %i blocks of %i features'
                  % (self.params.n_blocks, segwidth))
        # Execution can be done in parallel as the estimation is independent
        # across features

        def featuresegment_producer(ncols):
            for segstart in xrange(0, ds.nfeatures, ncols):
                # one average map for every stored bcombo
                # this also slices the input data into feature subsets
                # for the compute blocks
                yield [np.mean(
                       # get a view to a subset of the features
                       # -- should be somewhat efficient as feature axis is
                       # sliced
                       ds_samples[sidx, segstart:segstart + ncols],
                       axis=0)
                       for sidx in bcombos]
        if self.params.n_proc == 1:
            # Serial execution
            thrmap = np.hstack(  # merge across compute blocks
                [get_thresholding_map(d, self.params.feature_thresh_prob)
                 # compute a partial threshold map for as many features
                 # as fit into a compute block
                 for d in featuresegment_producer(segwidth)])
        else:
            # Parallel execution
            verbose_level_parallel = 50 \
                if (__debug__ and 'GCTHR' in debug.active) else 0
            # local import as only parallel execution needs this
            from joblib import Parallel, delayed
            # same code as above, just in parallel with joblib's Parallel
            thrmap = np.hstack(
                Parallel(n_jobs=self.params.n_proc,
                         pre_dispatch=self.params.n_proc,
                         verbose=verbose_level_parallel)(
                             delayed(get_thresholding_map)
                        (d, self.params.feature_thresh_prob)
                             for d in featuresegment_producer(segwidth)))
        # store for later thresholding of input data
        self._thrmap = thrmap
        #
        # Step 2: threshold all NULL maps and build distribution of NULL cluster
        #         sizes
        #
        cluster_sizes = Counter()
        # recompute the bootstrap average maps to threshold them and determine
        # cluster sizes
        dsa = dict(mapper=ds.a.mapper) if 'mapper' in ds.a else {}
        if __debug__:
            debug('GCTHR', 'Estimating NULL distribution of cluster sizes')
        # this step can be computed in parallel chunks to speeds things up
        if self.params.n_proc == 1:
            # Serial execution
            for sidx in bcombos:
                avgmap = np.mean(ds_samples[sidx], axis=0)[None]
                # apply threshold
                clustermap = avgmap > thrmap
                # wrap into a throw-away dataset to get the reverse mapping right
                bds = Dataset(clustermap, a=dsa)
                # this function reverse-maps every sample one-by-one, hence no need
                # to collect chunks of bootstrapped maps
                cluster_sizes = get_cluster_sizes(bds, cluster_sizes)
        else:
            # Parallel execution
            # same code as above, just restructured for joblib's Parallel
            for jobres in Parallel(n_jobs=self.params.n_proc,
                                   pre_dispatch=self.params.n_proc,
                                   verbose=verbose_level_parallel)(
                                       delayed(get_cluster_sizes)
                                  (Dataset(np.mean(ds_samples[sidx],
                                           axis=0)[None] > thrmap,
                                           a=dsa))
                                       for sidx in bcombos):
                # aggregate
                cluster_sizes += jobres
        # store cluster size histogram for later p-value evaluation
        # use a sparse matrix for easy consumption (max dim is the number of
        # features, i.e. biggest possible cluster)
        scl = dok_matrix((1, ds.nfeatures + 1), dtype=int)
        for s in cluster_sizes:
            scl[0, s] = cluster_sizes[s]
        self._null_cluster_sizes = scl

    def _call(self, ds):
        if len(ds) > 1:
            # average all samples into one, assuming we got something like one
            # sample per subject as input
            avgr = mean_sample()
            ds = avgr(ds)
        # threshold input; at this point we only have one sample left
        thrd = ds.samples[0] > self._thrmap
        # mapper default
        mapper = IdentityMapper()
        # overwrite if possible
        if hasattr(ds, 'a') and 'mapper' in ds.a:
            mapper = ds.a.mapper
        # reverse-map input
        othrd = _verified_reverse1(mapper, thrd)
        # TODO: what is your purpose in life osamp? ;-)
        osamp = _verified_reverse1(mapper, ds.samples[0])
        # prep output dataset
        outds = ds.copy(deep=False)
        outds.fa['featurewise_thresh'] = self._thrmap
        # determine clusters
        labels, num = measurements.label(othrd)
        area = measurements.sum(othrd,
                                labels,
                                index=np.arange(1, num + 1)).astype(int)
        com = measurements.center_of_mass(
            osamp, labels=labels, index=np.arange(1, num + 1))
        maxpos = measurements.maximum_position(
            osamp, labels=labels, index=np.arange(1, num + 1))
        # for the rest we need the labels flattened
        labels = mapper.forward1(labels)
        # relabel clusters starting with the biggest and increase index with
        # decreasing size
        ordered_labels = np.zeros(labels.shape, dtype=int)
        ordered_area = np.zeros(area.shape, dtype=int)
        ordered_com = np.zeros((num, len(osamp.shape)), dtype=float)
        ordered_maxpos = np.zeros((num, len(osamp.shape)), dtype=float)
        for i, idx in enumerate(np.argsort(area)):
            ordered_labels[labels == idx + 1] = num - i
            # kinda ugly, but we are looping anyway
            ordered_area[i] = area[idx]
            ordered_com[i] = com[idx]
            ordered_maxpos[i] = maxpos[idx]
        labels = ordered_labels
        area = ordered_area[::-1]
        com = ordered_com[::-1]
        maxpos = ordered_maxpos[::-1]
        del ordered_labels  # this one can be big
        # store cluster labels after forward-mapping
        outds.fa['clusters_featurewise_thresh'] = labels.copy()
        # location info
        outds.a['clusterlocations'] = \
            np.rec.fromarrays(
                [com, maxpos], names=('center_of_mass', 'max'))

        # update cluster size histogram with the actual result to get a
        # proper lower bound for p-values
        # this will make a copy, because the original matrix is int
        cluster_probs_raw = _transform_to_pvals(
            area, self._null_cluster_sizes.astype('float'))

        clusterstats = (
            [area, cluster_probs_raw],
            ['size', 'prob_raw']
        )
        # evaluate a bunch of stats for all clusters
        morestats = {}
        for cid in xrange(len(area)):
            # keep clusters on outer loop, because selection is more expensive
            clvals = ds.samples[0, labels == cid + 1]
            for id_, fx in (
                    ('mean', np.mean),
                    ('median', np.median),
                    ('min', np.min),
                    ('max', np.max),
                    ('std', np.std)):
                stats = morestats.get(id_, [])
                stats.append(fx(clvals))
                morestats[id_] = stats

        for k, v in morestats.items():
            clusterstats[0].append(v)
            clusterstats[1].append(k)

        if self.params.multicomp_correction is not None:
            # do a local import as only this tiny portion needs statsmodels
            import statsmodels.stats.multitest as smm
            rej, probs_corr = smm.multipletests(
                cluster_probs_raw,
                alpha=self.params.fwe_rate,
                method=self.params.multicomp_correction)[:2]
            # store corrected per-cluster probabilities
            clusterstats[0].append(probs_corr)
            clusterstats[1].append('prob_corrected')
            # remove cluster labels that did not pass the FWE threshold
            for i, r in enumerate(rej):
                if not r:
                    labels[labels == i + 1] = 0
            outds.fa['clusters_fwe_thresh'] = labels
        outds.a['clusterstats'] = \
            np.rec.fromarrays(clusterstats[0], names=clusterstats[1])
        return outds


def get_thresholding_map(data, p=0.001):
    """Return array of thresholds corresponding to a probability of such value in the input

    Thresholds are returned as an array with one value per column in the input
    data.

    Parameters
    ----------
    data : 2D-array
      Array with data on which the cumulative distribution is based.
      Values in each column are sorted and the value corresponding to the
      desired probability is returned.
    p : float [0,1]
      Value greater or equal than the returned threshold have a probability `p` or less.
    """
    # we need NumPy indexing logic, even if a dataset comes in
    data = np.asanyarray(data)
    p_index = int(len(data) * p)
    if p_index < 1:
        raise ValueError("requested probability is too low for the given number of samples")
    # threshold indices are all in one row of the argsorted inputs
    thridx = np.argsort(data, axis=0, kind='quicksort')[-p_index]
    return data[thridx, np.arange(data.shape[1])]


def _get_map_cluster_sizes(map_):
    labels, num = measurements.label(map_)
    area = measurements.sum(map_, labels, index=np.arange(1, num + 1))
    # TODO: So here if a given map didn't have any super-thresholded features,
    # we get 0 into our histogram.  BUT for the other maps, where at least 1 voxel
    # passed the threshold we might get multiple clusters recorded within our
    # distribution.  Which doesn't quite cut it for being called a FW cluster level.
    # MAY BE it should count only the maximal cluster size (a single number)
    # per given permutation (not all of them)
    if not len(area):
        return [0]
    else:
        return area.astype(int)


def get_cluster_sizes(ds, cluster_counter=None):
    """Compute cluster sizes from all samples in a boolean dataset.

    Individually for each sample, in the input dataset, clusters of non-zero
    values will be determined after reverse-applying any transformation of the
    dataset's mapper (if any).

    Parameters
    ----------
    ds : dataset or array
      A dataset with boolean samples.
    cluster_counter : list or None
      If not None, given list is extended with the cluster sizes computed
      from the present input dataset. Otherwise, a new list is generated.

    Returns
    -------
    list
      Unsorted list of cluster sizes from all samples in the input dataset
      (optionally appended to any values passed via ``cluster_counter``).
    """
    # XXX input needs to be boolean for the cluster size calculation to work
    if cluster_counter is None:
        cluster_counter = Counter()

    mapper = IdentityMapper()
    data = np.asanyarray(ds)
    if hasattr(ds, 'a') and 'mapper' in ds.a:
        mapper = ds.a.mapper

    for i in xrange(len(ds)):
        osamp = _verified_reverse1(mapper, data[i])
        m_clusters = _get_map_cluster_sizes(osamp)
        cluster_counter.update(m_clusters)
    return cluster_counter


def get_cluster_pvals(sizes, null_sizes):
    """Get p-value per each cluster size given cluster sizes for null-distribution

    Parameters
    ----------
    sizes, null_sizes : Counter
      Counters of cluster sizes (as returned by get_cluster_sizes) for target
      dataset and null distribution
    """
    # TODO: dedicated unit-test for this function
    """
    Development note:
     Functionality here somewhat duplicates functionality in _transform_to_pvals
     which does not operate on raw "Counters" and requires different input format.
     Altogether with such data preparation _transform_to_pvals was slower than
     this more naive implementation.
    """
    all_sizes = null_sizes + sizes
    total_count = float(np.sum(all_sizes.values()))
    # now we need to normalize them counting all to the "right", i.e larger than
    # current one
    right_tail = 0
    all_sizes_sf = {}
    for cluster_size in sorted(all_sizes)[::-1]:
        right_tail += all_sizes[cluster_size]
        all_sizes_sf[cluster_size] = right_tail/total_count

    # now figure out p values for our cluster sizes in real acc (not the P0 distribution),
    # since some of them might be missing
    all_sizes_sorted = sorted(all_sizes)
    pvals = {}
    for cluster_size in sizes:
        if cluster_size in all_sizes:
            pvals[cluster_size] = all_sizes_sf[cluster_size]
        else:
            # find the largest smaller than current size
            clusters = all_sizes_sorted[all_sizes_sorted < cluster_size]
            pvals[cluster_size] = all_sizes_sf[clusters[-1]]
    return pvals


def repeat_cluster_vals(cluster_counts, vals=None):
    """Repeat vals for each count of a cluster size as given in cluster_counts

    Parameters
    ----------
    cluster_counts: dict or Counter
      Contains counts per each cluster size
    vals : dict or Counter, optional

    Returns
    -------
    ndarray
      Values are ordered according to ascending order of cluster sizes
    """
    sizes = sorted(cluster_counts.keys())
    if vals is None:
        return np.repeat(sizes, [cluster_counts[s] for s in sizes])
    else:
        return np.repeat([vals[s] for s in sizes], [cluster_counts[s] for s in sizes])


def _transform_to_pvals(sizes, null_sizes):
    # null_sizes will be modified in-place
    for size in sizes:
        null_sizes[0, size] += 1
    # normalize histogram
    null_sizes /= null_sizes.sum()
    # compute p-values for each cluster
    cache = {}
    probs = []
    for cidx, csize in enumerate(sizes):
        # try the cache
        prob = cache.get(csize, None)
        if prob is None:
            # no cache
            # probability is the sum of a relative frequencies for clusters
            # larger OR EQUAL than the current one
            prob = null_sizes[0, csize:].sum()
            cache[csize] = prob
        # store for output
        probs.append(prob)
    return probs
