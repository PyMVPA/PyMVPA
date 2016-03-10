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
           'get_cluster_metric_counts', 'get_cluster_pvals']

if __debug__:
    from mvpa2.base import debug

import random
from collections import Counter

import numpy as np

from scipy.ndimage import measurements
from scipy.sparse import dok_matrix


from mvpa2.measures.label import Labeler
from mvpa2.misc.neighborhood import IndexQueryEngine, Sphere
from mvpa2.datasets import Dataset, dataset_wizard
from mvpa2.mappers.base import ChainMapper
from mvpa2.mappers.flatten import FlattenMapper
from mvpa2.base.learner import Learner
from mvpa2.base.param import Parameter
from mvpa2.base import warning
from mvpa2.base.constraints import \
    EnsureInt, EnsureFloat, EnsureRange, EnsureChoice, EnsureNone, EnsureStr
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
    # Note that  feature_thresh_prob  is relevant only for "cluster forming" so
    # will be irrelevant for e.g. metric="max_value"
    # TODO: address in documentation or some RF to make "Cluster*" implementation
    # specific (e.g. a subclass?)

    chunk_attr = Parameter(
        'chunks', constraints=EnsureStr() | EnsureNone(),
        doc="""Name of the attribute indicating the individual chunks from
            which a single sample each is drawn for averaging into a bootstrap
            sample.  If None, no bootstrapping will be performed, and each
            sample will be used as a NULL result sample, thus there should be
            a sufficient number of them""")
    # TODO: note that n_bootstrap will not be relevant if chunks_attr=None
    #       Address either in documentation or by subclassing...

    fwe_rate = Parameter(
        0.05, constraints=EnsureFloat() & EnsureRange(min=0.0, max=1.0),
        doc="""Family-wise error rate for multiple comparison correction
            of cluster size probabilities.""")

    metric = Parameter(
        'cluster_sizes',
        constraints=EnsureChoice('max_cluster_size', 'cluster_sizes',
                                 'cluster_sizes_non0', 'max_value'),
            doc="""What metric is measured per each value to be aggregated into
            H0 distribution.  'max_cluster_size' collects a distribution of
            maximal cluster size detected in a sample (or 0), 'cluster_sizes'
            collects a distribution of all cluster sizes encountered while providing
            a single 0 value per map if no clusters in that map were detected.
            'cluster_sizes_non0' does not count 0s. 'max_value' collects a distribution
            of maximal values.
            Some metrics ('max_cluster_size', 'max_value') estimate family-wise
            metric so no post-hoc correction is strictly necessary and then
            multicomp_correction will be set to None if not provided explicitly.""")

    multicomp_correction = Parameter(
        'fdr_bh', constraints=EnsureChoice('bonferroni', 'sidak', 'holm-sidak',
                                           'holm', 'simes-hochberg', 'hommel',
                                           'fdr_bh', 'fdr_by', None),
        doc="""Strategy for multiple comparison correction of cluster
            probabilities. All methods supported by statsmodels' ``multitest``
            are available. In addition, ``None`` can be specified to disable
            correction.""")
    # If default metric would be changed from 'cluster_sizes' to some fw metric
    # make multicomp_correction default to None

    labeler = Parameter(
        None,  #  constraints=  NOT SURE TODO
        doc="""``Labeler`` - some learner which if trained on the training
        dataset to group neighboring "spatially" features.
        If None provided, a `Labeler` with IndexQueryEngine operating on
        feature attribute of the space of this instance will be used.  If no
        space is assigned, space of the first FlattenMapper in the ds.a.mapper
        will be used.""")

    map_postproc = Parameter(
        None,
        doc="""A callable to be used to process target as well as each bootstrapped
        sample.  E.g. could be a TFCE mapper (yet TODO)
        """)
    # TODO: relevant only for cluster-based analyses.

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
        if kwargs.get('metric', '').startswith('max_') and \
                'multicomp_correction' not in kwargs:
            # TODO: better just reset multicomp_correction to be None by default
            # and demand setting it explicitly overall (change of behavior!)
            kwargs['multicomp_correction'] = None
        # force disable auto-train: would make no sense
        Learner.__init__(self, auto_train=False, **kwargs)
        self.untrain()

    def _untrain(self):
        self._labeler = None
        self._thrmap = None
        self._null_cluster_sizes = None

    @due.dcite(
        Doi("10.1016/j.neuroimage.2012.09.063"),
        description="Statistical assessment of (searchlight) MVPA results",
        tags=['implementation'])
    def _train(self, ds):
        # shortcuts
        chunk_attr = self.params.chunk_attr
        feature_thresh_prob = self.params.feature_thresh_prob
        n_bootstrap = self.params.n_bootstrap

        if self.params.map_postproc is not None:
            raise NotImplementedError(
                "Support for map_postproc is not yet implemented. Come later"
            )
        if self.params.metric not in ('cluster_sizes', 'cluster_sizes_non0'):
            raise NotImplementedError(
                    "Support for metric %r is not yet implemented"
                    % self.params.metric)

        if chunk_attr is not None:
            # we need to bootstrap
            if 1. / (n_bootstrap + 1) > feature_thresh_prob:
                raise ValueError(
                    'number of bootstrap samples (%d) is insufficient for'
                    ' the desired threshold probability %g'
                    % (n_bootstrap, feature_thresh_prob))
        else:
            if 1. / (len(ds) + 1) > feature_thresh_prob:
                raise ValueError(
                        'number of the NULL samples (%d) is insufficient for'
                        ' the desired threshold probability %g'
                        % (len(ds), feature_thresh_prob))
            raise NotImplementedError("Dealing with chunk_attr=None is not yet")


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
                   for i in xrange(n_bootstrap)]
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
                [get_thresholding_map(d, feature_thresh_prob)
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
                        (d, feature_thresh_prob)
                             for d in featuresegment_producer(segwidth)))
        # store for later thresholding of input data
        self._thrmap = thrmap

        #
        # Step 2: threshold all NULL maps and build distribution of NULL cluster
        #         sizes
        #
        # TODO: yoh: note that the same bcombos which were used to estimate thrmap
        #       will now be used to estimate cluster sizes/metric.
        #       Not sure if such double treatment of the same data doesn't provide
        #       any bias.  We could  a) split chunks pool into two  b) create new
        #       pool of bcombos for metric estimation

        # Labeler is needed to determine "clusters"
        labeler = self.params.labeler
        if labeler is None:
            labeler = _get_default_labeler(ds, fattr=self.space)
            warning("Labeler was not provided, deduced %s" % labeler)
        labeler.train(ds)
        self._labeler = labeler

        # recompute the bootstrap average maps to threshold them and determine
        # cluster sizes

        if __debug__:
            debug('GCTHR', 'Estimating NULL distribution of cluster sizes')

        # common drills
        def thresh_mean(idx):
            """Helper to  mean, apply threshold, wrap into a Dataset
            """
            return Dataset(np.mean(ds_samples[idx], axis=0)[None] > thrmap)
        # kwargs for get_cluster_metric_counts
        gcmc_kw = dict(labeler=labeler, metric=self.params.metric)

        cluster_metric_counts = Counter()
        # this step can be computed in parallel chunks to speeds things up
        if self.params.n_proc == 1:
            # Serial execution
            for sidx in bcombos:
                # this function reverse-maps every sample one-by-one, hence no need
                # to collect chunks of bootstrapped maps
                cluster_metric_counts += get_cluster_metric_counts(
                    thresh_mean(sidx), **gcmc_kw)
        else:
            # Parallel execution
            # same code as above, just restructured for joblib's Parallel
            for jobres in Parallel(n_jobs=self.params.n_proc,
                                   pre_dispatch=self.params.n_proc,
                                   verbose=verbose_level_parallel)(
                delayed(get_cluster_metric_counts)(thresh_mean(sidx), **gcmc_kw)
                for sidx in bcombos
            ):
                # aggregate
                cluster_metric_counts += jobres
        # store cluster size histogram for later p-value evaluation
        # use a sparse matrix for easy consumption (max dim is the number of
        # features, i.e. biggest possible cluster)
        scl = dok_matrix((1, ds.nfeatures + 1), dtype=int)
        for s in cluster_metric_counts:
            scl[0, s] = cluster_metric_counts[s]
        self._null_cluster_sizes = scl

    def _call(self, ds):
        if len(ds) > 1:
            # average all samples into one, assuming we got something like one
            # sample per subject as input
            avgr = mean_sample()
            ds = avgr(ds)
        # threshold input; at this point we only have one sample left
        thrd = ds.samples[0] > self._thrmap

        # # mapper default
        # mapper = IdentityMapper()
        # # overwrite if possible
        # if hasattr(ds, 'a') and 'mapper' in ds.a:
        #     mapper = ds.a.mapper
        # # reverse-map input
        # othrd = _verified_reverse1(mapper, thrd)
        # # TODO: what is your purpose in life osamp? ;-)
        # osamp = _verified_reverse1(mapper, ds.samples[0])
        # osamp_ndim = osamp.ndim

        # prep output dataset
        outds = ds.copy(deep=False)
        outds.fa['featurewise_thresh'] = self._thrmap

        # determine clusters
        labeler = self._labeler
        thrd_labeled = labeler(Dataset(thrd[None, :]))
        assert(len(thrd_labeled) == 1)  # just a single map at a time
        labels = thrd_labeled.samples[0]
        labeler_space = labeler.get_space()
        if labeler_space in thrd_labeled.sa:
            num = thrd_labeled.sa[labeler_space].value[0]
        else:
            # just compute from the result
            num = np.max(labels)

        area = measurements.sum(thrd,
                                labels,
                                index=np.arange(1, num + 1)).astype(int)

        # TODO:  must use labeler.qe's specification and provide those per each
        # one of the fa's used.  Better be absorbed into some function

        # com = measurements.center_of_mass(
        #     osamp, labels=labels, index=np.arange(1, num + 1))
        # maxpos = measurements.maximum_position(
        #     osamp, labels=labels, index=np.arange(1, num + 1))
        # # for the rest we need the labels flattened
        # labels = mapper.forward1(labels)
        # # relabel clusters starting with the biggest and increase index with
        # # decreasing size
        # ordered_labels = np.zeros(labels.shape, dtype=int)
        # ordered_area = np.zeros(area.shape, dtype=int)
        # ordered_com = np.zeros((num, osamp_ndim), dtype=float)
        # ordered_maxpos = np.zeros((num, osamp_ndim), dtype=float)
        # for i, idx in enumerate(np.argsort(area)):
        #     ordered_labels[labels == idx + 1] = num - i
        #     # kinda ugly, but we are looping anyway
        #     ordered_area[i] = area[idx]
        #     ordered_com[i] = com[idx]
        #     ordered_maxpos[i] = maxpos[idx]
        # labels = ordered_labels
        # area = ordered_area[::-1]
        # com = ordered_com[::-1]
        # maxpos = ordered_maxpos[::-1]
        # del ordered_labels  # this one can be big
        # # location info
        # outds.a['clusterlocations'] = \
        #     np.rec.fromarrays(
        #         [com, maxpos], names=('center_of_mass', 'max'))

        # store cluster labels after forward-mapping
        outds.fa['clusters_featurewise_thresh'] = labels.copy()

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


class _ClustersMetric(object):
    """A little helper to make callable specific for the cluster metric

    All metrics should return discrete (int) values since later is used by
    Counter
    """
    def __init__(self, metric):
        self._metric = getattr(self, '_' + metric)

    def __call__(self, map_, labels, num):
        area = measurements.sum(map_, labels, index=np.arange(1, num + 1))
        return self._metric(area)

    def _cluster_sizes(self, area):
        """Metric which for no clusters found returns [0]"""
        if not len(area):
            return [0]
        else:
            return area.astype(int)

    def _cluster_sizes_non0(self, area):
        """Metric which does not count cases where no clusters found"""
        if not len(area):
            return []
        else:
            return area.astype(int)

    def _max_cluster_size(self, area):
        """Metric which returns a list with the maximum cluster size"""
        if not len(area):
            return [0]
        else:
            return [int(area.max())]


def _old_get_map_cluster_sizes(map_):
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


def _get_map_cluster_metrics(map_, metric='cluster_sizes'):
    """Return a list with sizes per each cluster (unordered really) found in the map

    This one is left in for now to verify correct operation
    """
    # For compatibility with older API...?
    # TODO: deprecate entirely

    # we need to label clusters first
    map_ = np.asanyarray(map_)
    map_ds = Dataset([map_])
    flat = FlattenMapper(space='map_coords')
    flat.train(map_ds)
    map_ds_flat = map_ds.get_mapped(flat)
    labeler = Labeler(
        qe=IndexQueryEngine(**{
            'map_coords': Sphere(1)  # Sphere(np.sqrt(map_.ndim) + 0.0001) # for corners case
        })
    )
    labeler.train(map_ds_flat)

    cluster_sizes = get_cluster_metric_counts(
        map_ds_flat,
        labeler=labeler,
        metric=metric
    )  # returns a counter with numbers of clusters of a given size

    # all the logic is in the metrics above.  so if they returned 0 -- must be
    # the way we wanted

    # and this is a list of clusters detected on the map and their sizes
    # so per each one we would pretty much need to repeat it that many times
    return sum([[k] * v for k, v in cluster_sizes.iteritems()], [])


def _get_default_labeler(ds, fattr=None):
    """Given a dataset, deduce which space to operate on by finding first
    :class:`FlattenMapper` and using its space
    """
    if fattr is None:
        # So we need to figure it out
        if 'mapper' not in ds.a:
            raise ValueError(
                "Since no fattr was provided, dataset should have a "
                "mapper to figure out which feature attribute (among %s) to "
                "use to get original 'shape'" % str(ds.fa.keys()))

        mappers = ds.a.mapper
        if not isinstance(mappers, ChainMapper):
            mappers = [mappers]

        for mapper in mappers:
            if isinstance(mapper, FlattenMapper):
                fattr = mapper.get_space()
                if fattr is None:
                    raise ValueError(
                        "Mapper %s of the dataset %s has no space, so can't "
                        "figure out what feature attribute to use to deduce "
                        "neighborhood" % (mapper, ds))
                break

    if fattr is None:
        raise ValueError(
            "No fattr was provided and we could not find a flatten "
            "mapper which was used to produce %s" % ds
        )

    return Labeler(qe=IndexQueryEngine(**{fattr: Sphere(1)}))


def get_cluster_metric_counts(ds, labeler=None, fattr=None, metric='cluster_sizes'):
    """Compute counts for cluster metric (e.g. number of cluster sizes) from all samples in a boolean dataset.

    Individually for each sample, in the input dataset, clusters of non-zero
    values will be determined using labeler.

    TODO: talk about how labeler is created if not provided and how we deduce
    which fa to use for neighborhood

    Parameters
    ----------
    ds : dataset or array
      A dataset with boolean samples.  If an array, we assume that it is samples
      on which to operate on
    labeler : Learner, optional
      `Labeler` to figure out neighbors for each feature of the dataset
    fattr : str, optional
      If no labeler provided, we will use this fattr as coordinates (space) for
      a new `Labeler`
    metric : str, optional
      Metric to be used while estimating clusters statistic across samples. See
      `GroupClusterThreshold`'s parameter metric

    Returns
    -------
    counter
      A collections.Counter (subclass of dict) of cluster sizes from all samples
      in the input dataset (optionally appended to any values passed via
      ``cluster_counter``).
    """
    # XXX input needs to be boolean for the cluster size calculation to work
    if isinstance(ds, np.ndarray):
        ndim = ds.ndim
        ds = dataset_wizard(ds, space='temp_coord')
        if 'temp_coord' not in ds.fa:
            assert('mapper' not in ds.a)  # so no flattening happened
            ds.fa['temp_coord'] = np.arange(ds.nfeatures)
            fattr = 'temp_coord'

    if labeler is None:
        labeler = _get_default_labeler(ds, fattr=fattr)
        labeler.train(ds)

    dslabeled = labeler(ds)
    metric_callable = _ClustersMetric(metric=metric)
    cluster_counter = Counter()
    for d, labels, nlabels in zip(
            ds.samples,
            dslabeled.samples,
            dslabeled.sa[labeler.get_space()].value):
        cluster_counter.update(metric_callable(d, labels, nlabels))
    return cluster_counter


def get_cluster_pvals(sizes, null_sizes):
    """Get p-value per each cluster size given cluster sizes for null-distribution

    Parameters
    ----------
    sizes, null_sizes : Counter
      Counters of cluster sizes (as returned by get_cluster_metric_counts) for target
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


# TODO: consider it consuming Counter not its remanifestation in sparse matrix
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
