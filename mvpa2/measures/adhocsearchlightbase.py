# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Support functionality for GNB and M1NN searchlights"""

__docformat__ = 'restructuredtext'

import numpy as np

#from numpy import ones, zeros, sum, abs, isfinite, dot
#from mvpa2.base import warning, externals
from mvpa2.datasets.base import Dataset
from mvpa2.misc.errorfx import mean_mismatch_error
from mvpa2.measures.searchlight import BaseSearchlight
from mvpa2.base import externals, warning
from mvpa2.base.dochelpers import borrowkwargs, _repr_attrs
from mvpa2.generators.splitters import Splitter

#from mvpa2.base.param import Parameter
#from mvpa2.base.state import ConditionalAttribute
#from mvpa2.measures.base import Sensitivity

from mvpa2.misc.neighborhood import IndexQueryEngine, Sphere

if __debug__:
    from mvpa2.base import debug
    import time as time

if externals.exists('scipy'):
    import scipy.sparse as sps
    # API of scipy.sparse has changed in 0.7.0 -- lets account for this
    _coo_shape_argument = {
        True: 'shape',
        False: 'dims'} [externals.versions['scipy'] >= '0.7.0']

__all__ = [ "SimpleStatSearchlight" ]

def lastdim_columnsums_fancy_indexing(a, inds, out):
    for i, inds_ in enumerate(inds):
        out[..., i] = a[..., inds_].sum(axis=-1)

#
# Machinery for sparse matrix way
#

# silly Yarik failed to do np.r_[*neighbors] directly, so here is a
# trick
def r_helper(*args):
    return np.r_[args]

def _inds_list_to_coo(inds, shape=None):
    inds_r = r_helper(*(inds))
    inds_i = r_helper(*[[i]*len(ind)
                        for i,ind in enumerate(inds)])
    data = np.ones(len(inds_r))
    ij = np.array([inds_r, inds_i])

    spmat = sps.coo_matrix((data, ij), dtype=int, **{_coo_shape_argument:shape})
    return spmat

def _inds_array_to_coo(inds, shape=None):
    n_sums, n_cols_per_sum = inds.shape
    cps_inds = inds.ravel()
    row_inds = np.repeat(np.arange(n_sums)[None, :],
                         n_cols_per_sum, axis=0).T.ravel()
    ij = np.r_[cps_inds[None, :], row_inds[None, :]]
    data  = np.ones(ij.shape[1])

    inds_s = sps.coo_matrix((data, ij), **{_coo_shape_argument:shape})
    return inds_s

def inds_to_coo(inds, shape=None):
    """Dispatcher for conversion to coo
    """
    if isinstance(inds, np.ndarray):
        return _inds_array_to_coo(inds, shape)
    elif isinstance(inds, list):
        return _inds_list_to_coo(inds, shape)
    else:
        raise NotImplementedError, "add conversion here"

def lastdim_columnsums_spmatrix(a, inds, out):
    # inds is a 2D array or list or already a sparse matrix, with each
    # row specifying a set of columns (in fact last dimension indices)
    # to sum.  Thus there are the same number of sums as there are
    # rows in `inds`.

    n_cols = a.shape[-1]
    in_shape = a.shape[:-1]

    # first convert to sparse if necessary
    if sps.isspmatrix(inds):
        n_sums = inds.shape[1]
        inds_s = inds
    else:                               # assume regular iterable
        n_sums = len(inds)
        inds_s = inds_to_coo(inds, shape=(n_cols, n_sums))

    ar = a.reshape((-1, a.shape[-1]))
    sums = np.asarray((sps.csr_matrix(ar) * inds_s).todense())
    out[:] = sums.reshape(in_shape+(n_sums,))


class _STATS:
    """Just a dummy container to group/access stats
    """
    pass


class SimpleStatBaseSearchlight(BaseSearchlight):
    """Base class for clf searchlights based on basic univar. statistics

    Used for GNB and M1NN Searchlights

    TODO
    ----

    some stats are not needed (eg per sample X^2's) for M1NN, so we
    should make them optional depending on the derived class

    Notes
    -----

    refactored from the original GNBSearchlight

    """

    # TODO: implement parallelization (see #67) and then uncomment
    __init__doc__exclude__ = ['nproc']

    def __init__(self, generator, queryengine, errorfx=mean_mismatch_error,
                 indexsum=None,
                 reuse_neighbors=False,
                 **kwargs):
        """Initialize the base class for "naive" searchlight classifiers

        Parameters
        ----------
        generator : `Generator`
          Some `Generator` to prepare partitions for cross-validation.
          It must not change "targets", thus e.g. no AttributePermutator's
        errorfx : func, optional
          Functor that computes a scalar error value from the vectors of
          desired and predicted values (e.g. subclass of `ErrorFunction`).
        indexsum : ('sparse', 'fancy'), optional
          What use to compute sums over arbitrary columns.  'fancy'
          corresponds to regular fancy indexing over columns, whenever
          in 'sparse', product of sparse matrices is used (usually
          faster, so is default if `scipy` is available).
        reuse_neighbors : bool, optional
          Compute neighbors information only once, thus allowing for
          efficient reuse on subsequent calls where dataset's feature
          attributes remain the same (e.g. during permutation testing)
        """

        # init base class first
        BaseSearchlight.__init__(self, queryengine, **kwargs)

        self._errorfx = errorfx
        self._generator = generator

        # TODO: move into _call since resetting over default None
        #       obscures __repr__
        if indexsum is None:
            if externals.exists('scipy'):
                indexsum = 'sparse'
            else:
                indexsum = 'fancy'
        else:
            if indexsum == 'sparse' and not externals.exists('scipy'):
                warning("Scipy.sparse isn't available so taking 'fancy' as "
                        "'indexsum' method.")
                indexsum = 'fancy'
        self._indexsum = indexsum

        if not self.nproc in (None, 1):
            raise NotImplementedError, "For now only nproc=1 (or None for " \
                  "autodetection) is supported by GNBSearchlight"

        self.__pb = None            # statistics per each block/label
        self.__reuse_neighbors = reuse_neighbors

        # Storage to be used for neighborhood information
        self.__roi_fids = None

    def __repr__(self, prefixes=[]):
        return super(SimpleStatBaseSearchlight, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['generator'])
            + _repr_attrs(self, ['errorfx'], default=mean_mismatch_error)
            + _repr_attrs(self, ['indexsum'])
            + _repr_attrs(self, ['reuse_neighbors'], default=False)
            )

    def _get_space(self):
        raise NotImplementedError("Must be implemented in derived classes")

    def _untrain(self):
        super(SimpleStatBaseSearchlight, self)._untrain()
        self.__pb = None


    def _compute_pb_stats(self, labels_numeric,
                          X, shape):
        #
        # reusable containers which should stay of the same size
        #
        nblocks = shape[0]
        pb = self.__pb = _STATS()

        # sums and sums of squares per each block
        pb.sums = np.zeros(shape)
        # sums of squares
        pb.sums2 = np.zeros(shape)

        pb.nsamples = np.zeros((nblocks,))
        pb.labels = [None] * nblocks

        if np.issubdtype(X.dtype, np.int):
            # might result in overflow e.g. while taking .square which
            # would result in negative variances etc, thus to be on a
            # safe side -- convert to float
            X = X.astype(float)

        X2 = np.square(X)
        # silly way for now
        for l, s, s2, ib in zip(labels_numeric, X, X2, self.__sample2block):
            pb.sums[ib] += s
            pb.sums2[ib] += s2
            pb.nsamples[ib] += 1
            if pb.labels[ib] is None:
                pb.labels[ib] = l
            else:
                assert(pb.labels[ib] == l)

        pb.labels = np.asanyarray(pb.labels)
        # additional silly tests for paranoid
        assert(pb.labels.dtype.kind == 'i')


    def _compute_pl_stats(self, sis, pl):
        """
        Uses blocked stats to get stats across given samples' indexes
        (might be training or testing)

        Parameters
        ----------
        sis : array of int
          Indexes of samples
        *args:
          In-place containers
        """
        # local binding
        pb = self.__pb

        # convert to blocks training split
        bis = np.unique(self.__sample2block[sis])

        # Let's collect stats summaries
        nsamples = 0
        for il, l in enumerate(self._ulabels_numeric):
            bis_il = bis[pb.labels[bis] == l]
            pl.nsamples[il] = N_float = \
                                     float(np.sum(pb.nsamples[bis_il]))
            nsamples += N_float
            if N_float == 0.0:
                pl.variances[il] = pl.sums[il] \
                    = pl.means[il] = pl.sums2[il] = 0.
            else:
                pl.sums[il] = np.sum(pb.sums[bis_il], axis=0)
                pl.means[il] = pl.sums[il] / N_float
                pl.sums2[il] = np.sum(pb.sums2[bis_il], axis=0)

        ## Actually compute the non-0 pl.variances
        non0labels = (pl.nsamples.squeeze() != 0)
        if np.all(non0labels):
            # For a possible tiny speed up avoiding copying and
            # using (no) slicing
            non0labels = slice(None)

        return nsamples, non0labels


    def _sl_call(self, dataset, roi_ids, nproc):
        """Call to SimpleStatBaseSearchlight
        """
        # Local bindings
        generator = self.generator
        qe = self.queryengine
        errorfx = self.errorfx

        if __debug__:
            time_start = time.time()

        targets_sa_name = self._get_space()
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
        ulabels = targets_sa.unique
        nlabels = len(ulabels)
        label2index = dict((l, il) for il, l in enumerate(ulabels))
        labels_numeric = np.array([label2index[l] for l in labels])
        self._ulabels_numeric = [label2index[l] for l in ulabels]
        # set the feature dimensions
        nsamples = len(X)
        nrois = len(roi_ids)
        s_shape = X.shape[1:]           # shape of a single sample
        # The shape of results
        r_shape = (nrois,) + X.shape[2:]

        #
        # Everything toward optimization ;)
        #
        # Silly Yarik thinks that it might be worth to pre-compute
        # statistics per each feature within a block of the samples
        # which always come together in splits -- most often it is a
        # (chunk, label) combination, but since we simply use a
        # generator -- who knows! Therefore lets figure out what are
        # those blocks and operate on them instead of original samples.
        #
        # After additional thinking about this -- probably it would be
        # just minor additional improvements (ie not worth it) but
        # since it is coded already -- let it be so

        # 1. Query generator for the splits we will have
        if __debug__:
            debug('SLC',
                  'Phase 1. Initializing partitions using %s on %s'
                  % (generator, dataset))

        # Lets just create a dummy ds which will store for us actual sample
        # indicies
        # XXX we could make it even more lightweight I guess...
        dataset_indicies = Dataset(np.arange(nsamples), sa=dataset.sa)
        splitter = Splitter(attr=generator.get_space())
        partitions = list(generator.generate(dataset_indicies))
        if __debug__:
            for p in partitions:
                if not (np.all(p.sa[targets_sa_name].value == labels)):
                    raise NotImplementedError(
                        "%s does not yet support partitioners altering the targets "
                        "(e.g. permutators)" % self.__class__)

        nsplits = len(partitions)
        # ATM we need to keep the splits instead since they are used
        # in two places in the code: step 2 and 5
        splits = list(tuple(splitter.generate(ds_)) for ds_ in partitions)
        del partitions                    # not used any longer

        # 2. Figure out the new 'chunks x labels' blocks of combinations
        #    of samples
        if __debug__:
            debug('SLC',
                  'Phase 2. Blocking data for %i splits and %i labels'
                  % (nsplits, nlabels))
        # array of indicies for label, split1, split2, ...
        # through which we will pass later on to figure out
        # unique combinations
        combinations = np.ones((nsamples, 1+nsplits), dtype=int)*-1
        # labels
        combinations[:, 0] = labels_numeric
        for ipartition, (split1, split2) in enumerate(splits):
            combinations[split1.samples[:, 0], 1+ipartition] = 1
            combinations[split2.samples[:, 0], 1+ipartition] = 2
            # Check for over-sampling, i.e. no same sample used twice here
            if not (len(np.unique(split1.samples[:, 0])) == len(split1) and
                    len(np.unique(split2.samples[:, 0])) == len(split2)):
                raise RuntimeError(
                    "%s needs a partitioner which does not reuse "
                    "the same the same samples more than once"
                    % self.__class__)
        # sample descriptions -- should be unique for
        # samples within the same block
        descriptions = [tuple(c) for c in combinations]
        udescriptions = sorted(list(set(descriptions)))
        nblocks = len(udescriptions)
        description2block = dict([(d, i) for i, d in enumerate(udescriptions)])
        # Indices for samples to point to their block
        self.__sample2block = sample2block = \
            np.array([description2block[d] for d in descriptions])

        # 3. Compute statistics per each block
        #
        if __debug__:
            debug('SLC',
                  'Phase 3. Computing statistics for %i blocks' % (nblocks,))

        self._compute_pb_stats(labels_numeric, X, (nblocks,) + s_shape)

        # derived classes might decide differently on what they
        # actually need, so defer reserving the space and computing
        # stats to them
        self._reserve_pl_stats_space((nlabels, ) + s_shape)

        # results
        results = np.zeros((nsplits,) + r_shape)

        # 4. Lets deduce all neighbors... might need to be RF into the
        #    parallel part later on
        # TODO: needs OPT since this is the step consuming 50% of time
        #       or more allow to cache them entirely so this would
        #       not be an unnecessary burden during permutation testing
        if not self.reuse_neighbors or self.__roi_fids is None:
            if __debug__:
                debug('SLC',
                      'Phase 4. Deducing neighbors information for %i ROIs'
                      % (nrois,))
            roi_fids = [qe.query_byid(f) for f in roi_ids]

        else:
            if __debug__:
                debug('SLC',
                      'Phase 4. Reusing neighbors information for %i ROIs'
                      % (nrois,))
            roi_fids = self.__roi_fids

        self.ca.roi_feature_ids = roi_fids

        roi_sizes = []
        if isinstance(roi_fids, list):
            nroi_fids = len(roi_fids)
            if self.ca.is_enabled('roi_sizes'):
                roi_sizes = [len(x) for x in roi_fids]
        elif externals.exists('scipy') and isinstance(roi_fids, sps.spmatrix):
            nroi_fids = roi_fids.shape[1]
            if self.ca.is_enabled('roi_sizes'):
                # very expensive operation, so better not to ask over again
                # roi_sizes = [roi_fids.getrow(r).nnz for r in range(nroi_fids)]
                warning("Since 'sparse' trick is used, extracting sizes of "
                        "roi's are expensive at this point.  Get them from the "
                        ".ca value of the original instance before "
                        "calling again and using reuse_neighbors")
        else:
            raise RuntimeError("Should not be reachable")

        # Since this is ad-hoc implementation of the searchlight, we are not passing
        # those via ds.a  but rather assign directly to self.ca
        self.ca.roi_sizes = roi_sizes

        indexsum = self._indexsum
        if indexsum == 'sparse':
            if not self.reuse_neighbors or self.__roi_fids is None:
                if __debug__:
                    debug('SLC',
                          'Phase 4b. Converting neighbors to sparse matrix '
                          'representation')
                # convert to "sparse representation" where column j contains
                # 1s only at the roi_fids[j] indices
                roi_fids = inds_to_coo(roi_fids,
                                       shape=(dataset.nfeatures, nroi_fids))
            indexsum_fx = lastdim_columnsums_spmatrix
        elif indexsum == 'fancy':
            indexsum_fx = lastdim_columnsums_fancy_indexing
        else:
            raise ValueError, \
                  "Do not know how to deal with indexsum=%s" % indexsum

        # Store roi_fids
        if self.reuse_neighbors and self.__roi_fids is None:
            self.__roi_fids = roi_fids

        # 5. Lets do actual "splitting" and "classification"
        if __debug__:
            debug('SLC', 'Phase 5. Major loop' )


        for isplit, split in enumerate(splits):
            if __debug__:
                debug('SLC', ' Split %i out of %i' % (isplit, nsplits))
            # figure out for a given splits the blocks we want to work
            # with
            # sample_indicies
            training_sis = split[0].samples[:, 0]
            testing_sis = split[1].samples[:, 0]

            # That is the GNB specificity
            targets, predictions = self._sl_call_on_a_split(
                split, X,               # X2 might light to go
                training_sis, testing_sis,
                ## training_nsamples,      # GO? == np.sum(pl.nsamples)
                ## training_non0labels,
                ## pl.sums, pl.means, pl.sums2, pl.variances,
                # passing nroi_fids as well since in 'sparse' way it has no 'length'
                nroi_fids, roi_fids,
                indexsum_fx,
                labels_numeric,
                )

            # assess the errors
            if __debug__:
                debug('SLC', "  Assessing accuracies")

            if errorfx is mean_mismatch_error:
                results[isplit, :] = \
                    (predictions != targets[:, None]).sum(axis=0) \
                    / float(len(targets))
            else:
                # somewhat silly but a way which allows to use pre-crafted
                # error functions without a chance to screw up
                for i, fpredictions in enumerate(predictions.T):
                    results[isplit, i] = errorfx(fpredictions, targets)


        if __debug__:
            debug('SLC', "%s._call() is done in %.3g sec" %
                  (self.__class__.__name__, time.time() - time_start))

        return Dataset(results)

    generator = property(fget=lambda self: self._generator)
    errorfx = property(fget=lambda self: self._errorfx)
    indexsum = property(fget=lambda self: self._indexsum)
    reuse_neighbors = property(fget=lambda self: self.__reuse_neighbors)
