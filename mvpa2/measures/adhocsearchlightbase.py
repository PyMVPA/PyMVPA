# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""An efficient implementation of searchlight for GNB.
"""

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

def lastdim_columnsums_fancy_indexing(a, inds, out):#, out=None):
    ## if out is None:
    ##     out_ = np.empty(a.shape[:-1] + (len(inds),))
    ## else:
    ##     out_ = out
    for i, inds_ in enumerate(inds):
        ## if __debug__ and debug_slc_:
        ##     debug('SLC_', "   Doing %i ROIs: %i (%i features) [%i%%]" \
        ##           % (nroi_fids,
        ##              iroi,
        ##              len(roi_fids_),
        ##              float(iroi+1)/nroi_fids*100,), cr=True)
        out[..., i] = a[..., inds_].sum(axis=-1)
    ## # just a new line
    ## if __debug__ and debug_slc_:
    ##     debug('SLC_', '   ')

    ## if out is None:
    ##     return out_

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


class SimpleStatBaseSearchlight(BaseSearchlight):
    """TODO

    TODO s

    - some stats are not needed (eg per sample X^2's) for MNN, so we
      should make them optional depending on the derived class

    Note: refactored from the original GNBSearchlight

    """

    _ATTRIBUTE_COLLECTIONS = ['params', 'ca']

    @borrowkwargs(BaseSearchlight, '__init__')
    def __init__(self, generator, qe, errorfx=mean_mismatch_error,
                 indexsum=None, **kwargs):
        """Initialize a GNBSearchlight

        Parameters
        ----------
        generator : `Generator`
          Some `Generator` to prepare partitions for cross-validation.
        qe : `QueryEngine`
          Query engine which would provide neighborhood information
        errorfx : func, optional
          Functor that computes a scalar error value from the vectors of
          desired and predicted values (e.g. subclass of `ErrorFunction`).
        indexsum : ('sparse', 'fancy'), optional
          What use to compute sums over arbitrary columns.  'fancy'
          corresponds to regular fancy indexing over columns, whenever
          in 'sparse', product of sparse matrices is used (usually
          faster, so is default if `scipy` is available).
        """

        # init base class first
        BaseSearchlight.__init__(self, qe, **kwargs)

        self._errorfx = errorfx
        self._generator = generator

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

    def __repr__(self, prefixes=[]):
        return super(SimpleStatBaseSearchlight, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['generator'])
            + _repr_attrs(self, ['errorfx'], default=mean_mismatch_error)
            + _repr_attrs(self, ['indexsum'])
            )

    def _get_space(self):
        raise NotImplementedError("Must be implemented in derived classes")

    def _sl_call(self, dataset, roi_ids, nproc):
        """Call to SimpleStatBaseSearchlight
        """
        # Local bindings
        generator = self.generator
        qe = self.queryengine
        errorfx = self.errorfx

        ## if False:
        ##     class A(Learner):
        ##         pass
        ##     self = A()
        ##     import numpy as np
        ##     from mvpa2.clfs.gnb import GNB
        ##     from mvpa2.generators.partition import NFoldPartitioner
        ##     from mvpa2.misc.errorfx import mean_mismatch_error
        ##     from mvpa2.testing.datasets import datasets as tdatasets
        ##     from mvpa2.datasets import Dataset
        ##     from mvpa2.misc.neighborhood import IndexQueryEngine, Sphere
        ##     from mvpa2.clfs.distance import absmin_distance
        ##     import time
        ##     if __debug__:
        ##         from mvpa2.base import debug
        ##         debug.active += ['SLC.*']
        ##         # XXX is it that ugly?
        ##         debug.active.pop(debug.active.index('SLC_'))
        ##         debug.metrics += ['reltime']
        ##     dataset = tdatasets['3dlarge'].copy()
        ##     dataset.fa['voxel_indices'] = dataset.fa.myspace
        ##     sphere = Sphere(radius=1,
        ##                     distance_func=absmin_distance)
        ##     qe = IndexQueryEngine(myspace=sphere)

        ##     # Fracisco's data
        ##     #dataset = ds_fp
        ##     qe = IndexQueryEngine(voxel_indices=sphere)

        ##     qe.train(dataset)
        ##     roi_ids = np.arange(dataset.nfeatures)
        ##     gnb = GNB()
        ##     params = gnb.params
        ##     generator = NFoldPartitioner()
        ##     errorfx = mean_mismatch_error

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
        ulabels_numeric = [label2index[l] for l in ulabels]
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
        splits = list(tuple(splitter.generate(ds_))
                      for ds_ in generator.generate(dataset_indicies))
        nsplits = len(splits)

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
        sample2block = np.array([description2block[d] for d in descriptions])

        # 3. Compute statistics per each block
        #
        if __debug__:
            debug('SLC',
                  'Phase 3. Computing statistics for %i blocks' % (nblocks,))

        #
        # reusable containers which should stay of the same size
        #

        # sums and sums of squares per each block
        sums = np.zeros((nblocks, ) + s_shape)
        # sums of squares
        sums2 = np.zeros((nblocks, ) + s_shape)

        # per each label:
        means = np.zeros((nlabels, ) + s_shape)
        # means of squares for stddev computation
        means2 = np.zeros((nlabels, ) + s_shape)
        variances = np.zeros((nlabels, ) + s_shape)
        # degenerate dimension are added for easy broadcasting later on
        nsamples_per_class = np.zeros((nlabels,) + (1,)*len(s_shape))

        # results
        results = np.zeros((nsplits,) + r_shape)

        block_counts = np.zeros((nblocks,))
        block_labels = [None] * nblocks

        X2 = np.square(X)
        # silly way for now
        for l, s, s2, ib in zip(labels_numeric, X, X2, sample2block):
            sums[ib] += s
            sums2[ib] += s2
            block_counts[ib] += 1
            if block_labels[ib] is None:
                block_labels[ib] = l
            else:
                assert(block_labels[ib] == l)
        block_labels = np.asanyarray(block_labels)
        # additional silly tests for paranoid
        assert(block_labels.dtype.kind is 'i')

        # 4. Lets deduce all neighbors... might need to be RF into the
        #    parallel part later on
        if __debug__:
            debug('SLC',
                  'Phase 4. Deducing neighbors information for %i ROIs'
                  % (nrois,))
        roi_fids = [qe.query_byid(f) for f in roi_ids]
        nroi_fids = len(roi_fids)
        # makes sense to waste precious ms only if ca is enabled
        if self.ca.is_enabled('roi_sizes'):
            roi_sizes = [len(x) for x in roi_fids]
        else:
            roi_sizes = []

        indexsum = self._indexsum
        if indexsum == 'sparse':
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
            # convert to blocks training split
            training_bis = np.unique(sample2block[training_sis])


            # Let's collect stats summaries
            training_nsamples = 0
            for il, l in enumerate(ulabels_numeric):
                bis_il = training_bis[block_labels[training_bis] == l]
                nsamples_per_class[il] = N_float = \
                                         float(np.sum(block_counts[bis_il]))
                training_nsamples += N_float
                if N_float == 0.0:
                    variances[il] = means[il] = means2[il] = 0.
                else:
                    means[il] = np.sum(sums[bis_il], axis=0) / N_float
                    # Not yet normed
                    means2[il] = np.sum(sums2[bis_il], axis=0)

            ## Actually compute the non-0 variances
            non0labels = (nsamples_per_class.squeeze() != 0)
            if np.all(non0labels):
                # For a possible tiny speed up avoiding copying and
                # using (no) slicing
                non0labels = slice(None)

            # That is the GNB specificity
            predictions = self._sl_call_on_a_split(
                split, X, X2,           # X2 might light to go
                nsamples_per_class,
                training_nsamples,      # GO? == np.sum(nsamples_per_class)
                non0labels,
                means, means2, variances,
                nroi_fids, roi_fids,    # passing nroi_fids as well since in 'sparse' way it has no 'length'
                indexsum_fx,
                )

            # assess the errors
            if __debug__:
                debug('SLC', "  Assessing accuracies")

            targets = labels_numeric[split[1].samples[:, 0]]
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

        return Dataset(results), roi_sizes

    generator = property(fget=lambda self: self._generator)
    errorfx = property(fget=lambda self: self._errorfx)
    indexsum = property(fget=lambda self: self._indexsum)
