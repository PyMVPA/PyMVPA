# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Efficient implementation of searchlight for GNB

It takes advantage that :class:`~mvpa.clfs.gnb.GNB` is "naive" in its
reliance on massive univariate conditional probabilities of each feature
given a target class.  On the other side,
:class:`~mvpa.measures.searchlight.Searchlight` analysis approach
would ask for the same information over again and over again for the
same feature in multiple "lights".

Kudos for the idea and showing that it indeed might be beneficial over
generic Searchlight with GNB go to Francisco Pereira.
"""

__docformat__ = 'restructuredtext'

import numpy as np

#from numpy import ones, zeros, sum, abs, isfinite, dot
#from mvpa.base import warning, externals
from mvpa.datasets.base import Dataset
#from mvpa.clfs.gnb import GNB
from mvpa.misc.errorfx import MeanMismatchErrorFx
from mvpa.measures.searchlight import BaseSearchlight
from mvpa.base.dochelpers import borrowkwargs
#from mvpa.misc.param import Parameter
#from mvpa.misc.state import ConditionalAttribute
#from mvpa.measures.base import Sensitivity

from mvpa.misc.neighborhood import IndexQueryEngine, Sphere

if __debug__:
    from mvpa.base import debug
    import time as time

__all__ = [ "GNBSearchlight", 'sphere_gnbsearchlight' ]

class GNBSearchlight(BaseSearchlight):
    """Gaussian Naive Bayes `Searchlight`.

    See Also
    --------
    :class:`~mvpa.clfs.gnb.GNB`
    :class:`~mvpa.measures.searchlight.Searchlight`

    TODO
    """

    _ATTRIBUTE_COLLECTIONS = ['params', 'ca']

    @borrowkwargs(BaseSearchlight, '__init__')
    def __init__(self, gnb, splitter, errorfx=MeanMismatchErrorFx(),
                 *args, **kwargs):
        """Initialize a GNBSearchlight

        Parameters
        ----------
        gnb : `GNB`
          `GNB` classifier as the specification of what GNB parameters
          to use. Instance itself isn't used.
        splitter : `Splitter`
          `Splitter` to use to compute the error.
        errorfx: func, optional
          Functor that computes a scalar error value from the vectors of
          desired and predicted values (e.g. subclass of `ErrorFunction`)
        """

        # init base class first
        BaseSearchlight.__init__(self, *args, **kwargs)

        self._errorfx = errorfx
        self._splitter = splitter
        self._gnb = gnb

        if not self._nproc in (None, 1):
            raise NotImplementedError, "For now only nproc=1 (or None for " \
                  "autodetection) is supported by GNBSearchlight"


    def _sl_call(self, dataset, roi_ids, nproc):
        """Call to GNBSearchlight
        """
        # Local bindings
        gnb = self._gnb
        params = gnb.params
        splitter = self._splitter
        errorfx = self._errorfx
        qe = self._qe

        ## if False:
        ##     class A(object):
        ##         pass
        ##     self = A()
        ##     import numpy as np
        ##     from mvpa.clfs.gnb import GNB
        ##     from mvpa.datasets.splitters import NFoldSplitter
        ##     from mvpa.misc.errorfx import MeanMismatchErrorFx
        ##     #from mvpa.testing.datasets import datasets
        ##     from mvpa.datasets import Dataset
        ##     from mvpa.misc.neighborhood import IndexQueryEngine, Sphere
        ##     from mvpa.clfs.distance import absmin_distance
        ##     import time
        ##     if __debug__:
        ##         from mvpa.base import debug
        ##         debug.active += ['SLC.*']
        ##         # XXX is it that ugly?
        ##         debug.active.pop(debug.active.index('SLC_'))
        ##         debug.metrics += ['reltime']
        ##     dataset = datasets['3dlarge']
        ##     sphere = Sphere(radius=1,
        ##                     distance_func=absmin_distance)
        ##     qe = IndexQueryEngine(myspace=sphere)

        ##     # Fracisco's data
        ##     dataset = ds_fp
        ##     qe = IndexQueryEngine(voxel_indices=sphere)

        ##     qe.train(dataset)
        ##     roi_ids = np.arange(dataset.nfeatures)
        ##     gnb = GNB()
        ##     params = gnb.params
        ##     splitter = NFoldSplitter()
        ##     errorfx = MeanMismatchErrorFx()

        if __debug__:
            time_start = time.time()

        targets_sa_name = params.targets_attr
        targets_sa = dataset.sa[targets_sa_name]

        if __debug__:
            debug_slc_ = 'SLC_' in debug.active

        # get the dataset information into easy vars
        X = dataset.samples
        if len(X.shape) != 2:
            raise ValueError, \
                  'Unlike GNB, GNBSearchlight (for now) operates on already' \
                  'flattened datasets'
        labels = targets_sa.value
        ulabels = targets_sa.unique
        nlabels = len(ulabels)
        label2index = dict((l, il) for il, l in enumerate(ulabels))
        labels_numeric = np.array([label2index[l] for l in labels])
        ulabels_numeric = [label2index[l] for l in ulabels]
        # set the feature dimensions
        nsamples = len(X)
        s_shape = X.shape[1:]           # shape of a single sample

        #
        # Everything toward optimization ;)
        #
        # Silly Yarik thinks that it might be worth to pre-compute
        # statistics per each feature within a block of the samples
        # which always come together in splits -- most often it is a
        # (chunk, label) combination, but since we simply use a
        # splitter -- who knows! Therefore lets figure out what are
        # those blocks and operate on them instead of original samples.
        #
        # After additional thinking about this -- probably it would be
        # just minor additional improvements (ie not worth it) but
        # since it is coded already -- let it me so

        # 1. Query splitter for the splits we will have
        if __debug__:
            debug('SLC',
                  'Phase 1. Initializing splits using %s on %s'
                  % (splitter, dataset))
        # check the splitter -- splitcfg isn't sufficient
        # TODO: RF splitters so we could reliably obtain the configuration
        #       splitcfg just returns what to split into the other in terms
        #       of chunks... and we need actual indicies
        if splitter.permute:
            raise NotImplementedError, \
                  "Splitters which permute targets aren't supported here"
        # Lets just create a dummy ds which will store for us actual sample
        # indicies
        # XXX we could make it even more lightweight I guess...
        dataset_indicies = Dataset(np.arange(nsamples), sa=dataset.sa)
        splits = list(splitter(dataset_indicies))
        nsplits = len(splits)
        assert(len(splits[0]) == 2)     # assure that we have only 2
                                        # splits here for cvte

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
        for isplit, (split1, split2) in enumerate(splits):
            combinations[split1.samples[:, 0], 1+isplit] = 1
            combinations[split2.samples[:, 0], 1+isplit] = 2
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
        # number of samples in each block
        block_counts_test = np.histogram(sample2block,
                                         bins=np.arange(nblocks+1))[0]

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
        results = np.zeros((nsplits,) + s_shape)

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
        assert((block_counts == block_counts_test).all())

        # 4. Lets deduce all neighbors... might need to be RF into the
        #    parallel part later on
        nrois = len(roi_ids)
        if __debug__:
            debug('SLC',
                  'Phase 4. Deducing neighbors information for %i ROIs'
                  % (nrois,))
        roi_fids = [qe[f] for f in roi_ids]
        nroi_fids = len(roi_fids)

        # 5. Lets do actual "splitting" and "classification"
        if __debug__:
            debug('SLC', 'Phase 5. Major loop' )

        for isplit, split in enumerate(splits): # XXX
            if __debug__:
                debug('SLC', ' Split %i out of %i' % (isplit, nsplits))
            # figure out for a given splits the blocks we want to work
            # with
            # sample_indicies
            training_sis = split[0].samples[:, 0]
            # convert to blocks training split
            training_bis = np.unique(sample2block[training_sis])

            # now lets do our GNB business
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

            if params.common_variance:
                variances[:] = \
                    np.sum(means2 - nsamples_per_class*np.square(means),
                           axis=0) \
                    / training_nsamples
            else:
                variances[non0labels] = \
                    (means2 - nsamples_per_class*np.square(means))[non0labels] \
                    / nsamples_per_class[non0labels]

            # assign priors
            priors = gnb._get_priors(nlabels, training_nsamples, nsamples_per_class)

            # proceed in a way we have in GNB code with logprob=True,
            # i.e. operating within the exponents -- should lead to some
            # performance advantage
            norm_weight = -0.5 * np.log(2*np.pi*variances)
            # last added dimension would be for ROIs
            logpriors = np.log(priors[:, np.newaxis, np.newaxis])

            if __debug__:
                debug('SLC', "  'Training' is done")

            # Now it is time to "classify" our samples.
            # and for that we first need to compute corresponding
            # probabilities (or may be un
            data = X[split[1].samples[:, 0]]
            targets = labels_numeric[split[1].samples[:, 0]]

            # argument of exponentiation
            scaled_distances = \
                 -0.5 * (((data - means[:, np.newaxis, ...])**2) \
                         / variances[:, np.newaxis, ...])

            # incorporate the normalization from normals
            lprob_csfs = norm_weight[:, np.newaxis, ...] + scaled_distances

            ## First we need to reshape to get class x samples x features
            lprob_csf = lprob_csfs.reshape(lprob_csfs.shape[:2] + (-1,))

            ## Now we come to naive part which requires looping
            ## through all spheres
            ## TODO: check, that may be making use of sparse matrices
            ##       would give a benefit over a loop


            if __debug__:
                debug('SLC', "  Doing 'Searchlight'")
            # resultant logprobs for each class x sample x roi
            lprob_cs_sl = np.zeros(lprob_csfs.shape[:2] + (nroi_fids,))
            for iroi, roi_fids_ in enumerate(roi_fids):
                if __debug__ and debug_slc_:
                    debug('SLC_', "   Doing %i ROIs: %i (%i features) [%i%%]" \
                          % (nroi_fids,
                             iroi,
                             len(roi_fids_),
                             float(iroi+1)/nroi_fids*100,), cr=True)
                lprob_cs_sl[:, :, iroi] = lprob_csf[:, :, roi_fids_].sum(axis=2)

            # just a new line
            if __debug__ and debug_slc_:
                debug('SLC_', '   ')
            # XXX at some other point we might return back and start
            # worrying about unneeded memory consumption ;)
            #lprob_cs_cp_sl = lprob_cs_sl + logpriors
            # nah -- lets do right away
            lprob_cs_sl += logpriors
            lprob_cs_cp_sl = lprob_cs_sl
            # for each of the ROIs take the class with maximal (log)probability
            predictions = lprob_cs_cp_sl.argmax(axis=0)
            #predictions = winners # no need to map back [self.ulabels[c] for c in winners]
            # assess the errors
            if __debug__:
                debug('SLC', "  Assessing accuracies")

            # somewhat silly but a way which allows to use pre-crafted
            # error functions without a chance to screw up
            for i, fpredictions in enumerate(predictions.T):
                results[isplit, i] = errorfx(fpredictions, targets)

        if __debug__:
            debug('SLC', "GNBSearchlight is done in %.3g sec" %
                  (time.time() - time_start))

        # makes sense to waste precious ms only if ca is enabled
        if self.ca.is_enabled('roi_sizes'):
            roi_sizes = [len(x) for x in roi_fids]
        else:
            roi_sizes = []
        return Dataset(results), roi_sizes


@borrowkwargs(GNBSearchlight, '__init__', exclude=['roi_ids'])
def sphere_gnbsearchlight(gnb, splitter, radius=1, center_ids=None,
                          space='voxel_indices', *args, **kwargs):
    """Creates a `GNBSearchlight` to assess :term:`cross-validation`
    classification performance of GNB on all possible spheres of a
    certain size within a dataset.

    The idea of taking advantage of naiveness of GNB for the sake of
    quick searchlight-ing stems from Francisco Pereira (paper under
    review).

    Parameters
    ----------
    radius : float
      All features within this radius around the center will be part
      of a sphere.
    center_ids : list of int
      List of feature ids (not coordinates) the shall serve as sphere
      centers. By default all features will be used (it is passed
      roi_ids argument for Searchlight).
    space : str
      Name of a feature attribute of the input dataset that defines the spatial
      coordinates of all features.
    **kwargs
      In addition this class supports all keyword arguments of
      :class:`~mvpa.measures.gnbsearchlight.GNBSearchlight`.

    Notes
    -----
    If any `BaseSearchlight` is used as `SensitivityAnalyzer` one has to make
    sure that the specified scalar `DatasetMeasure` returns large
    (absolute) values for high sensitivities and small (absolute) values
    for low sensitivities. Especially when using error functions usually
    low values imply high performance and therefore high sensitivity.
    This would in turn result in sensitivity maps that have low
    (absolute) values indicating high sensitivities and this conflicts
    with the intended behavior of a `SensitivityAnalyzer`.
    """
    # build a matching query engine from the arguments
    kwa = {space: Sphere(radius)}
    qe = IndexQueryEngine(**kwa)
    # init the searchlight with the queryengine
    return GNBSearchlight(gnb, splitter, queryengine=qe, roi_ids=center_ids, *args, **kwargs)
