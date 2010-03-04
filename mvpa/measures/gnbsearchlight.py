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

from numpy import ones, zeros, sum, abs, isfinite, dot
from mvpa.base import warning, externals
from mvpa.clfs.gnb import GNB

from mvpa.misc.errorfx import MeanMismatchErrorFx
from mvpa.measures.searchlight import BaseSearchlight

from mvpa.misc.param import Parameter
from mvpa.misc.state import ConditionalAttribute
#from mvpa.measures.base import Sensitivity


if __debug__:
    from mvpa.base import debug

__all__ = [ "GNB" ]

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

        if not self._nproc in ('auto', 1):
            raise NotImplementedError, "For now only nproc=1 (or 'auto') " \
                  "is supported by GNBSearchlight"


    def _sl_call(self, dataset, roi_ids, nproc):
        """Call to GNBSearchlight
        """
        # Local bindings
        params = self._gnb.params
        splitter = self._splitter
        errorfx = self._errorfx
        qe = self._qe

        if True:
            class A(object):
                pass
            self = A()
            from mvpa.clfs.gnb import GNB
            #from mvpa.datasets.splitters import NFoldSplitter
            from mvpa.misc.errorfx import MeanMismatchErrorFx
            from mvpa.testing.datasets import datasets
            from mvpa.datasets import Dataset
            qe = IndexQueryEngine(myspace=Sphere(radius=1))
            dataset = datasets['3dlarge']
            qe.train(dataset)
            roi_ids = np.arange(dataset.nfeatures)
            gnb = GNB()
            params = gnb.params
            splitter = NFoldSplitter()
            errorfx = MeanMismatchErrorFx()

        targets_sa_name = params.targets_attr
        targets_sa = dataset.sa[targets_sa_name]

        # get the dataset information into easy vars
        X = dataset.samples
        labels = targets_sa.value
        self.ulabels = ulabels = targets_sa.unique
        nlabels = len(ulabels)
        label2index = dict((l, il) for il, l in enumerate(ulabels))
        labels_numeric = [label2index[l] for l in labels]
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
        # what samples participate in each block...
        block2samples = []           # TODO???
        # ??? also label per each block???

        # 3. Compute statistics per each block
        #
        if __debug__:
            debug('SLC',
                  'Phase 3. Computing statistics for %i blocks' % (nblocks,))
        # number of samples in each block
        block_counts_test = np.histogram(sample2block, bins=np.arange(nblocks+1))[0]

        # compute sums and sums of squares per each block
        sums = np.zeros((nblocks, ) + s_shape)
        # sums of squares
        sums2 = np.zeros((nblocks, ) + s_shape)
        # variables which will be used later on
        means = np.zeros((nlabels, ) + s_shape)
        # means of squares for stddev computation
        means2 = np.zeros((nlabels, ) + s_shape)
        variances = np.zeros((nlabels, ) + s_shape)

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
        # yet another silly test for paranoid
        assert((block_counts == block_counts_test).all())

        # 4. Lets deduce all neighbors... might need to be RF into the
        #    parallel part later on
        nrois = len(roi_ids)
        if __debug__:
            debug('SLC',
                  'Phase 4. Deducing neighbors information for %i ROIs'
                  % (nrois,))
        roi_fids = [qe[f] for f in roi_ids]

        # 5. Lets do actual "splitting" and "classification"
        if __debug__:
            debug('SLC', 'Phase 5. Major loop' )

        # reusable containers which should stay of the same size

        # degenerate dimension are added for easy broadcasting later on
        nsamples_per_class = np.zeros((nlabels,) + (1,)*len(s_shape))

        for isplit, split in enumerate(splits):
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
                nsamples_per_class[il] = N_float = float(np.sum(block_counts[bis_il]))
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
                    np.sum(means2 - nsamples_per_class*np.squared(means), axis=0) \
                    / training_nsamples
            else:
                variances[non0labels] = \
                    (means2 - nsamples_per_class*np.squared(means))[non0labels] \
                    / nsamples_per_class[non0labels]

            # assign priors
            priors = gnb._get_priors(nlabels, training_nsamples, nsamples_per_class)

            if __debug__ and 'GNB' in debug.active:
                debug('SLC', "'Training' is done")


    def untrain(self):
        """Untrain classifier and reset all learnt params
        """
        self.means = None
        self.variances = None
        self.ulabels = None
        self.priors = None
        super(GNB, self).untrain()


    @accepts_dataset_as_samples
    def _predict(self, data):
        """Predict the output for the provided data.
        """
        params = self.params
        # argument of exponentiation
        scaled_distances = \
            -0.5 * (((data - self.means[:, np.newaxis, ...])**2) \
                          / self.variances[:, np.newaxis, ...])
        if params.logprob:
            # if self.params.common_variance:
            # XXX YOH:
            # For decision there is no need to actually compute
            # properly scaled p, ie 1/sqrt(2pi * sigma_i) could be
            # simply discarded since it is common across features AND
            # classes
            # For completeness -- computing everything now even in logprob
            lprob_csfs = self._norm_weight[:, np.newaxis, ...] + scaled_distances

            # XXX for now just cut/paste with different operators, but
            #     could just bind them and reuse in the same equations
            # Naive part -- just a product of probabilities across features
            ## First we need to reshape to get class x samples x features
            lprob_csf = lprob_csfs.reshape(
                lprob_csfs.shape[:2] + (-1,))
            ## Now -- sum across features
            lprob_cs = lprob_csf.sum(axis=2)

            # Incorporate class probabilities:
            prob_cs_cp = lprob_cs + np.log(self.priors[:, np.newaxis])

        else:
            # Just a regular Normal distribution with per
            # feature/class mean and variances
            prob_csfs = \
                 self._norm_weight[:, np.newaxis, ...] * np.exp(scaled_distances)

            # Naive part -- just a product of probabilities across features
            ## First we need to reshape to get class x samples x features
            prob_csf = prob_csfs.reshape(
                prob_csfs.shape[:2] + (-1,))
            ## Now -- product across features
            prob_cs = prob_csf.prod(axis=2)

            # Incorporate class probabilities:
            prob_cs_cp = prob_cs * self.priors[:, np.newaxis]

        # Normalize by evidence P(data)
        if params.normalize:
            if params.logprob:
                prob_cs_cp_real = np.exp(prob_cs_cp)
            else:
                prob_cs_cp_real = prob_cs_cp
            prob_s_cp_marginals = np.sum(prob_cs_cp_real, axis=0)
            if params.logprob:
                prob_cs_cp -= np.log(prob_s_cp_marginals)
            else:
                prob_cs_cp /= prob_s_cp_marginals

        # Take the class with maximal (log)probability
        winners = prob_cs_cp.argmax(axis=0)
        predictions = [self.ulabels[c] for c in winners]

        # set to the probabilities per class
        self.ca.estimates = prob_cs_cp.T

        if __debug__ and 'GNB' in debug.active:
            debug('GNB', "predict on data.shape=%s min:max(data)=%f:%f " %
                  (data.shape, np.min(data), np.max(data)))

        return predictions

# Lazy way to reuse params definitions from GNB
GNBSearchlight._collections_template['params'].update(
    GNB._collections_template['params'])
GNBSearchlight._paramsdoc += GNB._paramsdoc
