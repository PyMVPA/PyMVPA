# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Misc function performing operations on datasets.

All the functions defined in this module must accept dataset as the
first argument since they are bound to Dataset class in the trailer.
"""

__docformat__ = 'restructuredtext'

from operator import isSequenceType

import numpy as N

from mvpa.datasets.base import Dataset, datasetmethod
from mvpa.base.dochelpers import table2string
from mvpa.misc.support import getBreakPoints

from mvpa.base import externals, warning

if __debug__:
    from mvpa.base import debug

if externals.exists('scipy'):
    from mvpa.datasets.miscfx_sp import detrend


@datasetmethod
def zscore(dataset, mean=None, std=None,
           perchunk=True, baselinelabels=None,
           pervoxel=True, targetdtype='float64'):
    """Z-Score the samples of a `Dataset` (in-place).

    `mean` and `std` can be used to pass custom values to the z-scoring.
    Both may be scalars or arrays.

    All computations are done *in place*. Data upcasting is done
    automatically if necessary into `targetdtype`

    If `baselinelabels` provided, and `mean` or `std` aren't provided, it would
    compute the corresponding measure based only on labels in `baselinelabels`

    If `perchunk` is True samples within the same chunk are z-scored independent
    of samples from other chunks, e.i. mean and standard deviation are
    calculated individually.
    """

    if __debug__ and perchunk \
      and N.array(dataset.samplesperchunk.values()).min() <= 2:
        warning("Z-scoring chunk-wise and one chunk with less than three "
                "samples will set features in these samples to either zero "
                "(with 1 sample in a chunk) "
                "or -1/+1 (with 2 samples in a chunk).")

    # cast to floating point datatype if necessary
    if str(dataset.samples.dtype).startswith('uint') \
       or str(dataset.samples.dtype).startswith('int'):
        dataset.setSamplesDType(targetdtype)

    def doit(samples, mean, std, statsamples=None):
        """Internal method."""

        if statsamples is None:
            # if nothing provided  -- mean/std on all samples
            statsamples = samples

        if pervoxel:
            axisarg = {'axis':0}
        else:
            axisarg = {}

        # calculate mean if necessary
        if mean is None:
            mean = statsamples.mean(**axisarg)

        # de-mean
        samples -= mean

        # calculate std-deviation if necessary
        # XXX YOH: would that be actually what we want?
        #          may be we want actually estimate of deviation from the mean,
        #          which per se might be not statsamples.mean (see above)?
        #          if logic to be changed -- adjust ZScoreMapper as well
        if std is None:
            std = statsamples.std(**axisarg)

        # do the z-scoring
        if pervoxel:
            # Assure std being an array
            if N.isscalar(std):
                std = N.ones(samples.shape[1])
            else:
                # and so we don't perform list comparison to 0
                std = N.asanyarray(std)
            samples[:, std != 0] /= std[std != 0]
        else:
            samples /= std

        return samples

    if baselinelabels is None:
        statids = None
    else:
        statids = set(dataset.idsbylabels(baselinelabels))

    # for the sake of speed yoh didn't simply create a list
    # [True]*dataset.nsamples to provide easy selection of everything
    if perchunk:
        for c in dataset.uniquechunks:
            slicer = N.where(dataset.chunks == c)[0]
            if not statids is None:
                statslicer = list(statids.intersection(set(slicer)))
                dataset.samples[slicer] = doit(dataset.samples[slicer],
                                               mean, std,
                                               dataset.samples[statslicer])
            else:
                slicedsamples = dataset.samples[slicer]
                dataset.samples[slicer] = doit(slicedsamples,
                                               mean, std,
                                               slicedsamples)
    elif statids is None:
        doit(dataset.samples, mean, std, dataset.samples)
    else:
        doit(dataset.samples, mean, std, dataset.samples[list(statids)])


@datasetmethod
def aggregateFeatures(dataset, fx=N.mean):
    """Apply a function to each row of the samples matrix of a dataset.

    The functor given as `fx` has to honour an `axis` keyword argument in the
    way that NumPy used it (e.g. NumPy.mean, var).

    :Returns:
       a new `Dataset` object with the aggregated feature(s).
    """
    agg = fx(dataset.samples, axis=1)

    return Dataset(samples=N.array(agg, ndmin=2).T,
                   labels=dataset.labels,
                   chunks=dataset.chunks)


@datasetmethod
def removeInvariantFeatures(dataset):
    """Returns a new dataset with all invariant features removed.
    """
    return dataset.selectFeatures(dataset.samples.std(axis=0).nonzero()[0])


@datasetmethod
def coarsenChunks(source, nchunks=4):
    """Change chunking of the dataset

    Group chunks into groups to match desired number of chunks. Makes
    sense if originally there were no strong groupping into chunks or
    each sample was independent, thus belonged to its own chunk

    :Parameters:
      source : Dataset or list of chunk ids
        dataset or list of chunk ids to operate on. If Dataset, then its chunks
        get modified
      nchunks : int
        desired number of chunks
    """

    if isinstance(source, Dataset):
        chunks = source.chunks
    else:
        chunks = source
    chunks_unique = N.unique(chunks)
    nchunks_orig = len(chunks_unique)

    if nchunks_orig < nchunks:
        raise ValueError, \
              "Original number of chunks is %d. Cannot coarse them " \
              "to get %d chunks" % (nchunks_orig, nchunks)

    # figure out number of samples per each chunk
    counts = dict(zip(chunks_unique, [ 0 ] * len(chunks_unique)))
    for c in chunks:
        counts[c] += 1

    # now we need to group chunks to get more or less equalized number
    # of samples per chunk. No sophistication is done -- just
    # consecutively group to get close to desired number of samples
    # per chunk
    avg_chunk_size = N.sum(counts.values())*1.0/nchunks
    chunks_groups = []
    cur_chunk = []
    nchunks = 0
    cur_chunk_nsamples = 0
    samples_counted = 0
    for i, c in enumerate(chunks_unique):
        cc = counts[c]

        cur_chunk += [c]
        cur_chunk_nsamples += cc

        # time to get a new chunk?
        if (samples_counted + cur_chunk_nsamples
            >= (nchunks+1)*avg_chunk_size) or i==nchunks_orig-1:
            chunks_groups.append(cur_chunk)
            samples_counted += cur_chunk_nsamples
            cur_chunk_nsamples = 0
            cur_chunk = []
            nchunks += 1

    if len(chunks_groups) != nchunks:
        warning("Apparently logic in coarseChunks is wrong. "
                "It was desired to get %d chunks, got %d"
                % (nchunks, len(chunks_groups)))

    # remap using groups
    # create dictionary
    chunks_map = {}
    for i, group in enumerate(chunks_groups):
        for c in group:
            chunks_map[c] = i

    chunks_new = [chunks_map[x] for x in chunks]

    if __debug__:
        debug("DS_", "Using dictionary %s to remap old chunks %s into new %s"
              % (chunks_map, chunks, chunks_new))

    if isinstance(source, Dataset):
        if __debug__:
            debug("DS", "Coarsing %d chunks into %d chunks for %s"
                  %(nchunks_orig, len(chunks_new), source))
        source.chunks = chunks_new
        return
    else:
        return chunks_new


@datasetmethod
def getSamplesPerChunkLabel(dataset):
    """Returns an array with the number of samples per label in each chunk.

    Array shape is (chunks x labels).

    :Parameters:
      dataset: Dataset
        Source dataset.
    """
    ul = dataset.uniquelabels
    uc = dataset.uniquechunks

    count = N.zeros((len(uc), len(ul)), dtype='uint')

    for cc, c in enumerate(uc):
        for lc, l in enumerate(ul):
            count[cc, lc] = N.sum(N.logical_and(dataset.labels == l,
                                                dataset.chunks == c))

    return count


class SequenceStats(dict):
    """Simple helper to provide representation of sequence statistics

    Matlab analog:
    http://cfn.upenn.edu/aguirre/code/matlablib/mseq/mtest.m

    WARNING: Experimental -- API might change without warning!
    Current implementation is ugly!
    """

    def __init__(self, seq, order=2):#, chunks=None, perchunk=False):
        """Initialize SequenceStats

        :Parameters:
          seq : list or ndarray
            Actual sequence of labels

        :Keywords:
          order : int
            Maximal order of counter-balancing check. For perfect
            counterbalancing all matrices should be identical
        """
        """
          chunks : None or list or ndarray
            Chunks to use if `perchunk`=True
          perchunk .... TODO
          """
        dict.__init__(self)
        self.order = order
        self._seq = seq
        self.stats = None
        self._str_stats = None
        self.__compute()


    def __repr__(self):
        """Representation of SequenceStats
        """
        return "SequenceStats(%s, order=%d)" % (repr(self._seq), self.order)

    def __str__(self):
        return self._str_stats

    def __compute(self):
        """Compute stats and string representation
        """
        # Do actual computation
        order = self.order
        seq = list(self._seq)               # assure list
        nsamples = len(seq)                 # # of samples/labels
        ulabels = sorted(list(set(seq)))    # unique labels
        nlabels = len(ulabels)              # # of labels

        # mapping for labels
        labels_map = dict([(l, i) for i,l in enumerate(ulabels)])

        # map sequence first
        seqm = [labels_map[i] for i in seq]
        nperlabel = N.bincount(seqm)

        res = dict(ulabels=ulabels)
        # Estimate counter-balance
        cbcounts = N.zeros((order, nlabels, nlabels), dtype=int)
        for cb in xrange(order):
            for i,j in zip(seqm[:-(cb+1)], seqm[cb+1:]):
                cbcounts[cb, i, j] += 1
        res['cbcounts'] = cbcounts

        """
        Lets compute relative counter-balancing
        Ideally, nperlabel[i]/nlabels should precede each label
        """
        # Autocorrelation
        corr = []
        # for all possible shifts:
        for shift in xrange(1, nsamples):
            shifted = seqm[shift:] + seqm[:shift]
            # ??? User pearsonsr with p may be?
            corr += [N.corrcoef(seqm, shifted)[0, 1]]
            # ??? report high (anti)correlations?
        res['corrcoef'] = corr = N.array(corr)
        res['sumabscorr'] = sumabscorr = N.sum(N.abs(corr))
        self.update(res)

        # Assign textual summary
        # XXX move into a helper function and do on demand
        t = [ [""] * (1 + self.order*(nlabels+1)) for i in xrange(nlabels+1) ]
        t[0][0] = "Labels/Order"
        for i, l  in enumerate(ulabels):
            t[i+1][0] = '%s:' % l
        for cb in xrange(order):
            t[0][1+cb*(nlabels+1)] = "O%d" % (cb+1)
            for i  in xrange(nlabels+1):
                t[i][(cb+1)*(nlabels+1)] = " | "
            m = cbcounts[cb]
            # ??? there should be better way to get indexes
            ind = N.where(~N.isnan(m))
            for i, j in zip(*ind):
                t[1+i][1+cb*(nlabels+1)+j] = '%d' % m[i, j]

        sout = "Original sequence had %d entries from set %s\n" \
               % (len(seq), ulabels) + \
               "Counter-balance table for orders up to %d:\n" % order \
               + table2string(t)
        sout += "Correlations: min=%.2g max=%.2g mean=%.2g sum(abs)=%.2g" \
                % (min(corr), max(corr), N.mean(corr), sumabscorr)
        self._str_stats = sout


    def plot(self):
        """Plot correlation coefficients
        """
        externals.exists('pylab', raiseException=True)
        import pylab as P
        P.plot(self['corrcoef'])
        P.title('Auto-correlation of the sequence')
        P.xlabel('Offset')
        P.ylabel('Correlation Coefficient')
        P.show()

