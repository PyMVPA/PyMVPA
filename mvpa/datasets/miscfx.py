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
import random

import numpy as N

from mvpa.base.dataset import datasetmethod
from mvpa.datasets.base import Dataset
from mvpa.base.dochelpers import table2string
from mvpa.misc.support import get_break_points

from mvpa.base import externals, warning

if __debug__:
    from mvpa.base import debug


@datasetmethod
##REF: Name was automagically refactored
def aggregate_features(dataset, fx=N.mean):
    """Apply a function to each row of the samples matrix of a dataset.

    The functor given as `fx` has to honour an `axis` keyword argument in the
    way that NumPy used it (e.g. NumPy.mean, var).

    Returns
    -------
     a new `Dataset` object with the aggregated feature(s).
    """
    agg = fx(dataset.samples, axis=1)

    return Dataset(samples=N.array(agg, ndmin=2).T, sa=dataset.sa)


@datasetmethod
##REF: Name was automagically refactored
def remove_invariant_features(dataset):
    """Returns a new dataset with all invariant features removed.
    """
    return dataset[:, dataset.samples.std(axis=0).nonzero()[0]]


@datasetmethod
##REF: Name was automagically refactored
def coarsen_chunks(source, nchunks=4):
    """Change chunking of the dataset

    Group chunks into groups to match desired number of chunks. Makes
    sense if originally there were no strong groupping into chunks or
    each sample was independent, thus belonged to its own chunk

    Parameters
    ----------
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

    # we always want an array!
    chunks_new = N.array([chunks_map[x] for x in chunks])

    if __debug__:
        debug("DS_", "Using dictionary %s to remap old chunks %s into new %s"
              % (chunks_map, chunks, chunks_new))

    if isinstance(source, Dataset):
        if __debug__:
            debug("DS", "Coarsing %d chunks into %d chunks for %s"
                  %(nchunks_orig, len(chunks_new), source))
        source.sa['chunks'].value = chunks_new
        return
    else:
        return chunks_new


@datasetmethod
##REF: Name was automagically refactored
def get_samples_per_chunk_label(dataset):
    """Returns an array with the number of samples per label in each chunk.

    Array shape is (chunks x targets).

    Parameters
    ----------
    dataset : Dataset
      Source dataset.
    """
    ul = dataset.sa['targets'].unique
    uc = dataset.sa['chunks'].unique

    count = N.zeros((len(uc), len(ul)), dtype='uint')

    for cc, c in enumerate(uc):
        for lc, l in enumerate(ul):
            count[cc, lc] = N.sum(N.logical_and(dataset.targets == l,
                                                dataset.chunks == c))

    return count


@datasetmethod
def permute_targets(dataset, perchunk=True, assure_permute=False):
    """Permute the targets of a Dataset.

    A new permuted set of targets is assigned to the dataset, replacing
    the previous targets. The targets are not modified in-place, hence
    it is safe to call the function on shallow copies of a dataset
    without modifying the original dataset's targets.

    Parameters
    ----------
    perchunk : bool
      If True, permutation is limited to samples sharing the same chunk
      value.  Therefore only the association of a certain sample with
      a label is permuted while keeping the absolute number of
      occurences of each label value within a certain chunk constant.
      If there is no `chunks` information in the dataset this flag has
      no effect.
    assure_permute : bool
      If True, assures that targets are permutted, i.e. any one is
      different from the original one
    """
    if __debug__:
        if len(N.unique(dataset.sa.targets)) < 2:
            raise RuntimeError(
                  "Permuting targets is only meaningful if there are "
                  "more than two different targets.")

    # local binding
    targets = dataset.sa['targets'].value

    # now scramble
    if perchunk and dataset.sa.has_key('chunks'):
        chunks = dataset.sa['chunks'].value

        ptargets = N.zeros(targets.shape, dtype=targets.dtype)

        for o in dataset.sa['chunks'].unique:
            ptargets[chunks == o] = \
                N.random.permutation(targets[chunks == o])
    else:
        ptargets = N.random.permutation(targets)

    if assure_permute:
        if not (ptargets != targets).any():
            if not (assure_permute is True):
                if assure_permute == 1:
                    raise RuntimeError, \
                          "Cannot assure permutation of targets %s for " \
                          "some reason with chunks %s and while " \
                          "perchunk=%s . Should not happen" % \
                          (targets, self.chunks, perchunk)
            else:
                assure_permute = 11 # make 10 attempts
            if __debug__:
                debug("DS",  "Recalling permute to assure different targets")
            permute_targets(dataset,
                           perchunk=perchunk,
                           assure_permute=assure_permute-1)
    # reassign to the dataset
    dataset.sa.targets = ptargets


@datasetmethod
def random_samples(dataset, nperlabel):
    """Create a dataset with a random subset of samples.

    Parameters
    ----------
    nperlabel : int, list

      If an integer is given, the specified number of samples is randomly
      choosen from the group of samples sharing a unique label value (total
      number of selected samples: nperlabel x len(uniquetargets).

      If a list is given which's length is matching the unique label values, it
      will specify the number of samples chosen for each particular unique
      label.

    Returns
    -------
    A dataset instance for the chosen samples. All feature attributes and
    dataset attribute share there data with the source dataset.
    """
    uniquetargets = dataset.sa['targets'].unique
    # if interger is given take this value for all classes
    if isinstance(nperlabel, int):
        nperlabel = [nperlabel for i in uniquetargets]

    sample = []
    # for each available class
    targets = dataset.targets
    for i, r in enumerate(uniquetargets):
        # get the list of pattern ids for this class
        sample += random.sample((targets == r).nonzero()[0], nperlabel[i] )

    return dataset[sample]


@datasetmethod
def get_nsamples_per_attr(dataset, attr):
    """Returns the number of samples per unique value of a sample attribute.

    Parameters
    ----------
    attr : str
      Name of the sample attribute

    Returns
    -------
    dict with the number of samples (value) per unique attribute (key).
    """
    uniqueattr = dataset.sa[attr].unique

    # use dictionary to cope with arbitrary targets
    result = dict(zip(uniqueattr, [ 0 ] * len(uniqueattr)))
    for l in dataset.sa[attr].value:
        result[l] += 1

    return result


@datasetmethod
def get_samples_by_attr(dataset, attr, values, sort=True):
    """Return indices of samples given a list of attributes
    """

    if not isSequenceType(values) \
           or isinstance(values, basestring):
        values = [ values ]

    # TODO: compare to plain for loop through the targets
    #       on a real data example
    sel = N.array([], dtype=N.int16)
    sa = dataset.sa
    for value in values:
        sel = N.concatenate((
            sel, N.where(sa[attr].value == value)[0]))

    if sort:
        # place samples in the right order
        sel.sort()

    return sel


class SequenceStats(dict):
    """Simple helper to provide representation of sequence statistics

    Matlab analog:
    http://cfn.upenn.edu/aguirre/code/matlablib/mseq/mtest.m

    WARNING: Experimental -- API might change without warning!
    Current implementation is ugly!
    """

    def __init__(self, seq, order=2):#, chunks=None, perchunk=False):
        """Initialize SequenceStats

        Parameters
        ----------
        seq : list or ndarray
          Actual sequence of labels
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
        self._compute()


    def __repr__(self):
        """Representation of SequenceStats
        """
        return "SequenceStats(%s, order=%d)" % (repr(self._seq), self.order)

    def __str__(self):
        return self._str_stats

    def _compute(self):
        """Compute stats and string representation
        """
        # Do actual computation
        order = self.order
        seq = list(self._seq)               # assure list
        nsamples = len(seq)                 # # of samples/labels
        ulabels = sorted(list(set(seq)))    # unique labels
        nlabels = len(ulabels)              # # of labels

        # mapping for labels
        labels_map = dict([(l, i) for i, l in enumerate(ulabels)])

        # map sequence first
        seqm = [labels_map[i] for i in seq]
        # nperlabel = N.bincount(seqm)

        res = dict(ulabels=ulabels)
        # Estimate counter-balance
        cbcounts = N.zeros((order, nlabels, nlabels), dtype=int)
        for cb in xrange(order):
            for i, j in zip(seqm[:-(cb+1)], seqm[cb+1:]):
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
