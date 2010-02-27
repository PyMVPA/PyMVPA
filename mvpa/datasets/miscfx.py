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

import numpy as np

from mvpa.base.dataset import datasetmethod
from mvpa.datasets.base import Dataset
from mvpa.base.dochelpers import table2string
from mvpa.misc.support import get_break_points
from mvpa.misc.support import idhash as idhash_

from mvpa.base import externals, warning

if __debug__:
    from mvpa.base import debug


@datasetmethod
##REF: Name was automagically refactored
def aggregate_features(dataset, fx=np.mean):
    """Apply a function to each row of the samples matrix of a dataset.

    The functor given as `fx` has to honour an `axis` keyword argument in the
    way that NumPy used it (e.g. NumPy.mean, var).

    Returns
    -------
     a new `Dataset` object with the aggregated feature(s).
    """
    agg = fx(dataset.samples, axis=1)

    return Dataset(samples=np.array(agg, ndmin=2).T, sa=dataset.sa)


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
    chunks_unique = np.unique(chunks)
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
    avg_chunk_size = np.sum(counts.values())*1.0/nchunks
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
    chunks_new = np.array([chunks_map[x] for x in chunks])

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

    count = np.zeros((len(uc), len(ul)), dtype='uint')

    for cc, c in enumerate(uc):
        for lc, l in enumerate(ul):
            count[cc, lc] = np.sum(np.logical_and(dataset.targets == l,
                                                dataset.chunks == c))

    return count


@datasetmethod
def permute_targets(dataset, targets_attr='targets', chunks_attr='chunks',
                    assure_permute=False):
    """Permute the targets of a Dataset.

    A new permuted set of targets is assigned to the dataset, replacing
    the previous targets. The targets are not modified in-place, hence
    it is safe to call the function on shallow copies of a dataset
    without modifying the original dataset's targets.

    Parameters
    ----------
    targets_attr : string, optional
      Name of the samples attribute which is used as ``targets``
      (i.e. gets permuted).
    chunks_attr : None or string
      If a string given, permutation is limited to samples sharing the
      same value of the `chunks_attr` samples attribute.  Therefore,
      only the association of a certain sample with a label is
      permuted while keeping the absolute number of occurrences of each
      label value within a certain chunk constant.
    assure_permute : bool, optional
      If True, assures that targets are permuted, i.e. any one is
      different from the original one

    Returns
    -------
    Dataset
       shallow copy of original dataset with permuted labels.
    """
    if __debug__:
        if len(dataset.sa[targets_attr].unique) < 2:
            raise RuntimeError(
                  "Permuting targets is only meaningful if there are "
                  "more than two different values of targets.")

    # local binding
    targets = dataset.sa[targets_attr].value

    # now scramble
    if chunks_attr:
        if chunks_attr in dataset.sa:
            chunks = dataset.sa[chunks_attr].value

            ptargets = np.zeros(targets.shape, dtype=targets.dtype)

            for o in dataset.sa[chunks_attr].unique:
                ptargets[chunks == o] = \
                    np.random.permutation(targets[chunks == o])
        else:
            raise ValueError, \
                  "There is no sa named %r in %s, thus no permutation is " \
                  "possible" % (chunks_attr, dataset)
    else:
        ptargets = np.random.permutation(targets)

    if assure_permute:
        if not (ptargets != targets).any():
            if not (assure_permute is True):
                if assure_permute == 1:
                    raise RuntimeError, \
                          "Cannot assure permutation of targets %s for " \
                          "some reason for dataset %s and chunks_attr=%r. " \
                          "Should not happen" % \
                          (targets, dataset, chunks_attr)
            else:
                assure_permute = 11 # make 10 attempts
            if __debug__:
                debug("DS",  "Recalling permute to assure different targets")
            permute_targets(dataset,
                            targets_attr=targets_attr,
                            chunks_attr=chunks_attr,
                            assure_permute=assure_permute-1)

    # reassign to the dataset
    dataset.sa[targets_attr].value = ptargets


@datasetmethod
def random_samples(dataset, nperlabel):
    """Create a dataset with a random subset of samples.

    Parameters
    ----------
    dataset : Dataset
    nperlabel : int or list
      If an `int` is given, the specified number of samples is randomly
      chosen from the group of samples sharing a unique label value. Total
      number of selected samples: nperlabel x len(uniquetargets).
      If a `list` is given of length matching the unique label values, it
      specifies the number of samples chosen for each particular unique
      label.

    Returns
    -------
    Dataset
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
    sel = np.array([], dtype=np.int16)
    sa = dataset.sa
    for value in values:
        sel = np.concatenate((
            sel, np.where(sa[attr].value == value)[0]))

    if sort:
        # place samples in the right order
        sel.sort()

    return sel

@datasetmethod
def summary(dataset, stats=True, lstats=True, idhash=False,
            maxc=30, maxt=20):
    """String summary over the object

    Parameters
    ----------
    stats : bool
     Include some basic statistics (mean, std, var) over dataset samples
    lstats : bool
     Include statistics on chunks/targets
    idhash : bool
     Include idhash value for dataset and samples
    maxt : int
      Maximal number of targets when provide details on targets/chunks
    maxc : int
      Maximal number of chunks when provide details on targets/chunks
    """
    # local bindings
    samples = dataset.samples
    s = str(dataset)[1:-1]

    if idhash:
        s += '\nID-Hashes: %s' % dataset.idhash

    ssep = (' ', '\n')[lstats]

    ## Possibly summarize attributes listed as having unique
    if stats:
        # TODO -- avg per chunk?
        # XXX We might like to use scipy.stats.describe to get
        # quick summary statistics (mean/range/skewness/kurtosis)
        if dataset.nfeatures:
            s += "%sstats: mean=%g std=%g var=%g min=%g max=%g\n" % \
                 (ssep, np.mean(samples), np.std(samples),
                  np.var(samples), np.min(samples), np.max(samples))
        else:
            s += "%sstats: dataset has no features\n" % ssep

    if lstats:
        s += dataset.summary_targets(maxc=maxc, maxt=maxt)

    return s

@datasetmethod
def summary_targets(dataset, maxc=30, maxt=20):
    """Provide summary statistics over the targets and chunks

    Parameters
    ----------
    maxc : int
      Maximal number of chunks when provide details
    maxt : int
      Maximal number of targets when provide details
    """
    # We better avoid bound function since if people only
    # imported Dataset without miscfx it would fail
    spcl = get_samples_per_chunk_target(dataset)
    # XXX couldn't they be unordered?
    ul = dataset.sa['targets'].unique.tolist()
    uc = dataset.sa['chunks'].unique.tolist()
    s = ""
    if len(ul) < maxt and len(uc) < maxc:
        s += "\nCounts of targets in each chunk:"
        # only in a resonable case do printing
        table = [['  chunks\\targets'] + ul]
        table += [[''] + ['---'] * len(ul)]
        for c, counts in zip(uc, spcl):
            table.append([ str(c) ] + counts.tolist())
        s += '\n' + table2string(table)
    else:
        s += "No details due to large number of targets or chunks. " \
             "Increase maxc and maxt if desired"


    def cl_stats(axis, u, name1, name2):
        """Compute statistics per target
        """
        stats = {'min': np.min(spcl, axis=axis),
                 'max': np.max(spcl, axis=axis),
                 'mean': np.mean(spcl, axis=axis),
                 'std': np.std(spcl, axis=axis),
                 '#%ss' % name2: np.sum(spcl>0, axis=axis)}
        entries = ['  ' + name1, 'mean', 'std', 'min', 'max', '#%ss' % name2]
        table = [ entries ]
        for i, l in enumerate(u):
            d = {'  ' + name1 : l}
            d.update(dict([ (k, stats[k][i]) for k in stats.keys()]))
            table.append( [ ('%.3g', '%s')[isinstance(d[e], basestring)]
                            % d[e] for e in entries] )
        return '\nSummary per %s across %ss\n' % (name1, name2) \
               + table2string(table)

    if len(ul) < maxt:
        s += cl_stats(0, ul, 'target', 'chunk')
    if len(uc) < maxc:
        s += cl_stats(1, uc, 'chunk', 'target')
    return s


class SequenceStats(dict):
    """Simple helper to provide representation of sequence statistics

    Matlab analog:
    http://cfn.upenn.edu/aguirre/code/matlablib/mseq/mtest.m

    WARNING: Experimental -- API might change without warning!
    Current implementation is ugly!
    """

    def __init__(self, seq, order=2):#, chunks=None, chunks_attr=None):
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
        # nperlabel = np.bincount(seqm)

        res = dict(ulabels=ulabels)
        # Estimate counter-balance
        cbcounts = np.zeros((order, nlabels, nlabels), dtype=int)
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
            corr += [np.corrcoef(seqm, shifted)[0, 1]]
            # ??? report high (anti)correlations?
        res['corrcoef'] = corr = np.array(corr)
        res['sumabscorr'] = sumabscorr = np.sum(np.abs(corr))
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
            ind = np.where(~np.isnan(m))
            for i, j in zip(*ind):
                t[1+i][1+cb*(nlabels+1)+j] = '%d' % m[i, j]

        sout = "Original sequence had %d entries from set %s\n" \
               % (len(seq), ulabels) + \
               "Counter-balance table for orders up to %d:\n" % order \
               + table2string(t)
        sout += "Correlations: min=%.2g max=%.2g mean=%.2g sum(abs)=%.2g" \
                % (min(corr), max(corr), np.mean(corr), sumabscorr)
        self._str_stats = sout


    def plot(self):
        """Plot correlation coefficients
        """
        externals.exists('pylab', raise_=True)
        import pylab as pl
        pl.plot(self['corrcoef'])
        pl.title('Auto-correlation of the sequence')
        pl.xlabel('Offset')
        pl.ylabel('Correlation Coefficient')
        pl.show()
