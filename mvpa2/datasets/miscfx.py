# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Miscellaneous functions to perform operations on datasets.

All the functions defined in this module must accept dataset as the
first argument since they are bound to Dataset class in the trailer.
"""

__docformat__ = 'restructuredtext'

import random

import numpy as np

from mvpa2.base.dataset import datasetmethod
from mvpa2.datasets.base import Dataset
from mvpa2.base.dochelpers import table2string
from mvpa2.misc.support import get_nelements_per_value

from mvpa2.base import externals, warning
from mvpa2.base.types import is_sequence_type

if __debug__:
    from mvpa2.base import debug


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
## TODO: make more efficient and more generic (accept >=1 attrs to
##       operate on)
def get_samples_per_chunk_target(dataset,
                                 targets_attr='targets', chunks_attr='chunks'):
    """Returns an array with the number of samples per target in each chunk.

    Array shape is (chunks x targets).

    Parameters
    ----------
    dataset : Dataset
      Source dataset.
    """
    # shortcuts/local bindings
    ta = dataset.sa[targets_attr]
    ca = dataset.sa[chunks_attr]

    # unique
    ut = ta.unique
    uc = ca.unique

    # all
    ts = ta.value
    cs = ca.value

    count = np.zeros((len(uc), len(ut)), dtype='uint')

    for ic, c in enumerate(uc):
        for it, t in enumerate(ut):
            count[ic, it] = np.sum(np.logical_and(ts==t, cs==c))

    return count


@datasetmethod
def random_samples(dataset, npertarget, targets_attr='targets'):
    """Create a dataset with a random subset of samples.

    Parameters
    ----------
    dataset : Dataset
    npertarget : int or list
      If an `int` is given, the specified number of samples is randomly
      chosen from the group of samples sharing a unique target value. Total
      number of selected samples: npertarget x len(uniquetargets).
      If a `list` is given of length matching the unique target values, it
      specifies the number of samples chosen for each particular unique
      target.
    targets_attr : str, optional

    Returns
    -------
    Dataset
      A dataset instance for the chosen samples. All feature attributes and
      dataset attribute share there data with the source dataset.
    """
    satargets = dataset.sa[targets_attr]
    utargets = satargets.unique
    # if interger is given take this value for all classes
    if isinstance(npertarget, int):
        npertarget = [npertarget for i in utargets]

    sample = []
    # for each available class
    targets = satargets.value
    for i, r in enumerate(utargets):
        # get the list of pattern ids for this class
        sample += random.sample(list((targets == r).nonzero()[0]), npertarget[i] )

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
    return get_nelements_per_value(dataset.sa[attr])


@datasetmethod
def get_samples_by_attr(dataset, attr, values, sort=True):
    """Return indices of samples given a list of attributes
    """

    if not is_sequence_type(values) \
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
def summary(dataset, stats=True, lstats='auto', sstats='auto', idhash=False,
            targets_attr='targets', chunks_attr='chunks',
            maxc=30, maxt=20):
    """String summary over the object

    Parameters
    ----------
    stats : bool
      Include some basic statistics (mean, std, var) over dataset samples
    lstats : 'auto' or bool
      Include statistics on chunks/targets.  If 'auto', includes only if both
      targets_attr and chunks_attr are present.
    sstats : 'auto' or bool
      Sequence (order) statistics. If 'auto', includes only if
      targets_attr is present.
    idhash : bool
      Include idhash value for dataset and samples
    targets_attr : str, optional
      Name of sample attributes of targets
    chunks_attr : str, optional
      Name of sample attributes of chunks -- independent groups of samples
    maxt : int
      Maximal number of targets when provide details on targets/chunks
    maxc : int
      Maximal number of chunks when provide details on targets/chunks
    """
    # local bindings
    samples = dataset.samples
    sa = dataset.sa
    s = str(dataset)[1:-1]

    if idhash:
        s += '\nID-Hashes: %s' % dataset.idhash

    # Deduce if necessary lstats and sstats
    if lstats == 'auto':
        lstats = (targets_attr in sa) and (chunks_attr in sa)
    if sstats == 'auto':
        sstats = (targets_attr in sa)

    ssep = (' ', '\n')[lstats]

    ## Possibly summarize attributes listed as having unique
    if stats:
        if np.issctype(samples.dtype):
            # TODO -- avg per chunk?
            # XXX We might like to use scipy.stats.describe to get
            # quick summary statistics (mean/range/skewness/kurtosis)
            if dataset.nfeatures:
                s += "%sstats: mean=%g std=%g var=%g min=%g max=%g\n" % \
                     (ssep, np.mean(samples), np.std(samples),
                      np.var(samples), np.min(samples), np.max(samples))
            else:
                s += "%sstats: dataset has no features\n" % ssep
        else:
            s += "%sstats: no stats for dataset of '%s' dtype" % (ssep, samples.dtype)
    if lstats:
        try:
            s += dataset.summary_targets(
                targets_attr=targets_attr, chunks_attr=chunks_attr,
                maxc=maxc, maxt=maxt)
        except KeyError, e:
            s += 'No per %s/%s due to %r' % (targets_attr, chunks_attr, e)

    if sstats and not targets_attr is None:
        if len(dataset.sa[targets_attr].unique) < maxt:
            ss = SequenceStats(dataset.sa[targets_attr].value)
            s += str(ss)
        else:
            s += "Number of unique %s > %d thus no sequence statistics" % \
                 (targets_attr, maxt)
    return s

@datasetmethod
def summary_targets(dataset, targets_attr='targets', chunks_attr='chunks',
                    maxc=30, maxt=20):
    """Provide summary statistics over the targets and chunks

    Parameters
    ----------
    dataset : `Dataset`
      Dataset to operate on
    targets_attr : str, optional
      Name of sample attributes of targets
    chunks_attr : str, optional
      Name of sample attributes of chunks -- independent groups of samples
    maxc : int
      Maximal number of chunks when provide details
    maxt : int
      Maximal number of targets when provide details
    """
    # We better avoid bound function since if people only
    # imported Dataset without miscfx it would fail
    spcl = get_samples_per_chunk_target(
        dataset, targets_attr=targets_attr, chunks_attr=chunks_attr)
    # XXX couldn't they be unordered?
    ul = dataset.sa[targets_attr].unique.tolist()
    uc = dataset.sa[chunks_attr].unique.tolist()
    s = ""
    if len(ul) < maxt and len(uc) < maxc:
        s += "\nCounts of targets in each chunk:"
        # only in a reasonable case do printing
        table = [['  %s\\%s' % (chunks_attr, targets_attr)] + ul]
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
                 '#%s' % name2: np.sum(spcl>0, axis=axis)}
        entries = ['  ' + name1, 'mean', 'std', 'min', 'max', '#%s' % name2]
        table = [ entries ]
        for i, l in enumerate(u):
            d = {'  ' + name1 : l}
            d.update(dict([ (k, stats[k][i]) for k in stats.keys()]))
            table.append( [ ('%.3g', '%s')[isinstance(d[e], basestring)]
                            % d[e] for e in entries] )
        return '\nSummary for %s across %s\n' % (name1, name2) \
               + table2string(table)

    if len(ul) < maxt:
        s += cl_stats(0, ul, targets_attr, chunks_attr)
    if len(uc) < maxc:
        s += cl_stats(1, uc, chunks_attr, targets_attr)
    return s


class SequenceStats(dict):
    """Simple helper to provide representation of sequence statistics

    Matlab analog:
    http://cfn.upenn.edu/aguirre/code/matlablib/mseq/mtest.m

    WARNING: Experimental -- API might change without warning!
    Current implementation is ugly!
    """

    # TODO: operate given some "chunks" so it could report also
    #       counter-balance for the borders, mean across chunks, etc
    def __init__(self, seq, order=2):#, chunks=None, chunks_attr=None):
        """Initialize SequenceStats

        Parameters
        ----------
        seq : list or ndarray
          Actual sequence of targets
        order : int
          Maximal order of counter-balancing check. For perfect
          counterbalancing all matrices should be identical
        """

        """
          chunks : None or list or ndarray
            Chunks to use if `perchunk`=True
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
        nsamples = len(seq)                 # # of samples/targets
        utargets = sorted(list(set(seq)))    # unique targets
        ntargets = len(utargets)              # # of targets

        # mapping for targets
        targets_map = dict([(l, i) for i, l in enumerate(utargets)])

        # map sequence first
        seqm = [targets_map[i] for i in seq]
        # npertarget = np.bincount(seqm)

        res = dict(utargets=utargets)
        # Estimate counter-balance
        cbcounts = np.zeros((order, ntargets, ntargets), dtype=int)
        for cb in xrange(order):
            for i, j in zip(seqm[:-(cb+1)], seqm[cb+1:]):
                cbcounts[cb, i, j] += 1
        res['cbcounts'] = cbcounts

        """
        Lets compute relative counter-balancing
        Ideally, npertarget[i]/ntargets should precede each target
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
        t = [ [""] * (1 + self.order*(ntargets+1)) for i in xrange(ntargets+1) ]
        t[0][0] = "Targets/Order"
        for i, l  in enumerate(utargets):
            t[i+1][0] = '%s:' % l
        for cb in xrange(order):
            t[0][1+cb*(ntargets+1)] = "O%d" % (cb+1)
            for i  in xrange(ntargets+1):
                t[i][(cb+1)*(ntargets+1)] = " | "
            m = cbcounts[cb]
            # ??? there should be better way to get indexes
            ind = np.where(~np.isnan(m))
            for i, j in zip(*ind):
                t[1+i][1+cb*(ntargets+1)+j] = '%d' % m[i, j]

        sout = "Sequence statistics for %d entries" \
               " from set %s\n" % (len(seq), utargets) + \
               "Counter-balance table for orders up to %d:\n" % order \
               + table2string(t)
        if len(corr):
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
