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

from sets import Set
from operator import isSequenceType
import random

import numpy as N

from mvpa.datasets.base import Dataset, datasetmethod
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
      and N.array(get_nsamples_per_attr(dataset, 'chunks').values()).min() < 2:
        warning("Z-scoring chunk-wise and one chunk with less than two " \
                "samples will set features in these samples to zero.")

    # cast the data to float, since in-place operations below to not upcast!
    if N.issubdtype(dataset.samples.dtype, N.integer):
        dataset.samples = dataset.samples.astype(targetdtype)

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
            samples[:, std != 0] /= std[std != 0]
        else:
            samples /= std

        return samples

    if baselinelabels is None:
        statids = None
    else:
        statids = Set(get_samples_by_attr(dataset, 'labels', baselinelabels))

    # for the sake of speed yoh didn't simply create a list
    # [True]*dataset.nsamples to provide easy selection of everything
    if perchunk:
        for c in dataset.sa['chunks'].unique:
            slicer = N.where(dataset.chunks == c)[0]
            if not statids is None:
                statslicer = list(statids.intersection(Set(slicer)))
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

    return Dataset(samples=N.array(agg, ndmin=2).T, sa=dataset.sa)


@datasetmethod
def removeInvariantFeatures(dataset):
    """Returns a new dataset with all invariant features removed.
    """
    return dataset[:, dataset.samples.std(axis=0).nonzero()[0]]


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
def getSamplesPerChunkLabel(dataset):
    """Returns an array with the number of samples per label in each chunk.

    Array shape is (chunks x labels).

    :Parameters:
      dataset: Dataset
        Source dataset.
    """
    ul = dataset.sa['labels'].unique
    uc = dataset.sa['chunks'].unique

    count = N.zeros((len(uc), len(ul)), dtype='uint')

    for cc, c in enumerate(uc):
        for lc, l in enumerate(ul):
            count[cc, lc] = N.sum(N.logical_and(dataset.labels == l,
                                                dataset.chunks == c))

    return count


@datasetmethod
def permute_labels(dataset, perchunk=True, assure_permute=False):
    """Permute the labels of a Dataset.

    A new permuted set of labels is assigned to the dataset, replacing
    the previous labels. The labels are not modified in-place, hence
    it is safe to call the function on shallow copies of a dataset
    without modifying the original dataset's labels.

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
      If True, assures that labels are permutted, i.e. any one is
      different from the original one
    """
    if __debug__:
        if len(N.unique(dataset.sa.labels)) < 2:
            raise RuntimeError(
                  "Permuting labels is only meaningful if there are "
                  "more than two different labels.")

    # local binding
    labels = dataset.sa['labels'].value

    # now scramble
    if perchunk and dataset.sa.isKnown('chunks'):
        chunks = dataset.sa['chunks'].value

        plabels = N.zeros(labels.shape)

        for o in dataset.sa['chunks'].unique:
            plabels[chunks == o] = \
                N.random.permutation(labels[chunks == o])
    else:
        plabels = N.random.permutation(labels)

    if assure_permute:
        if not (plabels != labels).any():
            if not (assure_permute is True):
                if assure_permute == 1:
                    raise RuntimeError, \
                          "Cannot assure permutation of labels %s for " \
                          "some reason with chunks %s and while " \
                          "perchunk=%s . Should not happen" % \
                          (labels, self.chunks, perchunk)
            else:
                assure_permute = 11 # make 10 attempts
            if __debug__:
                debug("DS",  "Recalling permute to assure different labels")
            permute_labels(dataset,
                           perchunk=perchunk,
                           assure_permute=assure_permute-1)
    # reassign to the dataset
    dataset.sa.labels = plabels


@datasetmethod
def random_samples(dataset, nperlabel):
    """Create a dataset with a random subset of samples.

    Parameters
    ----------
    nperlabel : int, list

      If an integer is given, the specified number of samples is randomly
      choosen from the group of samples sharing a unique label value (total
      number of selected samples: nperlabel x len(uniquelabels).

      If a list is given which's length is matching the unique label values, it
      will specify the number of samples chosen for each particular unique
      label.

    Returns
    -------
    A dataset instance for the chosen samples. All feature attributes and
    dataset attribute share there data with the source dataset.
    """
    uniquelabels = dataset.sa['labels'].unique
    # if interger is given take this value for all classes
    if isinstance(nperlabel, int):
        nperlabel = [nperlabel for i in uniquelabels]

    sample = []
    # for each available class
    labels = dataset.labels
    for i, r in enumerate(uniquelabels):
        # get the list of pattern ids for this class
        sample += random.sample((labels == r).nonzero()[0], nperlabel[i] )

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

    # use dictionary to cope with arbitrary labels
    result = dict(zip(uniqueattr, [ 0 ] * len(uniqueattr)))
    for l in dataset.sa[attr].value:
        result[l] += 1

    return result


@datasetmethod
def get_samples_by_attr(dataset, attr, values, sort=True):
    """Return indecies of samples given a list of attributes
    """

    if not isSequenceType(values) \
           or isinstance(values, basestring):
        values = [ values ]

    # TODO: compare to plain for loop through the labels
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
