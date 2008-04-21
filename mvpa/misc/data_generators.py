#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Miscelaneous data generators for unittests and demos"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.datasets.dataset import Dataset


def dumbFeatureDataset():
    data = [[1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1], [4, 0], [4, 1],
            [5, 0], [5, 1], [6, 0], [6, 1], [7, 0], [7, 1], [8, 0], [8, 1],
            [9, 0], [9, 1], [10, 0], [10, 1], [11, 0], [11, 1], [12, 0],
            [12, 1]]
    regs = ([1] * 8) + ([2] * 8) + ([3] * 8)

    return Dataset(samples=data, labels=regs)


def dumbFeatureBinaryDataset():
    data = [[1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1], [4, 0], [4, 1],
            [5, 0], [5, 1], [6, 0], [6, 1], [7, 0], [7, 1], [8, 0], [8, 1],
            [9, 0], [9, 1], [10, 0], [10, 1], [11, 0], [11, 1], [12, 0],
            [12, 1]]
    regs = ([0] * 12) + ([1] * 12)

    return Dataset(samples=data, labels=regs)



def normalFeatureDataset(perlabel=50, nlabels=2, nfeatures=4, nchunks=5,
                         means=None, nonbogus_features=None, snr=1.0):
    """Generate a dataset where each label is some normally
    distributed beastie around specified mean (0 if None).

    snr is assuming that signal has std 1.0 so we just divide noise by snr

    Probably it is a generalization of pureMultivariateSignal where
    means=[ [0,1], [1,0] ]

    Specify either means or nonbogus_features so means get assigned
    accordingly
    """

    data = N.random.standard_normal((perlabel*nlabels, nfeatures))/N.sqrt(snr)
    if (means is None) and (not nonbogus_features is None):
        if len(nonbogus_features) > nlabels:
            raise ValueError, "Can't assign simply a feature to a " + \
                  "class: more nonbogus_features than labels"
        means = N.zeros((len(nonbogus_features), nfeatures))
        # pure multivariate -- single bit per feature
        for i in xrange(len(nonbogus_features)):
            means[i, nonbogus_features[i]] = 1.0
    if not means is None:
        # add mean
        data += N.repeat(N.array(means, ndmin=2), perlabel, axis=0)
    labels = N.concatenate([N.repeat(i, perlabel) for i in range(nlabels)])
    chunks = N.concatenate([N.repeat(range(nchunks),
                                     perlabel/nchunks) for i in range(nlabels)])
    return Dataset(samples=data, labels=labels, chunks=chunks)


def pureMultivariateSignal(patterns, signal2noise = 1.5, chunks=None):
    """ Create a 2d dataset with a clear multivariate signal, but no
    univariate information.

    %%%%%%%%%
    % O % X %
    %%%%%%%%%
    % X % O %
    %%%%%%%%%
    """

    # start with noise
    data = N.random.normal(size=(4*patterns, 2))

    # add signal
    data[:2*patterns, 1] += signal2noise

    data[2*patterns:4*patterns, 1] -= signal2noise
    data[:patterns, 0] -= signal2noise
    data[2*patterns:3*patterns, 0] -= signal2noise
    data[patterns:2*patterns, 0] += signal2noise
    data[3*patterns:4*patterns, 0] += signal2noise

    # two conditions
    regs = N.array(([0] * patterns) + ([1] * 2 * patterns) + ([0] * patterns))

    return Dataset(samples=data, labels=regs, chunks=chunks)


def normalFeatureDataset__(dataset=None, labels=None, nchunks=None,
                           perlabel=50, activation_probability_steps=1,
                           randomseed=None, randomvoxels=False):

    if dataset is None and labels is None:
        raise ValueError, \
              "Provide at least labels or a background dataset"

    if dataset is None:
        Nlabels = len(labels)
    else:
        nchunks = len(dataset.uniquechunks)

    N.random.seed(randomseed)

    # Create a sequence of indexes from which to select voxels to be used
    # for features
    if randomvoxels:
        indexes = N.random.permutation(dataset.nfeatures)
    else:
        indexes = N.arange(dataset.nfeatures)

    allind, maps = [], []
    if __debug__:
        debug('DG', "Ugly creation of the copy of background")

    dtype = dataset.samples.dtype
    if not N.issubdtype(dtype, N.float):
        dtype = N.float
    totalsignal = N.zeros(dataset.samples.shape, dtype=dtype)

    for l in xrange(len(labels)):
        label = labels[l]
        if __debug__:
            debug('DG', "Simulating independent labels for %s" % label)

        # What sample ids belong to this label
        labelids = dataset.idsbylabels(label)


        # What features belong here and what is left over
        nfeatures = perlabel * activation_probability_steps
        ind, indexes = indexes[0:nfeatures], \
                       indexes[nfeatures+1:]
        allind += list(ind)              # store what indexes we used

        # Create a dataset only for 'selected' features
        # NB there is sideeffect that selectFeatures will sort those ind provided
        ds = dataset.selectFeatures(ind)
        ds.samples[:] = 0.0             # zero them out

        # assign data
        prob = [1.0 - x*1.0/activation_probability_steps
                for x in xrange(activation_probability_steps)]

        # repeat so each feature gets itw own
        probabilities = N.repeat(prob, perlabel)
        verbose(4, 'For prob=%s probabilities=%s' % (prob, probabilities))

        for chunk in ds.uniquechunks:
            chunkids = ds.idsbychunks(chunk) # samples in this chunk
            ids = list(Set(chunkids).intersection(Set(labelids)))
            chunkvalue = N.random.uniform() # random number to decide either
                                        # to 'activate' the voxel
            for id_ in ids:
                ds.samples[id_, :] = (chunkvalue <= probabilities).astype('float')
            #verbose(5, "Chunk %d Chunkids %s ids %s" % (chunk, chunkids, ids))

        maps.append(N.array(probabilities, copy=True))

        signal = ds.map2Nifti(ds.samples)
        totalsignal[:,ind] += ds.samples

    # figure out average variance across all 'working' features
    wfeatures = dataset.samples[:, allind]
    meanstd = N.mean(N.std(wfeatures, 1))
    verbose(2, "Mean deviation is %f" % meanstd)

    totalsignal *= meanstd * options.snr
    # add signal on top of background
    dataset.samples += totalsignal

    return dataset

def getMVPattern(s2n):
    run1 = pureMultivariateSignal(5, s2n, 1)
    run2 = pureMultivariateSignal(5, s2n, 2)
    run3 = pureMultivariateSignal(5, s2n, 3)
    run4 = pureMultivariateSignal(5, s2n, 4)
    run5 = pureMultivariateSignal(5, s2n, 5)
    run6 = pureMultivariateSignal(5, s2n, 6)

    data = run1 + run2 + run3 + run4 + run5 + run6

    return data


