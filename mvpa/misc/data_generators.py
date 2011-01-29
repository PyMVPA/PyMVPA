# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Miscelaneous data generators for unittests and demos"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.datasets import Dataset

if __debug__:
    from mvpa.base import debug

def multipleChunks(func, n_chunks, *args, **kwargs):
    """Replicate datasets multiple times raising different chunks

    Given some randomized (noisy) generator of a dataset with a single
    chunk call generator multiple times and place results into a
    distinct chunks
    """
    for chunk in xrange(n_chunks):
        dataset_ = func(*args, **kwargs)
        dataset_.chunks[:] = chunk + 1
        if chunk == 0:
            dataset = dataset_
        else:
            dataset += dataset_

    return dataset


def dumbFeatureDataset():
    """Create a very simple dataset with 2 features and 3 labels
    """
    data = [[1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1], [4, 0], [4, 1],
            [5, 0], [5, 1], [6, 0], [6, 1], [7, 0], [7, 1], [8, 0], [8, 1],
            [9, 0], [9, 1], [10, 0], [10, 1], [11, 0], [11, 1], [12, 0],
            [12, 1]]
    regs = ([1] * 8) + ([2] * 8) + ([3] * 8)

    return Dataset(samples=data, labels=regs)


def dumbFeatureBinaryDataset():
    """Very simple binary (2 labels) dataset
    """
    data = [[1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1], [4, 0], [4, 1],
            [5, 0], [5, 1], [6, 0], [6, 1], [7, 0], [7, 1], [8, 0], [8, 1],
            [9, 0], [9, 1], [10, 0], [10, 1], [11, 0], [11, 1], [12, 0],
            [12, 1]]
    regs = ([0] * 12) + ([1] * 12)

    return Dataset(samples=data, labels=regs)



def normalFeatureDataset(perlabel=50, nlabels=2, nfeatures=4, nchunks=5,
                         means=None, nonbogus_features=None, snr=3.0):
    """Generate a univariate dataset with normal noise and specified means.

    :Keywords:
      perlabel : int
         Number of samples per each label
      nlabels : int
         Number of labels in the dataset
      nfeatures : int
         Total number of features (including bogus features which carry
         no label-related signal)
      nchunks : int
         Number of chunks (perlabel should be multiple of nchunks)
      means : None or list of float or ndarray
         Specified means for each of features among nfeatures.
      nonbogus_features : None or list of int
         Indexes of non-bogus features (1 per label)
      snr : float
         Signal-to-noise ration assuming that signal has std 1.0 so we
         just divide random normal noise by snr

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
    # bring it 'under 1', since otherwise some classifiers have difficulties
    # during optimization
    data = 1.0/(N.max(N.abs(data))) * data
    labels = N.concatenate([N.repeat('L%d' % i, perlabel)
                                for i in range(nlabels)])
    chunks = N.concatenate([N.repeat(range(nchunks),
                                     perlabel/nchunks) for i in range(nlabels)])
    ds = Dataset(samples=data, labels=labels, chunks=chunks, labels_map=True)
    ds.nonbogus_features = nonbogus_features
    return ds

def pureMultivariateSignal(patterns, signal2noise = 1.5, chunks=None):
    """ Create a 2d dataset with a clear multivariate signal, but no
    univariate information.

    ::

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
    """ NOT FINISHED """
    raise NotImplementedError

    if dataset is None and labels is None:
        raise ValueError, \
              "Provide at least labels or a background dataset"

    if dataset is None:
        nlabels = len(labels)
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

        # Create a dataset only for 'selected' features NB there is
        # sideeffect that selectFeatures will sort those ind provided
        ds = dataset.selectFeatures(ind)
        ds.samples[:] = 0.0             # zero them out

        # assign data
        prob = [1.0 - x*1.0/activation_probability_steps
                for x in xrange(activation_probability_steps)]

        # repeat so each feature gets itw own
        probabilities = N.repeat(prob, perlabel)
        if __debug__:
            debug('DG', 'For prob=%s probabilities=%s' % (prob, probabilities))

        for chunk in ds.uniquechunks:
            chunkids = ds.idsbychunks(chunk) # samples in this chunk
            ids = list(set(chunkids).intersection(set(labelids)))
            chunkvalue = N.random.uniform() # random number to decide either
                                        # to 'activate' the voxel
            for id_ in ids:
                ds.samples[id_, :] = \
                                (chunkvalue <= probabilities).astype('float')

        maps.append(N.array(probabilities, copy=True))

        signal = ds.map2Nifti(ds.samples)
        totalsignal[:, ind] += ds.samples

    # figure out average variance across all 'working' features
    wfeatures = dataset.samples[:, allind]
    meanstd = N.mean(N.std(wfeatures, 1))
    if __debug__:
        debug('DG', "Mean deviation is %f" % meanstd)

    totalsignal *= meanstd * options.snr
    # add signal on top of background
    dataset.samples += totalsignal

    return dataset

def getMVPattern(s2n):
    """Simple multivariate dataset"""
    return multipleChunks(pureMultivariateSignal, 6,
                          5, s2n, 1)


def wr1996(size=200):
    """Generate '6d robot arm' dataset (Williams and Rasmussen 1996)

    Was originally created in order to test the correctness of the
    implementation of kernel ARD.  For full details see:
    http://www.gaussianprocess.org/gpml/code/matlab/doc/regression.html#ard

    x_1 picked randomly in [-1.932, -0.453]
    x_2 picked randomly in [0.534, 3.142]
    r_1 = 2.0
    r_2 = 1.3
    f(x_1,x_2) = r_1 cos (x_1) + r_2 cos(x_1 + x_2) + N(0,0.0025)
    etc.

    Expected relevances:
    ell_1      1.804377
    ell_2      1.963956
    ell_3      8.884361
    ell_4     34.417657
    ell_5   1081.610451
    ell_6    375.445823
    sigma_f    2.379139
    sigma_n    0.050835
    """
    intervals = N.array([[-1.932, -0.453], [0.534, 3.142]])
    r = N.array([2.0, 1.3])
    x = N.random.rand(size, 2)
    x *= N.array(intervals[:, 1]-intervals[:, 0])
    x += N.array(intervals[:, 0])
    if __debug__:
        for i in xrange(2):
            debug('DG', '%d columnt Min: %g Max: %g' %
                  (i, x[:, i].min(), x[:, i].max()))
    y = r[0]*N.cos(x[:, 0] + r[1]*N.cos(x.sum(1))) + \
        N.random.randn(size)*N.sqrt(0.0025)
    y -= y.mean()
    x34 = x + N.random.randn(size, 2)*0.02
    x56 = N.random.randn(size, 2)
    x = N.hstack([x, x34, x56])
    return Dataset(samples=x, labels=y)


def sinModulated(n_instances, n_features,
                  flat=False, noise=0.4):
    """ Generate a (quite) complex multidimensional non-linear dataset

    Used for regression testing. In the data label is a sin of a x^2 +
    uniform noise
    """
    if flat:
        data = (N.arange(0.0, 1.0, 1.0/n_instances)*N.pi)
        data.resize(n_instances, n_features)
    else:
        data = N.random.rand(n_instances, n_features)*N.pi
    label = N.sin((data**2).sum(1)).round()
    label += N.random.rand(label.size)*noise
    return Dataset(samples=data, labels=label)

def chirpLinear(n_instances, n_features=4, n_nonbogus_features=2,
                data_noise=0.4, noise=0.1):
    """ Generates simple dataset for linear regressions

    Generates chirp signal, populates n_nonbogus_features out of
    n_features with it with different noise level and then provides
    signal itself with additional noise as labels
    """
    x = N.linspace(0, 1, n_instances)
    y = N.sin((10 * N.pi * x **2))

    data = N.random.normal(size=(n_instances, n_features ))*data_noise
    for i in xrange(n_nonbogus_features):
        data[:, i] += y[:]

    labels = y + N.random.normal(size=(n_instances,))*noise

    return Dataset(samples=data, labels=labels)


def linear_awgn(size=10, intercept=0.0, slope=0.4, noise_std=0.01, flat=False):
    """Generate a dataset from a linear function with AWGN
    (Added White Gaussian Noise).

    It can be multidimensional if 'slope' is a vector. If flat is True
    (in 1 dimesion) generate equally spaces samples instead of random
    ones. This is useful for the test phase.
    """
    dimensions = 1
    if isinstance(slope, N.ndarray):
        dimensions = slope.size

    if flat and dimensions == 1:
        x = N.linspace(0, 1, size)[:, N.newaxis]
    else:
        x = N.random.rand(size, dimensions)

    y = N.dot(x, slope)[:, N.newaxis] \
        + (N.random.randn(*(x.shape[0], 1)) * noise_std) + intercept

    return Dataset(samples=x, labels=y)


def noisy_2d_fx(size_per_fx, dfx, sfx, center, noise_std=1):
    """
    """
    x = []
    y = []
    labels = []
    for fx in sfx:
        nx = N.random.normal(size=size_per_fx)
        ny = fx(nx) + N.random.normal(size=nx.shape, scale=noise_std)
        x.append(nx)
        y.append(ny)

        # whenever larger than first function value
        labels.append(N.array(ny < dfx(nx), dtype='int'))

    samples = N.array((N.hstack(x), N.hstack(y))).squeeze().T
    labels = N.hstack(labels).squeeze().T

    samples += N.array(center)

    return Dataset(samples=samples, labels=labels)
