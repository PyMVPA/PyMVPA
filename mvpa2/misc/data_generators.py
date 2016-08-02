# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Miscellaneous data generators for unittests and demos"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.datasets.base import dataset_wizard, Dataset
from mvpa2.misc.neighborhood import IndexQueryEngine
from mvpa2 import pymvpa_dataroot, pymvpa_datadbroot
from mvpa2.misc.fx import get_random_rotation
from mvpa2.base.dataset import vstack

from mvpa2.misc.fx import double_gamma_hrf, single_gamma_hrf
from mvpa2.misc.support import Event

if __debug__:
    from mvpa2.base import debug

def multiple_chunks(func, n_chunks, *args, **kwargs):
    """Replicate datasets multiple times raising different chunks

    Given some randomized (noisy) generator of a dataset with a single
    chunk call generator multiple times and place results into a
    distinct chunks.

    Returns
    -------
    ds : `mvpa2.datasets.base.Dataset`
    """
    dss = []
    for chunk in xrange(n_chunks):
        ds_ = func(*args, **kwargs)
        # might not have chunks at all
        if not 'chunks' in ds_.sa:
            ds_.sa['chunks'] = np.repeat(chunk + 1, ds_.nsamples)
        else:
            ds_.sa.chunks[:] = chunk + 1
        dss.append(ds_)

    return vstack(dss)


def dumb_feature_dataset():
    """Create a very simple dataset with 2 features and 3 labels
    """
    data = [[1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1], [4, 0], [4, 1],
            [5, 0], [5, 1], [6, 0], [6, 1], [7, 0], [7, 1], [8, 0], [8, 1],
            [9, 0], [9, 1], [10, 0], [10, 1], [11, 0], [11, 1], [12, 0],
            [12, 1]]
    regs = ([1] * 8) + ([2] * 8) + ([3] * 8)

    return dataset_wizard(samples=np.array(data), targets=regs, chunks=range(len(regs)))


def dumb_feature_binary_dataset():
    """Very simple binary (2 labels) dataset
    """
    data = [[1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1], [4, 0], [4, 1],
            [5, 0], [5, 1], [6, 0], [6, 1], [7, 0], [7, 1], [8, 0], [8, 1],
            [9, 0], [9, 1], [10, 0], [10, 1], [11, 0], [11, 1], [12, 0],
            [12, 1]]
    regs = ([0] * 12) + ([1] * 12)

    return dataset_wizard(samples=np.array(data), targets=regs, chunks=range(len(regs)))


def normal_feature_dataset(perlabel=50, nlabels=2, nfeatures=4, nchunks=5,
                           means=None, nonbogus_features=None, snr=3.0,
                           normalize=True):
    """Generate a univariate dataset with normal noise and specified means.

    Could be considered to be a generalization of
    `pure_multivariate_signal` where means=[ [0,1], [1,0] ].

    Specify either means or `nonbogus_features` so means get assigned
    accordingly.  If neither `means` nor `nonbogus_features` are
    provided, data will be pure noise and no per-label information.

    Parameters
    ----------
    perlabel : int, optional
      Number of samples per each label
    nlabels : int, optional
      Number of labels in the dataset
    nfeatures : int, optional
      Total number of features (including bogus features which carry
      no label-related signal)
    nchunks : int, optional
      Number of chunks (perlabel should be multiple of nchunks)
    means : None or ndarray of (nlabels, nfeatures) shape
      Specified means for each of features (columns) for all labels (rows).
    nonbogus_features : None or list of int
      Indexes of non-bogus features (1 per label).
    snr : float, optional
      Signal-to-noise ration assuming that signal has std 1.0 so we
      just divide random normal noise by snr
    normalize : bool, optional
      Divide by max(abs()) value to bring data into [-1, 1] range.
    """

    data = np.random.standard_normal((perlabel * nlabels, nfeatures))
    if snr != 0:
        data /= np.sqrt(snr)
    if means is None and nonbogus_features is not None:
        if len(nonbogus_features) != nlabels:
            raise ValueError(
                "Provide as many nonbogus features as many labels you have")
        means = np.zeros((len(nonbogus_features), nfeatures))
        # pure multivariate -- single bit per feature
        for i, nbf in enumerate(nonbogus_features):
            means[i, nbf] = 1.0
    if means is not None and snr != 0:
        # add mean
        data += np.repeat(np.array(means, ndmin=2), perlabel, axis=0)
    if normalize:
        # bring it 'under 1', since otherwise some classifiers have difficulties
        # during optimization
        data = 1.0 / (np.max(np.abs(data))) * data
    labels = np.concatenate([np.repeat('L%d' % i, perlabel)
                             for i in range(nlabels)])
    chunks = np.concatenate([np.repeat(range(nchunks),
                                       perlabel // nchunks)
                             for i in range(nlabels)])
    ds = dataset_wizard(data, targets=labels, chunks=chunks)

    # If nonbogus was provided -- assign .a and .fa accordingly
    if nonbogus_features is not None:
        ds.fa['nonbogus_targets'] = np.array([None] * nfeatures)
        ds.fa.nonbogus_targets[nonbogus_features] = ['L%d' % i for i in range(nlabels)]
        ds.a['nonbogus_features'] = nonbogus_features
        ds.a['bogus_features'] = [x for x in range(nfeatures)
                                  if not x in nonbogus_features]
    return ds


def pure_multivariate_signal(patterns, signal2noise=1.5, chunks=None,
                             targets=None):
    """ Create a 2d dataset with a clear purely multivariate signal.

    This is known is the XOR problem.

    ::

      %%%%%%%%%
      % O % X %
      %%%%%%%%%
      % X % O %
      %%%%%%%%%

    Parameters
    ----------
    patterns: int
      Number of data points in each of the four dot clouds
    signal2noise: float, optional
      Univariate signal pedestal.
    chunks: array, optional
      Vector for chunk labels for all generated samples.
    targets: list, optional
      Length-2 sequence of target values for both classes. If None,
      [0, 1] is used.
    """
    if targets is None:
        targets = [0, 1]

    # start with noise
    data = np.random.normal(size=(4 * patterns, 2))

    # add signal
    data[:2 * patterns, 1] += signal2noise

    data[2 * patterns:4 * patterns, 1] -= signal2noise
    data[:patterns, 0] -= signal2noise
    data[2 * patterns:3 * patterns, 0] -= signal2noise
    data[patterns:2 * patterns, 0] += signal2noise
    data[3 * patterns:4 * patterns, 0] += signal2noise

    # two conditions
    regs = np.array((targets[0:1] * patterns) + (targets[1:2] * 2 * patterns) + (targets[0:1] * patterns))

    if chunks is None:
        chunks = range(len(data))
    return dataset_wizard(samples=data, targets=regs, chunks=chunks)


def get_mv_pattern(s2n):
    """Simple multivariate dataset"""
    return multiple_chunks(pure_multivariate_signal, 6, 5, s2n, 1)


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
    intervals = np.array([[-1.932, -0.453], [0.534, 3.142]])
    r = np.array([2.0, 1.3])
    x = np.random.rand(size, 2)
    x *= np.array(intervals[:, 1] - intervals[:, 0])
    x += np.array(intervals[:, 0])
    if __debug__:
        for i in xrange(2):
            debug('DG', '%d columnt Min: %g Max: %g' %
                  (i, x[:, i].min(), x[:, i].max()))
    y = r[0] * np.cos(x[:, 0] + r[1] * np.cos(x.sum(1))) + \
        np.random.randn(size) * np.sqrt(0.0025)
    y -= y.mean()
    x34 = x + np.random.randn(size, 2) * 0.02
    x56 = np.random.randn(size, 2)
    x = np.hstack([x, x34, x56])
    return dataset_wizard(samples=x, targets=y)


def sin_modulated(n_instances, n_features,
                  flat=False, noise=0.4):
    """ Generate a (quite) complex multidimensional non-linear dataset

    Used for regression testing. In the data label is a sin of a x^2 +
    uniform noise
    """
    if flat:
        data = (np.arange(0.0, 1.0, 1.0 / n_instances) * np.pi)
        data.resize(n_instances, n_features)
    else:
        data = np.random.rand(n_instances, n_features) * np.pi
    label = np.sin((data ** 2).sum(1)).round()
    label += np.random.rand(label.size) * noise
    return dataset_wizard(samples=data, targets=label)


def chirp_linear(n_instances, n_features=4, n_nonbogus_features=2,
                 data_noise=0.4, noise=0.1):
    """ Generates simple dataset for linear regressions

    Generates chirp signal, populates n_nonbogus_features out of
    n_features with it with different noise level and then provides
    signal itself with additional noise as labels
    """
    x = np.linspace(0, 1, n_instances)
    y = np.sin((10 * np.pi * x ** 2))

    data = np.random.normal(size=(n_instances, n_features)) * data_noise
    for i in xrange(n_nonbogus_features):
        data[:, i] += y[:]

    labels = y + np.random.normal(size=(n_instances,)) * noise

    return dataset_wizard(samples=data, targets=labels)


def linear_awgn(size=10, intercept=0.0, slope=0.4, noise_std=0.01, flat=False):
    """Generate a dataset from a linear function with AWGN
    (Added White Gaussian Noise).

    It can be multidimensional if 'slope' is a vector. If flat is True
    (in 1 dimesion) generate equally spaces samples instead of random
    ones. This is useful for the test phase.
    """
    dimensions = 1
    if isinstance(slope, np.ndarray):
        dimensions = slope.size

    if flat and dimensions == 1:
        x = np.linspace(0, 1, size)[:, np.newaxis]
    else:
        x = np.random.rand(size, dimensions)

    y = np.dot(x, slope)[:, np.newaxis] \
        + (np.random.randn(*(x.shape[0], 1)) * noise_std) + intercept

    return dataset_wizard(samples=x, targets=y)


def noisy_2d_fx(size_per_fx, dfx, sfx, center, noise_std=1):
    """Yet another generator of random dataset

    """
    # used in projection example
    x = []
    y = []
    labels = []
    for fx in sfx:
        nx = np.random.normal(size=size_per_fx)
        ny = fx(nx) + np.random.normal(size=nx.shape, scale=noise_std)
        x.append(nx)
        y.append(ny)

        # whenever larger than first function value
        labels.append(np.array(ny < dfx(nx), dtype='int'))

    samples = np.array((np.hstack(x), np.hstack(y))).squeeze().T
    labels = np.hstack(labels).squeeze().T

    samples += np.array(center)

    return dataset_wizard(samples=samples, targets=labels)


def linear1d_gaussian_noise(size=100, slope=0.5, intercept=1.0,
                            x_min=-2.0, x_max=3.0, sigma=0.2):
    """A straight line with some Gaussian noise.
    """
    x = np.linspace(start=x_min, stop=x_max, num=size)
    noise = np.random.randn(size) * sigma
    y = x * slope + intercept + noise
    return dataset_wizard(samples=x[:, None], targets=y)


def autocorrelated_noise(ds, sr, cutoff, lfnl=3.0, bord=10, hfnl=None, add_baseline=True):
    """Generate a dataset with samples being temporally autocorrelated noise.

    Parameters
    ----------
    ds : Dataset
      Source dataset whose mean samples serves as the pedestal of the new noise
      samples. All attributes of this dataset will also go into the generated
      one.
    sr : float
      Sampling rate (in Hz) of the samples in the dataset.
    cutoff : float
      Cutoff frequency of the low-pass butterworth filter.
    bord : int
      Order of the butterworth filter that is applied for low-pass
      filtering.
    lfnl : float
      Low frequency noise level in percent signal (per feature).
    hfnl : float or None
      High frequency noise level in percent signal (per feature). If None, no
      HF noise is added.
    """
    from scipy.signal import butter, lfilter

    # something to play with
    fds = ds.copy(deep=False)

    # compute the pedestal
    msample = fds.samples.mean(axis=0)

    # noise/signal amplitude relative to each feature mean signal
    noise_amps = msample * (lfnl / 100.)

    # generate gaussian noise for the full dataset
    nsamples = np.random.standard_normal(fds.samples.shape)
    # scale per each feature
    nsamples *= noise_amps

    # nyquist frequency
    nf = sr / 2.0

    # along samples low-pass filtering
    fb, fa = butter(bord, cutoff / nf)
    nsamples = lfilter(fb, fa, nsamples, axis=0)

    # add the pedestal
    if add_baseline:
        nsamples += msample

    # HF noise
    if hfnl is not None:
        noise_amps = msample * (hfnl / 100.)
        nsamples += np.random.standard_normal(nsamples.shape) * noise_amps

    fds.samples = nsamples
    return fds


def random_affine_transformation(ds, scale_fac=100., shift_fac=10.):
    """Distort a dataset by random scale, shift, and rotation.

    The original data samples are transformed by applying a random rotation,
    shifting by a random vector (randomly selected, scaled input sample), and
    scaled by a random factor (randomly selected input feature values, scaled
    by an additional factor). The effective transformation values are stored in
    the output dataset's attribute collection as 'random_rotation',
    'random_shift', and 'random_scale' respectively.

    Parameters
    ----------
    ds : Dataset
      Input dataset. Its sample and features attributes will be assigned to the
      output dataset.
    scale_fac : float
      Factor by which the randomly selected value for data scaling is scaled
      itself.
    shift_fac : float
      Factor by which the randomly selected shift vector is scaled.
    """
    rndidx = np.random.randint
    R = get_random_rotation(ds.nfeatures)
    samples = ds.samples
    # reusing random data from dataset itself
    random_scale = samples[rndidx(len(ds)), rndidx(ds.nfeatures)] * scale_fac
    random_shift = samples[rndidx(len(ds))] * shift_fac
    samples = np.dot(samples, R) * random_scale + random_shift
    return Dataset(samples, sa=ds.sa, fa=ds.fa,
                   a={'random_rotation': R,
                      'random_scale': random_scale,
                      'random_shift': random_shift})


def simple_hrf_dataset(
        events=None,
        hrf_gen=lambda t: double_gamma_hrf(t) - single_gamma_hrf(t, 0.8, 1, 0.05),
        fir_length=15,
        nsamples=None,
        tr=2.0,
        tres=1,
        baseline=800.0,
        signal_level=1,
        noise='normal',
        noise_level=1,
        resampling='scipy'):
    """
    events: list of Events or ndarray of onsets for simple(r) designs
    """
    if events is None:
        events = [1, 20, 25, 50, 60, 90, 92, 140]
    if isinstance(events, np.ndarray) or not isinstance(events[0], dict):
        events = [Event(onset=o) for o in events]
    else:
        assert(isinstance(events, list))
        for e in events:
            assert(isinstance(e, dict))

    # play fmri
    # full-blown HRF with initial dip and undershoot ;-)
    hrf_x = np.arange(0, float(fir_length) * tres, tres)
    if isinstance(hrf_gen, np.ndarray):
        # just accept provided HRF and only verify size match
        assert(len(hrf_x) == len(hrf_gen))
        hrf = hrf_gen
    else:
        # actually generate it
        hrf = hrf_gen(hrf_x)
    if not nsamples:
        # estimate number of samples needed if not provided
        max_onset = max([e['onset'] for e in events])
        nsamples = int(max_onset / tres + len(hrf_x) * 1.5)

    # come up with an experimental design
    fast_er = np.zeros(nsamples)
    for e in events:
        on = int(e['onset'] / float(tres))
        off = int((e['onset'] + e.get('duration', 1.)) / float(tres))
        if off == on:
            off += 1                      # so we have at least 1 point
        assert(range(on, off))
        fast_er[on:off] = e.get('intensity', 1)
    # high resolution model of the convolved regressor
    model_hr = np.convolve(fast_er, hrf)[:nsamples]

    # downsample the regressor to fMRI resolution
    if resampling == 'scipy':
        from scipy import signal
        model_lr = signal.resample(model_hr,
                                   int(tres * nsamples / tr),
                                   window='ham')
    elif resampling == 'naive':
        if tr % tres != 0.0:
            raise ValueError("You must use resample='scipy' since your TR=%.2g"
                             " is not multiple of tres=%.2g" % (tr, tres))
        if tr < tres:
            raise ValueError("You must use resample='scipy' since your TR=%.2g"
                             " is less than tres=%.2g" % (tr, tres))
        step = int(tr // tres)
        model_lr = model_hr[::step]
    else:
        raise ValueError("resampling can only be 'scipy' or 'naive'. Got %r"
                         % resampling)

    # generate artifical fMRI data: two voxels one is noise, one has
    # something
    wsignal = baseline + model_lr * signal_level
    nsignal = np.ones(wsignal.shape) * baseline

    # build design matrix: bold-regressor and constant
    design = np.array([model_lr, np.repeat(1, len(model_lr))]).T

    # two 'voxel' dataset
    ds = dataset_wizard(samples=np.array((wsignal, nsignal)).T, targets=1)
    ds.a['baseline'] = baseline
    ds.a['tr'] = tr
    ds.sa['design'] = design

    ds.fa['signal_level'] = [signal_level, False]

    if noise == 'autocorrelated':
        # this one seems to be quite unstable and can provide really
        # funky noise at times
        noise = autocorrelated_noise(ds, 1 / tr, 1 / (2 * tr),
                                     lfnl=noise_level, hfnl=noise_level,
                                     add_baseline=False)
    elif noise == 'normal':
        noise = np.random.randn(*ds.shape) * noise_level
    else:
        raise ValueError(noise)
    ds.sa['noise'] = noise
    ds.samples += noise
    return ds

def local_random_affine_transformations(
        ds, distort_seeds, distort_neighbor, space, scale_fac=100,
        shift_fac=10):
    """Distort a dataset in the local neighborhood of selected features.

    This function is similar to ``random_affine_transformation()``, but applies
    multiple random affine transformations to a spatially constraint local
    neighborhood.

    Parameters
    ----------
    ds : Dataset
      The to be transformed/distorted dataset.
    distort_seeds : list(int)
      This a sequence of feature ids (corresponding to the input dataset) that
      serve as anchor to determine the local neighborhood for a distortion. The
      number of seeds also determines the number of different local distortions
      that are going to be applied.
    distort_neighbor : callable
      And object that when called with a coordinate generates a sequence of
      coordinates that comprise its neighborhood (see e.g. ``Sphere()``).
    space : str
      Name of the feature attribute of the input dataset that contains the
      relevant feature coordinates (e.g. 'voxel_indices').
    scale_fac : float
      See ``random_affine_transformation()``
    shift_fac : float
      See ``random_affine_transformation()``

    Returns
    -------
    Dataset
      A dataset derived from the input dataset with added local distortions.
    """
    # which dataset attributes to aggregate
    random_stats = ['random_rotation', 'random_scale', 'random_shift']
    kwa = {space: distort_neighbor}
    qe = IndexQueryEngine(**kwa)
    qe.train(ds)
    ds_distorted = ds.copy()
    for stat in random_stats:
        ds_distorted.a[stat + 's'] = {}
    # for each seed region
    for seed in distort_seeds:
        # select the neighborhood for this seed
        # take data from the distorted dataset to avoid
        # 'loosing' previous distortions
        distort_ids = qe[seed]
        ds_d = random_affine_transformation(
                               ds_distorted[:, distort_ids],
                               scale_fac=scale_fac,
                               shift_fac=shift_fac)
        # recover the distortions stats for this seed
        for stat in random_stats:
            ds_distorted.a[stat + 's'].value[seed] = ds_d.a[stat].value
        # put the freshly distorted data back
        ds_distorted.samples[:, distort_ids] = ds_d.samples
    return ds_distorted
