# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Provides convenience datasets for unittesting.

Also performs testing of storing/reloading datasets into hdf5 file if
cfg.getboolean('tests', 'use hdf datasets'
"""

__docformat__ = 'restructuredtext'

import tempfile
import shutil
import os
import traceback as tbm
import sys
import numpy as np

from mvpa import cfg, externals
from mvpa.datasets import Dataset
from mvpa.datasets.splitters import OddEvenSplitter
from mvpa.misc.data_generators import *

__all__ = [ 'datasets', 'get_random_rotation', 'saveload_warehouse',
            'pure_multivariate_signal']

# Define datasets to be used all over. Split-half later on is used to
# split into training/testing
#
snr_scale = cfg.get_as_dtype('tests', 'snr scale', float, default=1.0)

specs = {'large' : { 'perlabel': 99, 'nchunks': 11,
                     'nfeatures': 20, 'snr': 8 * snr_scale},
         'medium' :{ 'perlabel': 24, 'nchunks': 6,
                     'nfeatures': 14, 'snr': 8 * snr_scale},
         'small' : { 'perlabel': 12, 'nchunks': 4,
                     'nfeatures': 6, 'snr' : 14 * snr_scale} }

# Lets permute upon each invocation of test, so we could possibly
# trigger some funny cases
nonbogus_pool = np.random.permutation([0, 1, 3, 5])

datasets = {}

for kind, spec in specs.iteritems():
    # set of univariate datasets
    for nlabels in [ 2, 3, 4 ]:
        basename = 'uni%d%s' % (nlabels, kind)
        nonbogus_features = nonbogus_pool[:nlabels]

        dataset = normal_feature_dataset(
            nlabels=nlabels,
            nonbogus_features=nonbogus_features,
            **spec)

        oes = OddEvenSplitter()
        splits = [(train, test) for (train, test) in oes(dataset)]
        for i, replication in enumerate( ['test', 'train'] ):
            dataset_ = splits[0][i]
            datasets["%s_%s" % (basename, replication)] = dataset_

        # full dataset
        datasets[basename] = dataset

    # sample 3D
    total = 2*spec['perlabel']
    nchunks = spec['nchunks']
    data = np.random.standard_normal(( total, 3, 6, 6 ))
    labels = np.concatenate( ( np.repeat( 0, spec['perlabel'] ),
                              np.repeat( 1, spec['perlabel'] ) ) )
    chunks = np.asarray(range(nchunks)*(total/nchunks))
    mask = np.ones((3, 6, 6), dtype='bool')
    mask[0, 0, 0] = 0
    mask[1, 3, 2] = 0
    ds = Dataset.from_wizard(samples=data, targets=labels, chunks=chunks,
                             mask=mask, space='myspace')
    datasets['3d%s' % kind] = ds


# some additional datasets
datasets['dumb2'] = dumb_feature_binary_dataset()
datasets['dumb'] = dumb_feature_dataset()
# dataset with few invariant features
_dsinv = dumb_feature_dataset()
_dsinv.samples = np.hstack((_dsinv.samples,
                           np.zeros((_dsinv.nsamples, 1)),
                           np.ones((_dsinv.nsamples, 1))))
datasets['dumbinv'] = _dsinv

# Datasets for regressions testing
datasets['sin_modulated'] = multiple_chunks(sin_modulated, 4, 30, 1)
# use the same full for training
datasets['sin_modulated_train'] = datasets['sin_modulated']
datasets['sin_modulated_test'] = sin_modulated(30, 1, flat=True)

# simple signal for linear regressors
datasets['chirp_linear'] = multiple_chunks(chirp_linear, 6, 50, 10, 2, 0.3, 0.1)
datasets['chirp_linear_test'] = chirp_linear(20, 5, 2, 0.4, 0.1)

datasets['wr1996'] = multiple_chunks(wr1996, 4, 50)
datasets['wr1996_test'] = wr1996(50)


def saveload_warehouse():
    """Store all warehouse datasets into HDF5 and reload them.
    """
    import h5py
    from mvpa.base.hdf5 import obj2hdf, hdf2obj

    tempdir = tempfile.mkdtemp()

    # store the whole datasets warehouse in one hdf5 file
    hdf = h5py.File(os.path.join(tempdir, 'myhdf5.hdf5'), 'w')
    for d in datasets:
        obj2hdf(hdf, datasets[d], d)
    hdf.close()

    hdf = h5py.File(os.path.join(tempdir, 'myhdf5.hdf5'), 'r')
    rc_ds = {}
    for d in hdf:
        rc_ds[d] = hdf2obj(hdf[d])
    hdf.close()

    #cleanup temp dir
    shutil.rmtree(tempdir, ignore_errors=True)

    # return the reconstructed datasets (for use in datasets warehouse)
    return rc_ds


def get_random_rotation(ns, nt=None, data=None):
    """Return some random rotation (or rotation + dim reduction) matrix

    Parameters
    ----------
    ns : int
      Dimensionality of source space
    nt : int, optional
      Dimensionality of target space
    data : array, optional
      Some data (should have rank high enough) to derive
      rotation
    """
    if nt is None:
        nt = ns
    # figure out some "random" rotation
    d = max(ns, nt)
    if data is None:
        data = np.random.normal(size=(d*10, d))
    _u, _s, _vh = np.linalg.svd(data[:, :d])
    R = _vh[:ns, :nt]
    if ns == nt:
        # Test if it is indeed a rotation matrix ;)
        # Lets flip first axis if necessary
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1.0
    return R


if cfg.getboolean('tests', 'use hdf datasets', False):
    if not externals.exists('h5py'):
        raise RuntimeError(
            "Cannot perform HDF5 dump of all datasets in the warehouse, "
            "because 'h5py' is not available")

    datasets = saveload_warehouse()
    print "Replaced all dataset warehouse for HDF5 loaded alternative."
