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
import traceback as tbm
import sys
import numpy as np
from os.path import join as pathjoin

from mvpa2 import cfg, externals
from mvpa2.datasets.base import Dataset, HollowSamples
from mvpa2.generators.partition import OddEvenPartitioner
from mvpa2.misc.data_generators import *
from mvpa2.testing.tools import reseed_rng

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

# to assure reproducibility -- lets reseed the RNG at this point
@reseed_rng()
def generate_testing_datasets(specs):
    # Lets permute upon each invocation of test, so we could possibly
    # trigger some funny cases
    nonbogus_pool = np.random.permutation([0, 1, 3, 5])

    datasets = {}

    # use a partitioner to flag odd/even samples as training and test
    ttp = OddEvenPartitioner(space='train', count=1)

    for kind, spec in specs.iteritems():
        # set of univariate datasets
        for nlabels in [ 2, 3, 4 ]:
            basename = 'uni%d%s' % (nlabels, kind)
            nonbogus_features = nonbogus_pool[:nlabels]

            dataset = normal_feature_dataset(
                nlabels=nlabels,
                nonbogus_features=nonbogus_features,
                **spec)

            # full dataset
            datasets[basename] = list(ttp.generate(dataset))[0]

        # sample 3D
        total = 2*spec['perlabel']
        nchunks = spec['nchunks']
        data = np.random.standard_normal(( total, 3, 6, 6 ))
        labels = np.concatenate( ( np.repeat( 0, spec['perlabel'] ),
                                  np.repeat( 1, spec['perlabel'] ) ) )
        data[:, 1, 0, 0] += 2*labels           # add some signal
        chunks = np.asarray(range(nchunks)*(total//nchunks))
        mask = np.ones((3, 6, 6), dtype='bool')
        mask[0, 0, 0] = 0
        mask[1, 3, 2] = 0
        ds = Dataset.from_wizard(samples=data, targets=labels, chunks=chunks,
                                 mask=mask, space='myspace')
        # and to stress tests on manipulating sa/fa possibly containing
        # attributes of dtype object
        ds.sa['test_object'] = [['a'], [1, 2]] * (ds.nsamples//2)
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
    datasets['sin_modulated'] = list(ttp.generate(multiple_chunks(sin_modulated, 4, 30, 1)))[0]
    # use the same full for training
    datasets['sin_modulated_train'] = datasets['sin_modulated']
    datasets['sin_modulated_test'] = sin_modulated(30, 1, flat=True)

    # simple signal for linear regressors
    datasets['chirp_linear'] = multiple_chunks(chirp_linear, 6, 50, 10, 2, 0.3, 0.1)
    datasets['chirp_linear_test'] = chirp_linear(20, 5, 2, 0.4, 0.1)

    datasets['wr1996'] = multiple_chunks(wr1996, 4, 50)
    datasets['wr1996_test'] = wr1996(50)

    datasets['hollow'] = Dataset(HollowSamples((40,20)),
                                 sa={'targets': np.tile(['one', 'two'], 20)})

    return datasets

# avoid treating it as a test by nose
generate_testing_datasets.__test__ = False

def saveload_warehouse():
    """Store all warehouse datasets into HDF5 and reload them.
    """
    import h5py
    from mvpa2.base.hdf5 import obj2hdf, hdf2obj

    tempdir = tempfile.mkdtemp()

    # store the whole datasets warehouse in one hdf5 file
    hdf = h5py.File(pathjoin(tempdir, 'myhdf5.hdf5'), 'w')
    for d in datasets:
        obj2hdf(hdf, datasets[d], d)
    hdf.close()

    hdf = h5py.File(pathjoin(tempdir, 'myhdf5.hdf5'), 'r')
    rc_ds = {}
    for d in hdf:
        rc_ds[d] = hdf2obj(hdf[d])
    hdf.close()

    #cleanup temp dir
    shutil.rmtree(tempdir, ignore_errors=True)

    # return the reconstructed datasets (for use in datasets warehouse)
    return rc_ds


datasets = generate_testing_datasets(specs)

if cfg.getboolean('tests', 'use hdf datasets', False):
    if not externals.exists('h5py'):
        raise RuntimeError(
            "Cannot perform HDF5 dump of all datasets in the warehouse, "
            "because 'h5py' is not available")

    datasets = saveload_warehouse()
    print "Replaced all dataset warehouse for HDF5 loaded alternative."
