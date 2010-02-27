# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for HDF5 converter'''

import numpy as np

from mvpa.testing import *
from mvpa.testing.datasets import datasets, saveload_warehouse

skip_if_no_external('h5py')
import h5py

import os
import tempfile

from mvpa.base.dataset import AttrDataset
from mvpa.base.hdf5 import h5save, h5load, obj2hdf
from mvpa.misc.data_generators import load_example_fmri_dataset
from mvpa.mappers.fx import mean_sample



def test_h5py_datasets():
    # this one stores and reloads all datasets in the warehouse
    rc_ds = saveload_warehouse()

    # global checks
    assert_equal(len(datasets), len(rc_ds))
    assert_equal(sorted(datasets.keys()), sorted(rc_ds.keys()))
    # check each one
    for d in datasets:
        ds = datasets[d]
        ds2 = rc_ds[d]
        assert_array_equal(ds.samples, ds2.samples)
        # we can check all sa and fa attrs
        for attr in ds.sa:
            assert_array_equal(ds.sa[attr].value, ds2.sa[attr].value)
        for attr in ds.fa:
            assert_array_equal(ds.fa[attr].value, ds2.fa[attr].value)
        # with datasets attributes it is more difficult, but we'll do some
        assert_equal(len(ds.a), len(ds2.a))
        assert_equal(sorted(ds.a.keys()), sorted(ds2.a.keys()))
        if 'mapper' in ds.a:
            # since we have no __equal__ do at least some comparison
            assert_equal(repr(ds.a.mapper), repr(ds2.a.mapper))

def test_h5py_dataset_typecheck():
    ds = datasets['uni2small']

    _, fpath = tempfile.mkstemp('mvpa', 'test')

    h5save(fpath, [[1, 2, 3]])
    assert_raises(ValueError, AttrDataset.from_hdf5, fpath)
    # this one just catches if there is such a group
    assert_raises(ValueError, AttrDataset.from_hdf5, fpath, name='bogus')

    hdf = h5py.File(fpath, 'w')
    ds = AttrDataset([1, 2, 3])
    obj2hdf(hdf, ds, name='non-bogus')
    obj2hdf(hdf, [1, 2, 3], name='bogus')
    hdf.close()

    assert_raises(ValueError, AttrDataset.from_hdf5, fpath, name='bogus')
    ds_loaded = AttrDataset.from_hdf5(fpath, name='non-bogus')
    assert_array_equal(ds, ds_loaded)   # just to do smth useful with ds ;)

    # cleanup and ignore stupidity
    os.remove(fpath)


def test_matfile_v73_compat():
    mat = h5load(os.path.join(pymvpa_dataroot, 'v73.mat'))
    assert_equal(len(mat), 2)
    assert_equal(sorted(mat.keys()), ['x', 'y'])
    assert_array_equal(mat['x'], np.arange(6)[None].T)
    assert_array_equal(mat['y'], np.array([(1,0,1)], dtype='uint8').T)


def test_directaccess():
    f = tempfile.NamedTemporaryFile()
    h5save(f.name, 'test')
    assert_equal(h5load(f.name), 'test')
    f.close()
    f = tempfile.NamedTemporaryFile()
    h5save(f.name, datasets['uni4medium'])
    assert_array_equal(h5load(f.name).samples,
                       datasets['uni4medium'].samples)


def test_function_ptrs():
    ds = load_example_fmri_dataset()
    # add a mapper with a function ptr inside
    ds = ds.get_mapped(mean_sample())
    f = tempfile.NamedTemporaryFile()
    h5save(f.name, ds)
    ds_loaded = h5load(f.name)
    fresh = load_example_fmri_dataset().O
    # check that the reconstruction function pointer in the FxMapper points
    # to the right one
    assert_array_equal(ds_loaded.a.mapper.forward(fresh),
                        ds.samples)


#@sweepargs(lrn=clfswh[:] + regrswh[:])
def _test_h5py_clfs():
    # YOH: For now just to see which ones work (could be stored/loaded)
    #      Later on to become a proper valid test
    from mvpa.clfs.warehouse import clfswh, regrswh

    for lrn in clfswh[:] + regrswh[:]:
        print lrn
        f = tempfile.NamedTemporaryFile()
        try:
            h5save(f.name, lrn)
            lrn_ = h5load(f.name)
            print "ok: %s" % lrn_
        except Exception, e:
            #raise AssertionError,
            print "Failed to store %s due to %r" % (lrn, e)
