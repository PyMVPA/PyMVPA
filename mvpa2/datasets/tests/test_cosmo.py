# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

"""Unit tests for CoSMoMVPA dataset (http://cosmomvpa.org)"""

from mvpa2.testing.tools import assert_raises, assert_false, assert_true, \
    assert_equal, assert_array_equal, assert_array_almost_equal, \
    with_tempfile
from mvpa2.testing import skip_if_no_external

skip_if_no_external('scipy')
from scipy.io import loadmat, savemat, matlab
from mvpa2.datasets import cosmo
from mvpa2 import pymvpa_dataroot

from mvpa2.measures.base import Measure
from mvpa2.datasets.base import Dataset
from mvpa2.mappers.fx import mean_feature

import numpy as np

from os.path import join as pathjoin

arr = np.asarray

#########################
# helper functions


def _tup2obj(tuples):
    # tup is a list of (key, value) tuples
    # returns a numpy object array with the same data
    dtypes = []
    values = []

    for k, v in tuples:
        dtypes.append((k, 'O'))
        values.append(v)

    return np.array([[tuple(values)]], dtype=np.dtype(dtypes))



def _obj2tup(obj):
    # obj is an object array from scipy Matlab data structure
    names = obj.dtype.names

    if names is None:
        # not an object array
        return None

    tups = []

    if obj.shape != (1, 1):
        raise ValueError('Unsupported non-singleton shape')

    for k in names:
        # get singleton element out
        tup = (k, obj[0, 0][k])

        tups.append(tup)

    return tups



def _create_small_mat_dataset_dict():
    '''
    Generate small dataset as represented in matlab.
    Equivalent is to do in matlab:

        ds=struct();
        ds.samples=[1 2 3; 4 5 6];
        ds.a.name='input';
        ds.a.size=[3 2 1];
        ds.fa.i=[3 2 1];
        ds.fa.j=[1 2 2];
        ds.sa.chunks=[2 2]';
        ds.sa.targets=[1 2]';
        ds.sa.labels={'yin','yan'};
        save('simple_ds.mat','-struct','ds');

    and do in python:

         ds=loadmat('simple_ds.mat')
    '''

    samples = arr([[1, 2, 3], [4, 5, 6]])
    sa = _tup2obj([('chunks', arr([[2], [2]])),
                   ('targets', arr([[1], [2]])),
                   ('labels', arr([arr(['yin'], dtype='O'),
                                   arr(['yan'], dtype='O')]))])
    fa = _tup2obj([('i', arr([[3., 2., 1.]])),
                   ('j', arr([[1., 2., 2.]]))])
    a = _tup2obj([('name', arr(arr(['input'], dtype='O'))),
                  ('size', arr([[3.], [2.], [1.]]))])

    # dictionary with these value
    return dict(samples=samples, sa=sa, fa=fa, a=a)



def _build_cell(elems):
    '''
    Put elements in a an array compatible
    with scipy's matlab cell structure.

    Necessary for recent versions of numpy
    '''
    n = len(elems)
    c = np.zeros((1, n), dtype=object)
    for i, elem in enumerate(elems):
        c[0, i] = elem

    return c



def _create_small_mat_nbrhood_dict():
    '''
    Generate small neighborhood as represented in matlab.
    Equivalent is to do in matlab:

        nbrhood=struct();
        nbrhood.neighbors={1, [1 3], [1 2 3], [2 2]};
        nbrhood.fa.k=[4 3 2 1];
        nbrhood.a.name='output';
        save('simple_nbrhood.mat','-struct','nbrhood');

    and do in python:

         nbrhood=loadmat('simple_nbrhood.mat')
    '''

    elems = [arr([[1]]), arr([[1, 3]]), arr([[1, 2, 3]]), arr([[2, 2]])]
    neighbors = _build_cell(elems)
    fa = _tup2obj([('k', arr([[4., 3., 2., 1.]]))])
    a = _tup2obj([('name', arr(arr(['output'], dtype='O')))])

    # XXX in the future we may want to use a real origin with
    # contents of .a and .fa taken from the dataset
    origin = ('unused', 0)

    return dict(neighbors=neighbors, fa=fa, a=a, origin=origin)



def _assert_ds_mat_attributes_equal(ds, m, attr_keys=('a', 'sa', 'fa')):
    # ds is a Dataset object, m a matlab-like dictionary
    for attr_k in attr_keys:
        attr_v = getattr(ds, attr_k)

        for k in attr_v.keys():
            v = attr_v[k].value
            assert_array_equal(m[attr_k][k][0, 0].ravel(), v)



def _assert_ds_less_or_equal(x, y):
    # x and y are a Dataset; x should contain a subset of
    # elements in .sa, fa, a and have the same samples as y
    # Note: no support for fancy objects such as mappers
    assert_array_equal(x.samples, y.samples)
    for label in ('a', 'fa', 'sa'):
        vx = getattr(x, label)
        vy = getattr(y, label)
        _assert_array_collectable_less_or_equal(vx, vy)



def _assert_ds_equal(x, y):
    # test for two Dataset objects to be equal
    # Note: no support for fancy objects such as mappers
    _assert_ds_less_or_equal(x, y)
    _assert_ds_less_or_equal(y, x)



def _assert_array_collectable_less_or_equal(x, y):
    # test for the keys in x to be a subset of those in y,
    # and the values corresponding to k in x being equal to those in y
    assert_true(set(x.keys()).issubset(set(y.keys())))
    for k in x.keys():
        assert_array_equal(x[k].value, y[k].value)



def _assert_array_collectable_equal(x, y):
    # test for keys and values equal in x and y
    _assert_array_collectable_less_or_equal(x, y)
    _assert_array_collectable_less_or_equal(y, x)



def _assert_subset(x, y):
    # test that first argument is a subset of the second
    assert_true(set(x).issubset(set(y)))



def _assert_set_equal(x, y):
    # test for two sets being equal
    assert_equal(set(x), set(y))



#########################
# testing functions


@with_tempfile('.mat', 'matlab_file')
def test_cosmo_dataset(fn):
    skip_if_no_external('scipy', min_version='0.8')
    mat = _create_small_mat_dataset_dict()
    ds_mat = cosmo.from_any(mat)
    savemat(fn, mat)

    # test Dataset, filename, dict in matlab form, and input from loadmat
    for input in (ds_mat, fn, mat, loadmat(fn)):

        # check dataset creation
        ds = cosmo.from_any(mat)

        # ensure dataset has expected vlaues
        assert_array_equal(ds.samples, mat['samples'])

        _assert_set_equal(ds.sa.keys(), ['chunks', 'labels', 'targets'])
        _assert_set_equal(ds.sa.keys(), ['chunks', 'labels', 'targets'])
        _assert_set_equal(ds.a.keys(), ['name', 'size'])

        assert_array_equal(ds.a.name, 'input')
        assert_array_equal(ds.a.size, [3, 2, 1])
        assert_array_equal(ds.sa.chunks, [2, 2])
        assert_array_equal(ds.sa.targets, [1, 2])
        assert_array_equal(ds.sa.labels, ['yin', 'yan'])
        assert_array_equal(ds.fa.i, [3, 2, 1])
        assert_array_equal(ds.fa.j, [1, 2, 2])

        for convert_tuples in (True, False):
            ds_copy = ds.copy(deep=True)

            if convert_tuples:
                # use dataset with tuple data
                ds_copy.a.size = tuple(ds_copy.a.size)

            # check mapping to matlab format
            mat_mapped = cosmo.map2cosmo(ds_copy)

            for m in (mat, mat_mapped):
                assert_array_equal(ds_mat.samples, m['samples'])
                _assert_ds_mat_attributes_equal(ds_mat, m)



@with_tempfile('.mat', 'matlab_file')
def test_cosmo_queryengine(fn):
    skip_if_no_external('scipy', min_version='0.8')
    nbrhood_mat = _create_small_mat_nbrhood_dict()
    neighbors = nbrhood_mat['neighbors']
    savemat(fn, nbrhood_mat)

    # test dict in matlab form, filename, and through QueryEngine loader
    for input in (nbrhood_mat, fn, cosmo.CosmoQueryEngine.from_mat(neighbors)):
        qe = cosmo.from_any(input)
        assert_array_equal(qe.ids, [0, 1, 2, 3])

        for i in qe.ids:
            nbr_fids_base0 = neighbors[0, i][0] - 1
            assert_array_equal(qe.query_byid(i), nbr_fids_base0)

        _assert_ds_mat_attributes_equal(qe, nbrhood_mat, ('fa', 'a'))



def test_cosmo_searchlight():
    ds = cosmo.from_any(_create_small_mat_dataset_dict())
    sl = cosmo.CosmoSearchlight(mean_feature(),
                                _create_small_mat_nbrhood_dict())

    ds_count = sl(ds)
    dict_count = Dataset(samples=ds_count.samples,
                         fa=dict(k=arr([4, 3, 2, 1])),
                         sa=dict((k, ds.sa[k].value) for k in ds.sa.keys()),
                         a=dict(name=['output']))

    _assert_ds_less_or_equal(dict_count, ds_count)



@with_tempfile('.h5py', 'pymvpa_file')
def test_cosmo_io_h5py(fn):
    skip_if_no_external('h5py')
    from mvpa2.base.hdf5 import h5save, h5load

    # Dataset from cosmo
    ds = cosmo.from_any(_create_small_mat_dataset_dict())
    h5save(fn, ds)
    ds_loaded = h5load(fn)

    _assert_ds_equal(ds, ds_loaded)

    # Queryengine
    qe = cosmo.from_any(_create_small_mat_nbrhood_dict())
    h5save(fn, qe)
    qe_loaded = h5load(fn)

    assert_array_equal(qe.ids, qe_loaded.ids)
    _assert_array_collectable_equal(qe.a, qe_loaded.a)
    _assert_array_collectable_equal(qe.fa, qe_loaded.fa)



def test_cosmo_exceptions():
    m = _create_small_mat_dataset_dict()
    m.pop('samples')
    assert_raises(KeyError, cosmo.cosmo_dataset, m)
    assert_raises(ValueError, cosmo.from_any, m)
    assert_raises(ValueError, cosmo.from_any, ['illegal input'])

    mapping = {1: arr([1, 2]), 2: arr([2, 0, 0])}
    qe = cosmo.CosmoQueryEngine(mapping)  # should be fine

    assert_raises(TypeError, cosmo.CosmoQueryEngine, [])
    mapping[1] = 1.5
    assert_raises(TypeError, cosmo.CosmoQueryEngine, mapping)
    mapping[1] = 'foo'
    assert_raises(TypeError, cosmo.CosmoQueryEngine, mapping)
    mapping[1] = -1
    assert_raises(TypeError, cosmo.CosmoQueryEngine, mapping)
    mapping[1] = arr([1.5, 2.1])
    assert_raises(ValueError, cosmo.CosmoQueryEngine, mapping)

    neighbors = _create_small_mat_nbrhood_dict()['neighbors']
    qe = cosmo.CosmoQueryEngine.from_mat(neighbors)  # should be fine
    neighbors[0, 0][0] = -1
    assert_raises(ValueError, cosmo.CosmoQueryEngine.from_mat, neighbors)
    neighbors[0, 0] = arr(1.5)
    assert_raises(ValueError, cosmo.CosmoQueryEngine.from_mat, neighbors)

    for illegal_nbrhood in (['fail'], cosmo.QueryEngineInterface):
        assert_raises((TypeError, ValueError),
                      lambda x: cosmo.CosmoSearchlight([], x),
                      illegal_nbrhood)



def test_cosmo_repr_and_str():
    # simple smoke test for __repr__ and __str__

    creators = (_create_small_mat_nbrhood_dict, _create_small_mat_dataset_dict)
    for creator in creators:
        obj = cosmo.from_any(creator())
        for fmt in 'rs':
            obj_str = (("%%%s" % fmt) % obj)
            assert_true(obj.__class__.__name__ in obj_str)



def test_fmri_to_cosmo():
    skip_if_no_external('nibabel')
    from mvpa2.datasets.mri import fmri_dataset
    # test exporting an fMRI dataset to CoSMoMVPA
    pymvpa_ds = fmri_dataset(
        samples=pathjoin(pymvpa_dataroot, 'example4d.nii.gz'),
        targets=[1, 2], sprefix='voxel')
    cosmomvpa_struct = cosmo.map2cosmo(pymvpa_ds)
    _assert_set_equal(cosmomvpa_struct.keys(), ['a', 'fa', 'sa', 'samples'])

    a_dict = dict(_obj2tup(cosmomvpa_struct['a']))
    mri_keys = ['imgaffine', 'voxel_eldim', 'voxel_dim']
    _assert_subset(mri_keys, a_dict.keys())

    for k in mri_keys:
        c_value = a_dict[k]
        p_value = pymvpa_ds.a[k].value

        if isinstance(p_value, tuple):
            c_value = c_value.ravel()
            p_value = np.asarray(p_value).ravel()

        assert_array_almost_equal(c_value, p_value)

    fa_dict = dict(_obj2tup(cosmomvpa_struct['fa']))
    fa_keys = ['voxel_indices']
    _assert_set_equal(fa_dict.keys(), fa_keys)
    for k in fa_keys:
        assert_array_almost_equal(fa_dict[k].T, pymvpa_ds.fa[k].value)



def test_cosmo_empty_dataset():
    ds = Dataset(np.zeros((0, 0)))
    c = cosmo.map2cosmo(ds)
    assert_equal(c['samples'].shape, (0, 0))



def test_cosmo_do_not_store_unsupported_datatype():
    ds = Dataset(np.zeros((0, 0)))

    class ArbitraryClass(object):
        pass

    ds.a['unused'] = ArbitraryClass()
    c = cosmo.map2cosmo(ds)
    assert_false('a' in c.keys())

    ds.a['foo'] = np.zeros((1,))
    c = cosmo.map2cosmo(ds)
    assert_true('a' in c.keys())
