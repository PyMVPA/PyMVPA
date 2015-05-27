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

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets, saveload_warehouse

skip_if_no_external('h5py')
import h5py

from glob import glob
import os
from os.path import join as opj, exists, realpath
import sys
import tempfile

from mvpa2.base.dataset import AttrDataset, save
from mvpa2.base.hdf5 import h5save, h5load, obj2hdf, HDF5ConversionError
from mvpa2.datasets.sources import load_example_fmri_dataset
from mvpa2.mappers.fx import mean_sample
from mvpa2.mappers.boxcar import BoxcarMapper
from mvpa2.misc.support import SmartVersion

from mvpa2 import pymvpa_dataroot
from mvpa2.testing import sweepargs
from mvpa2.testing.regress import get_testing_fmri_dataset_filename

class HDFDemo(object):
    pass

class CustomOldStyle:
    pass

@nodebug(['ID_IN_REPR', 'MODULE_IN_REPR'])
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
            if __debug__:
                # debug mode needs special test as it enhances the repr output
                # with module info and id() appendix for objects
                assert_equal('#'.join(repr(ds.a.mapper).split('#')[:-1]),
                             '#'.join(repr(ds2.a.mapper).split('#')[:-1]))
            else:
                assert_equal(repr(ds.a.mapper), repr(ds2.a.mapper))


def test_h5py_dataset_typecheck():
    ds = datasets['uni2small']

    fd, fpath = tempfile.mkstemp('mvpa', 'test'); os.close(fd)
    fd, fpath2 = tempfile.mkstemp('mvpa', 'test'); os.close(fd)

    h5save(fpath2, [[1, 2, 3]])
    assert_raises(ValueError, AttrDataset.from_hdf5, fpath2)
    # this one just catches if there is such a group
    assert_raises(ValueError, AttrDataset.from_hdf5, fpath2, name='bogus')

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
    os.remove(fpath2)


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
    if not externals.exists('nibabel'):
        raise SkipTest
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

def test_various_special_cases():
    # 0d object ndarray
    f = tempfile.NamedTemporaryFile()
    a = np.array(0, dtype=object)
    h5save(f.name, a)
    a_ = h5load(f.name)
    ok_(a == a_)
    # slice
    h5save(f.name, slice(2,5,3))
    sl = h5load(f.name)
    ok_(sl == slice(2,5,3))

def test_class_oldstyle():
    # AttributeError: CustomOld instance has no attribute '__reduce__'

    # old style classes do not define reduce -- sure thing we might
    # not need to support them at all, but then some meaningful
    # exception should be thrown
    co = CustomOldStyle()
    co.v = 1
    f = tempfile.NamedTemporaryFile()
    assert_raises(HDF5ConversionError, save, co, f.name, compression='gzip')

def test_locally_defined_class():
    # cannot store locally defined classes
    class Custom(object):
        pass
    c = Custom()
    f = tempfile.NamedTemporaryFile()
    assert_raises(HDF5ConversionError, h5save, f.name, c, compression='gzip')

def test_dataset_without_chunks():
    #  ValueError: All chunk dimensions must be positive (Invalid arguments to routine: Out of range)
    # MH: This is not about Dataset chunks, but about an empty samples array
    f = tempfile.NamedTemporaryFile()
    ds = AttrDataset([8], a=dict(custom=1))
    save(ds, f.name, compression='gzip')
    ds_loaded = h5load(f.name)
    ok_(ds_loaded.a.custom == ds.a.custom)

def test_recursion():
    obj = range(2)
    obj.append(HDFDemo())
    obj.append(obj)
    f = tempfile.NamedTemporaryFile()
    h5save(f.name, obj)
    lobj = h5load(f.name)
    assert_equal(obj[:2], lobj[:2])
    assert_equal(type(obj[2]), type(lobj[2]))
    ok_(obj[3] is obj)
    ok_(lobj[3] is lobj)

@with_tempfile()
def test_h5save_mkdir(dirname):
    # create deeper directory name
    filename = os.path.join(dirname, 'a', 'b', 'c', 'test_file.hdf5')
    assert_raises(IOError, h5save, filename, {}, mkdir=False)

    # And create by default
    h5save(filename, {})
    ok_(os.path.exists(filename))
    d = h5load(filename)
    assert_equal(d, {})

    # And that we can still just save into a file in current directory
    # Let's be safe and assure going back to the original directory
    cwd = os.getcwd()
    try:
        os.chdir(dirname)
        h5save("TEST.hdf5", [1,2,3])
    finally:
        os.chdir(cwd)

def test_state_cycle_with_custom_reduce():
    # BoxcarMapper has a custom __reduce__ implementation . The 'space'
    # setting will only survive a svae/load cycle if the state is correctly
    # handle for custom reduce iplementations.
    bm = BoxcarMapper([0], 1, space='boxy')
    f = tempfile.NamedTemporaryFile()
    h5save(f.name, bm)
    bm_rl = h5load(f.name)
    assert_equal(bm_rl.get_space(), 'boxy')

def test_store_metaclass_types():
    f = tempfile.NamedTemporaryFile()
    from mvpa2.kernels.base import Kernel
    allowedtype=Kernel
    h5save(f.name, allowedtype)
    lkrn = h5load(f.name)
    assert_equal(lkrn, Kernel)
    assert_equal(lkrn.__metaclass__, Kernel.__metaclass__)

def test_state_setter_getter():
    # make sure the presence of custom __setstate__, __getstate__ methods
    # is honored -- numpy's RNGs have it
    from numpy.random.mtrand import RandomState
    f = tempfile.NamedTemporaryFile()
    r = RandomState()
    h5save(f.name, r)
    rl = h5load(f.name)
    rl_state = rl.get_state()
    for i, v in enumerate(r.get_state()):
        assert_array_equal(v, rl_state[i])


@sweepargs(obj=(
    # simple 1d -- would have worked before as well
    np.array([{'d': np.empty(shape=(2,3))}], dtype=object),
    # 2d -- before fix would be reconstructed incorrectly
    np.array([[{'d': np.empty(shape=(2,3))}]], dtype=object),
    # a bit more elaborate storage
    np.array([[{'d': np.empty(shape=(2,3)),
                'k': 33}]*2]*3, dtype=object),
    # Swaroop's use-case
    AttrDataset(np.array([{'d': np.empty(shape=(2,3))}], dtype=object)),
    # as it would be reconstructed before the fix -- obj array of obj arrays
    np.array([np.array([{'d': np.empty(shape=(2,3))}], dtype=object)],
             dtype=object),
    np.array([],dtype='int64'),
    ))
def test_save_load_object_dtype_ds(obj=None):
    """Test saving of custom object ndarray (GH #84)
    """
    aobjf = np.asanyarray(obj).flatten()

    if not aobjf.size and externals.versions['hdf5'] < '1.8.7':
        raise SkipTest("Versions of hdf5 before 1.8.7 have problems with empty arrays")

    # print obj, obj.shape
    f = tempfile.NamedTemporaryFile()

    # save/reload
    obj_ = saveload(obj, f.name)

    # and compare
    # neh -- not versatile enough
    #assert_objectarray_equal(np.asanyarray(obj), np.asanyarray(obj_))

    assert_array_equal(obj.shape, obj_.shape)
    assert_equal(type(obj), type(obj_))
    # so we could test both ds and arrays
    aobjf_ = np.asanyarray(obj_).flatten()
    # checks if having just array above
    if aobjf.size:
        assert_equal(type(aobjf[0]), type(aobjf_[0]))
        assert_array_equal(aobjf[0]['d'], aobjf_[0]['d'])


_python_objs = [
    # lists
    [1, 2], [],
    # tuples
    (1, 2), tuple(),
    # pure Python sets
    set([1,2]), set(), set([None]), set([tuple()]),
    # Our SmartVersion which was missing __reduce__
    SmartVersion("0.1"),
    ]
import collections
_python_objs.append([collections.deque([1,2])])
if hasattr(collections, 'OrderedDict'):
    _python_objs.append([collections.OrderedDict(),
                         collections.OrderedDict(a9=1, a0=2)])
if hasattr(collections, 'Counter'):
    _python_objs.append([collections.Counter({'red': 4, 'blue': 2})])
if hasattr(collections, 'namedtuple') and sys.version_info > (2, 7, 4):
    # only test this on >2.7.4, because of this:
    # http://bugs.python.org/issue15535
    _NamedTuple = collections.namedtuple('_NamedTuple', ['red', 'blue'])
    # And the one with non-matching name
    _NamedTuple_ = collections.namedtuple('NamedTuple', ['red', 'blue'])
    _python_objs.extend([_NamedTuple(4, 2),
                         _NamedTuple_(4, 2),])
if hasattr(collections, 'OrderedDict'):
    _python_objs.extend([collections.OrderedDict(a=1, b=2)])


@sweepargs(obj=_python_objs)
def test_save_load_python_objs(obj):
    """Test saving objects of various types
    """
    # print obj, obj.shape
    f = tempfile.NamedTemporaryFile()

    # save/reload
    try:
        h5save(f.name, obj)
    except Exception, e:
        raise AssertionError("Failed to h5save %s: %s" % (obj, e))
    try:
        obj_ = h5load(f.name)
    except Exception, e:
        raise AssertionError("Failed to h5load %s: %s" % (obj, e))
    assert_equal(type(obj), type(obj_))
    assert_equal(obj, obj_)

def saveload(obj, f, backend='hdf5'):
    """Helper to save/load using some of tested backends
    """
    if backend == 'hdf5':
        h5save(f, obj)
        #import os; os.system('h5dump %s' % f)
        obj_ = h5load(f)
    else:
        #check pickle -- does it correctly
        import cPickle
        with open(f, 'w') as f_:
            cPickle.dump(obj, f_)
        with open(f) as f_:
            obj_ = cPickle.load(f_)
    return obj_

# Test some nasty nested constructs of mutable beasts
_nested_d = {0: 2}
_nested_d[1] = {
    0: {3: 4}, # to ease comprehension of the dump
    1: _nested_d}
_nested_d[1][2] = ['crap', _nested_d]   # 3rd level of nastiness

_nested_l = [2, None]
_nested_l[1] = [{3: 4}, _nested_l, None]
_nested_l[1][2] = ['crap', _nested_l]   # 3rd level of nastiness

@sweepargs(obj=[_nested_d, _nested_l])
@sweepargs(backend=['hdf5', 'pickle'])
@with_tempfile()
def test_nested_obj(f, backend, obj):
    ok_(obj[1][1] is obj)
    obj_ = saveload(obj, f, backend=backend)
    assert_equal(obj_[0], 2)
    assert_equal(obj_[1][0], {3: 4})
    ok_(obj_[1][1] is obj_)
    ok_(obj_[1][1] is not obj)  # nobody does teleportation

    # 3rd level
    ok_(obj_[1][2][1] is obj_)

_nested_a = np.array([1, 2], dtype=object)
_nested_a[1] = {1: 0, 2: _nested_a}

@sweepargs(a=[_nested_a])
@sweepargs(backend=['hdf5', 'pickle'])
@with_tempfile()
def test_nested_obj_arrays(f, backend, a):
    assert_equal(a.dtype, np.object)
    a_ = saveload(a, f, backend=backend)
    # import pydb; pydb.debugger()
    ok_(a_[1][2] is a_)

@sweepargs(backend=['hdf5','pickle'])
@with_tempfile()
def test_ca_col(f, backend):
    from mvpa2.base.state import ConditionalAttributesCollection, ConditionalAttribute
    c1 = ConditionalAttribute(name='ca1', enabled=True)
    #c2 = ConditionalAttribute(name='test2', enabled=True)
    col = ConditionalAttributesCollection([c1], name='whoknows')
    col.ca1 = col # {0: c1, 1: [None, col]}  # nest badly
    assert_true(col.ca1 is col)
    col_ = saveload(col, f, backend=backend)
    # seems to work niceish with pickle
    #print col_, col_.ca1, col_.ca1.ca1, col_.ca1.ca1.ca1
    assert_true(col_.ca1.ca1 is col_.ca1)
    # but even there top-level assignment test fails, which means it creates two
    # instances
    if backend != 'pickle':
        assert_true(col_.ca1 is col_)

# regression tests for datasets which have been previously saved

def test_reg_load_hyperalignment_example_hdf5():
    from mvpa2 import pymvpa_datadbroot
    filepath = os.path.join(pymvpa_datadbroot,
                        'hyperalignment_tutorial_data',
                        'hyperalignment_tutorial_data.hdf5.gz')
    if not os.path.exists(filepath):
        raise SkipTest("No hyperalignment tutorial data available under %s" %
                       filepath)
    ds_all = h5load(filepath)

    ds = ds_all[0]
    # First mapper was a FlattenMapper
    flat_mapper = ds.a.mapper[0]
    assert_equal(flat_mapper.shape, (61, 73, 61))
    assert_equal(flat_mapper.pass_attr, None)
    assert_false('ERROR' in str(flat_mapper))
    ds_reversed = ds.a.mapper.reverse(ds)
    assert_equal(ds_reversed.shape, (len(ds),) + flat_mapper.shape)

@with_tempfile()
def test_save_load_FlattenMapper(f):
    from mvpa2.mappers.flatten import FlattenMapper
    fm = FlattenMapper()
    ds = datasets['3dsmall']
    ds_ = fm(ds)
    ds_r = fm.reverse(ds_)
    fm_ = saveload(fm, f)
    assert_equal(fm_.shape, fm.shape)

@with_tempfile()
def test_versions(f):
    h5save(f, [])
    hdf = h5py.File(f, 'r')
    assert_equal(hdf.attrs.get('__pymvpa_hdf5_version__'), '2')
    assert_equal(hdf.attrs.get('__pymvpa_version__'), mvpa2.__version__)


def test_present_fmri_dataset():
    # just a helper to signal if we have any of those available
    f = get_testing_fmri_dataset_filename()
    if not os.path.exists(f):
        raise SkipTest("Absent %s. Verify that you got submodule" % f)


test_files = glob(opj(pymvpa_dataroot, 'testing', 'fmri_dataset', '*.hdf5'))


@sweepargs(testfile=test_files)
@with_tempfile(suffix=".nii.gz")
def test_regress_fmri_dataset(tempfile=None, testfile=None):
    if not externals.exists('nibabel'):
        raise SkipTest("can't test without nibabel")

    # verify that we have actual load
    if not (exists(testfile) and exists(realpath(testfile))):
        raise SkipTest("File %s seems to be missing -- 'git annex get .' "
                       "to fetch all test files first" % testfile)
    # Still might be a direct mode, or windows -- so lets check the size
    if os.stat(testfile).st_size < 1000:
        raise SkipTest("File %s seems to be small/empty -- 'git annex get .' "
                       "to fetch all test files first" % testfile)

    from mvpa2.datasets.mri import map2nifti

    ds = h5load(testfile)  # load previously generated dataset
    # rudimentary checks that data was loaded correctly
    assert_equal(np.sum(ds), 11444)
    assert_equal(sorted(ds.sa.keys()),
                 ['chunks', 'targets', 'time_coords', 'time_indices'])
    assert_equal(sorted(ds.fa.keys()), ['voxel_indices'])

    # verify that map2nifti works whenever version of nibabel on the system
    # greater or equal that one it was saved with:
    if externals.versions['nibabel'] >= ds.a.versions['nibabel']:
        # test that we can get str of the niftihdr:
        # to avoid such issues as https://github.com/PyMVPA/PyMVPA/issues/278
        hdr_str = str(ds.a.imghdr)
        assert(hdr_str != "")
        ds_ni = map2nifti(ds)
        # verify that we can store generated nifti to a file
        ds_ni.to_filename(tempfile)
        assert(os.path.exists(tempfile))
    else:
        raise SkipTest(
            "Our version of nibabel %s is older than the one file %s was saved "
            "with: %s" % (externals.versions['nibabel'],
                          testfile,
                          ds.a.versions['nibabel']))
