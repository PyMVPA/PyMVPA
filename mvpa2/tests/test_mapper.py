# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for basic mappers'''

import numpy as np
# for repr
from numpy import array

from mvpa2.testing.tools import ok_, assert_raises, assert_false, assert_equal, \
        assert_true, assert_array_equal, nodebug

from mvpa2.testing.datasets import datasets
from mvpa2.mappers.flatten import FlattenMapper
from mvpa2.mappers.base import ChainMapper
from mvpa2.featsel.base import StaticFeatureSelection
from mvpa2.mappers.slicing import SampleSliceMapper, StripBoundariesSamples
from mvpa2.support.copy import copy
from mvpa2.datasets.base import Dataset
from mvpa2.base.collections import ArrayCollectable
from mvpa2.datasets.base import dataset_wizard
from mvpa2.mappers.flatten import ProductFlattenMapper

import itertools
import operator

from mvpa2.base import externals

# arbitrary ndarray subclass for testing
class myarray(np.ndarray):
    pass

def test_flatten():
    samples_shape = (2, 2, 4)
    data_shape = (4,) + samples_shape
    data = np.arange(np.prod(data_shape)).reshape(data_shape).view(myarray)
    pristinedata = data.copy()
    target = [[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
              [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
              [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]]
    target = np.array(target).view(myarray)
    index_target = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3],
                            [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3],
                            [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 0, 3],
                            [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3]])

    # test only flattening the first two dimensions
    fm_max = FlattenMapper(maxdims=2)
    fm_max.train(data)
    assert_equal(fm_max(data).shape, (4, 4, 4))

    # array subclass survives
    ok_(isinstance(data, myarray))

    # actually, there should be no difference between a plain FlattenMapper and
    # a chain that only has a FlattenMapper as the one element
    for fm in [FlattenMapper(space='voxel'),
               ChainMapper([FlattenMapper(space='voxel'),
                            StaticFeatureSelection(slice(None))])]:
        # not working if untrained
        assert_raises(RuntimeError,
                      fm.forward1,
                      np.arange(np.sum(samples_shape) + 1))

        fm.train(data)

        ok_(isinstance(fm.forward(data), myarray))
        ok_(isinstance(fm.forward1(data[2]), myarray))
        assert_array_equal(fm.forward(data), target)
        assert_array_equal(fm.forward1(data[2]), target[2])
        assert_raises(ValueError, fm.forward, np.arange(4))

        # all of that leaves that data unmodified
        assert_array_equal(data, pristinedata)

        # reverse mapping
        ok_(isinstance(fm.reverse(target), myarray))
        ok_(isinstance(fm.reverse1(target[0]), myarray))
        ok_(isinstance(fm.reverse(target[1:2]), myarray))
        assert_array_equal(fm.reverse(target), data)
        assert_array_equal(fm.reverse1(target[0]), data[0])
        assert_array_equal(fm.reverse(target[1:2]), data[1:2])
        assert_raises(ValueError, fm.reverse, np.arange(14))

        # check one dimensional data, treated as scalar samples
        oned = np.arange(5)
        fm.train(Dataset(oned))
        # needs 2D
        assert_raises(ValueError, fm.forward, oned)
        # doesn't match mapper, since Dataset turns `oned` into (5,1)
        assert_raises(ValueError, fm.forward, oned)
        assert_equal(Dataset(oned).nfeatures, 1)

        # try dataset mode, with some feature attribute
        fattr = np.arange(np.prod(samples_shape)).reshape(samples_shape)
        ds = Dataset(data, fa={'awesome': fattr.copy()})
        assert_equal(ds.samples.shape, data_shape)
        fm.train(ds)
        dsflat = fm.forward(ds)
        ok_(isinstance(dsflat, Dataset))
        ok_(isinstance(dsflat.samples, myarray))
        assert_array_equal(dsflat.samples, target)
        assert_array_equal(dsflat.fa.awesome, np.arange(np.prod(samples_shape)))
        assert_true(isinstance(dsflat.fa['awesome'], ArrayCollectable))
        # test index creation
        assert_array_equal(index_target, dsflat.fa.voxel)

        # and back
        revds = fm.reverse(dsflat)
        ok_(isinstance(revds, Dataset))
        ok_(isinstance(revds.samples, myarray))
        assert_array_equal(revds.samples, data)
        assert_array_equal(revds.fa.awesome, fattr)
        assert_true(isinstance(revds.fa['awesome'], ArrayCollectable))
        assert_false('voxel' in revds.fa)


def test_product_flatten():
    nsamples = 17
    product_name_values = [('chan', ['C1', 'C2']),
                         ('freq', np.arange(4, 20, 6)),
                         ('time', np.arange(-200, 800, 200))]

    shape = (nsamples,) + tuple(len(v) for _, v in product_name_values)

    sample_names = ['samp%d' % i for i in xrange(nsamples)]

    # generate random data in four dimensions
    data = np.random.normal(size=shape)
    ds = Dataset(data, sa=dict(sample_names=sample_names))
    for n, v in product_name_values:
        ds.a[n] = v

    # apply flattening to ds
    names, values = zip(*(product_name_values))

    flattener = ProductFlattenMapper(names)

    # test I/O (only if h5py is available)
    if externals.exists('h5py'):
        from mvpa2.base.hdf5 import h5save, h5load
        import tempfile
        import os

        fd, testfn = tempfile.mkstemp('mapper.h5py', 'test_product'); os.close(fd)
        h5save(testfn, flattener)
        flattener = h5load(testfn)
        os.unlink(testfn)

    mds = flattener(ds)

    prod = lambda x:reduce(operator.mul, x)

    # ensure the size is ok
    assert_equal(mds.shape, (nsamples,) + (prod(shape[1:]),))

    ndim = len(product_name_values)

    idxs = [range(len(v)) for v in values]
    for si in xrange(nsamples):
        for fi, p in enumerate(itertools.product(*idxs)):
            data_tup = (si,) + p

            x = mds[si, fi]

            # value should match
            assert_equal(data[data_tup], x.samples[0, 0])

            # indices should match as well
            all_idxs = tuple(x.fa['chan_freq_time_indices'].value.ravel())
            assert_equal(p, all_idxs)

            # values and indices in each dimension should match
            for i, (name, value) in enumerate(product_name_values):
                assert_equal(x.fa[name].value, value[p[i]])
                assert_equal(x.fa[name + '_indices'].value, p[i])

    dsr = flattener.reverse(mds)
    assert_equal(dsr.shape, ds.shape)


    names += ('foo',)

    flattener = ProductFlattenMapper(names)
    assert_raises(KeyError, flattener, ds)


def test_subset():
    data = np.array(
            [[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
            [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]])
    # float array doesn't work
    sm = StaticFeatureSelection(np.ones(16))
    assert_raises(IndexError, sm.forward, data)

    # full mask
    sm = StaticFeatureSelection(slice(None))
    # should not change single samples
    assert_array_equal(sm.forward(data[0:1].copy()), data[0:1])
    # or multi-samples
    assert_array_equal(sm.forward(data.copy()), data)
    sm.train(data)
    # same on reverse
    assert_array_equal(sm.reverse(data[0:1].copy()), data[0:1])
    # or multi-samples
    assert_array_equal(sm.reverse(data.copy()), data)

    # identical mappers
    sm_none = StaticFeatureSelection(slice(None))
    sm_int = StaticFeatureSelection(np.arange(16))
    sm_bool = StaticFeatureSelection(np.ones(16, dtype='bool'))
    sms = [sm_none, sm_int, sm_bool]

    # test subsets
    sids = [3, 4, 5, 6]
    bsubset = np.zeros(16, dtype='bool')
    bsubset[sids] = True
    subsets = [sids, slice(3, 7), bsubset, [3, 3, 4, 4, 6, 6, 6, 5]]
    # all test subset result in equivalent masks, hence should do the same to
    # the mapper and result in identical behavior
    for st in sms:
        for i, sub in enumerate(subsets):
            # shallow copy
            orig = copy(st)
            subsm = StaticFeatureSelection(sub)
            # should do copy-on-write for all important stuff!!
            orig += subsm
            # test if selection did its job
            if i == 3:
                # special case of multiplying features
                assert_array_equal(orig.forward1(data[0].copy()), subsets[i])
            else:
                assert_array_equal(orig.forward1(data[0].copy()), sids)

    ## all of the above shouldn't change the original mapper
    #assert_array_equal(sm.get_mask(), np.arange(16))

    # check for some bug catcher
    # no 3D input
    #assert_raises(IndexError, sm.forward, np.ones((3,2,1)))
    # no input of wrong length
    if __debug__:
        # checked only in __debug__
        assert_raises(ValueError, sm.forward, np.ones(4))
    # same on reverse
    #assert_raises(ValueError, sm.reverse, np.ones(16))
    # invalid ids
    #assert_false(subsm.is_valid_inid(-1))
    #assert_false(subsm.is_valid_inid(16))

    # intended merge failures
    fsm = StaticFeatureSelection(np.arange(16))
    assert_equal(fsm.__iadd__(None), NotImplemented)
    assert_equal(fsm.__iadd__(Dataset([2, 3, 4])), NotImplemented)


def test_subset_filler():
    sm = StaticFeatureSelection(np.arange(3))
    sm_f0 = StaticFeatureSelection(np.arange(3), filler=0)
    sm_fm1 = StaticFeatureSelection(np.arange(3), filler=-1)
    sm_fnan = StaticFeatureSelection(np.arange(3), filler=np.nan)
    data = np.arange(12).astype(float).reshape((2, -1))

    sm.train(data)
    data_forwarded = sm.forward(data)

    for m in (sm, sm_f0, sm_fm1, sm_fnan):
        m.train(data)
        assert_array_equal(data_forwarded, m.forward(data))

    data_back_fm1 = sm_fm1.reverse(data_forwarded)
    ok_(np.all(data_back_fm1[:, 3:] == -1))
    data_back_fnan = sm_fnan.reverse(data_forwarded)
    ok_(np.all(np.isnan(data_back_fnan[:, 3:])))

@nodebug(['ID_IN_REPR', 'MODULE_IN_REPR'])
def test_repr():
    # this time give mask only by its target length
    sm = StaticFeatureSelection(slice(None), space='myspace')

    # check reproduction
    sm_clone = eval(repr(sm))
    assert_equal(repr(sm_clone), repr(sm))

@nodebug(['ID_IN_REPR', 'MODULE_IN_REPR'])
def test_chainmapper():
    # the chain needs at lest one mapper
    assert_raises(ValueError, ChainMapper, [])
    # a typical first mapper is to flatten
    cm = ChainMapper([FlattenMapper()])

    # few container checks
    assert_equal(len(cm), 1)
    assert_true(isinstance(cm[0], FlattenMapper))

    # now training
    # come up with data
    samples_shape = (2, 2, 4)
    data_shape = (4,) + samples_shape
    data = np.arange(np.prod(data_shape)).reshape(data_shape)
    pristinedata = data.copy()
    target = [[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
              [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
              [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]]
    target = np.array(target)

    # if it is not trained it knows nothing
    cm.train(data)

    # a new mapper should appear when doing feature selection
    cm.append(StaticFeatureSelection(range(1, 16)))
    assert_equal(cm.forward1(data[0]).shape, (15,))
    assert_equal(len(cm), 2)
    # multiple slicing
    cm.append(StaticFeatureSelection([9, 14]))
    assert_equal(cm.forward1(data[0]).shape, (2,))
    assert_equal(len(cm), 3)

    # check reproduction
    if __debug__:
        # debug mode needs special test as it enhances the repr output
        # with module info and id() appendix for objects
        import mvpa2
        cm_clone = eval(repr(cm))
        assert_equal('#'.join(repr(cm_clone).split('#')[:-1]),
                     '#'.join(repr(cm).split('#')[:-1]))
    else:
        cm_clone = eval(repr(cm))
        assert_equal(repr(cm_clone), repr(cm))

    # what happens if we retrain the whole beast an same data as before
    cm.train(data)
    assert_equal(cm.forward1(data[0]).shape, (2,))
    assert_equal(len(cm), 3)

    # let's map something
    mdata = cm.forward(data)
    assert_array_equal(mdata, target[:, [10, 15]])
    # and back
    rdata = cm.reverse(mdata)
    # original shape
    assert_equal(rdata.shape, data.shape)
    # content as far it could be restored
    assert_array_equal(rdata[rdata > 0], data[rdata > 0])
    assert_equal(np.sum(rdata > 0), 8)

    # Lets construct a dataset with mapper assigned and see
    # if sub-selecting a feature adjusts trailing StaticFeatureSelection
    # appropriately
    ds_subsel = Dataset.from_wizard(data, mapper=cm)[:, 1]
    tail_sfs = ds_subsel.a.mapper[-1]
    assert_equal(repr(tail_sfs), 'StaticFeatureSelection(slicearg=array([14]))')

def test_sampleslicemapper():
    # this does nothing but Dataset.__getitem__ which is tested elsewhere -- but
    # at least we run it
    ds = datasets['uni2small']
    ssm = SampleSliceMapper(slice(3, 8, 2))
    sds = ssm(ds)
    assert_equal(len(sds), 3)


def test_strip_boundary():
    ds = datasets['hollow']
    ds.sa['btest'] = np.repeat([0, 1], 20)
    sn = StripBoundariesSamples('btest', 1, 2)
    sds = sn(ds)
    assert_equal(len(sds), len(ds) - 3)
    for i in [19, 20, 21]:
        assert_false(i in sds.samples.sid)

def test_transpose():
    from mvpa2.mappers.shape import TransposeMapper
    ds = Dataset(np.arange(24).reshape(2, 3, 4),
                 sa={'testsa': np.arange(2)},
                 fa={'testfa': np.arange(3)})
    tp = TransposeMapper()
    tds = tp(ds)
    assert_equal(tds.shape, (3, 2, 4))
    assert_true('testfa' in tds.sa)
    assert_true('testsa' in tds.fa)
    assert_false(tds.fa is tds.sa)
    # and back
    ttds = tp(tds)
    assert_array_equal(ttds.samples, ds.samples)
    assert_equal(ttds.sa, ds.sa)
    assert_equal(ttds.fa, ds.fa)
    # or this way
    rds = tp.reverse(tds)
    assert_array_equal(rds.samples, ds.samples)
    assert_equal(rds.sa, ds.sa)
    assert_equal(rds.fa, ds.fa)
    assert_array_equal(rds.samples, ttds.samples)
    assert_equal(rds.sa, ttds.sa)
    assert_equal(rds.fa, ttds.fa)

def test_addaxis():
    from mvpa2.mappers.shape import AddAxisMapper
    ds = Dataset(np.arange(24).reshape(2, 3, 4),
                 sa={'testsa': np.arange(2)},
                 fa={'testfa': np.arange(3)})
    ds0 = AddAxisMapper(pos=0)(ds)
    assert_array_equal(ds0.shape, (1,) + ds.shape)
    # sas have extra dimension
    assert_array_equal(ds0.sa.testsa[0], ds.sa.testsa)
    # fas are duplicated
    assert_array_equal(ds0.fa.testfa[0], ds0.fa.testfa[1])
    ds1 = AddAxisMapper(pos=1)(ds)
    assert_array_equal(ds1.shape, (2, 1, 3, 4))
    # same sample attribute
    assert_equal(ds1.sa, ds.sa)
    # fas have extra dimension
    assert_array_equal(ds1.fa.testfa[0], ds.fa.testfa)
    ds2 = AddAxisMapper(pos=2)(ds)
    assert_array_equal(ds2.shape, (2, 3, 1, 4))
    # no change to attribute collections
    assert_equal(ds2.sa, ds.sa)
    assert_equal(ds2.fa, ds.fa)
    # append an axis
    ds3 = AddAxisMapper(pos=3)(ds)
    assert_array_equal(ds3.shape, ds.shape + (1,))
    # reverse indexing
    ds_1 = AddAxisMapper(pos=-1)(ds)
    assert_array_equal(ds3.samples, ds_1.samples)
    assert_equal(ds3.sa, ds_1.sa)
    assert_equal(ds3.fa, ds_1.fa)
    # add multiple axes
    ds4 = AddAxisMapper(pos=4)(ds)
    assert_array_equal(ds4.shape, ds.shape + (1, 1))
