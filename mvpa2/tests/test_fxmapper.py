# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SampleGroup mapper"""

from mvpa2.testing import sweepargs
from mvpa2.testing.datasets import datasets
from mvpa2.measures.anova import OneWayAnova

import numpy as np
from mvpa2.mappers.fx import *
from mvpa2.datasets.base import dataset_wizard, Dataset

from mvpa2.testing.tools import *

def test_samplesgroup_mapper():
    data = np.arange(24).reshape(8, 3)
    labels = [0, 1] * 4
    chunks = np.repeat(np.array((0, 1)), 4)

    # correct results
    csamples = [[3, 4, 5], [6, 7, 8], [15, 16, 17], [18, 19, 20]]
    clabels = [0, 1, 0, 1]
    cchunks = [0, 0, 1, 1]

    ds = dataset_wizard(samples=data, targets=labels, chunks=chunks)
    # add some feature attribute -- just to check
    ds.fa['checker'] = np.arange(3)
    ds.init_origids('samples')

    m = mean_group_sample(['targets', 'chunks'])
    mds = m.forward(ds)
    assert_array_equal(mds.samples, csamples)
    # FAs should simply remain the same
    assert_array_equal(mds.fa.checker, np.arange(3))

    # now without grouping
    m = mean_sample()
    # forwarding just the samples should yield the same result
    assert_array_equal(m.forward(ds.samples),
                       m.forward(ds).samples)

    # directly apply to dataset
    # using untrained mapper
    m = mean_group_sample(['targets', 'chunks'])
    mapped = ds.get_mapped(m)

    assert_equal(mapped.nsamples, 4)
    assert_equal(mapped.nfeatures, 3)
    assert_array_equal(mapped.samples, csamples)
    assert_array_equal(mapped.targets, clabels)
    assert_array_equal(mapped.chunks, cchunks)
    # make sure origids get regenerated
    assert_array_equal([s.count('+') for s in mapped.sa.origids], [1] * 4)

    # disbalanced dataset -- lets remove 0th sample so there is no target
    # 0 in 0th chunk
    ds_ = ds[[0, 1, 3, 5]]
    mapped = ds_.get_mapped(m)
    ok_(len(mapped) == 3)
    ok_(not None in mapped.sa.origids)

    # with such a dataset we should get identical results if we order groups
    # by their occurence
    mo = mean_group_sample(['targets', 'chunks'], order='occurrence')
    mappedo = ds.get_mapped(mo)
    assert_array_equal(mappedo.samples, csamples)
    assert_array_equal(mappedo.targets, clabels)
    assert_array_equal(mappedo.chunks, cchunks)

    # but if we would get different result if we swap order
    # of specified uattrs: now first targets would be groupped
    # and only then chunks:
    mr = mean_group_sample(['chunks', 'targets'])
    mappedr = ds.get_mapped(mr)
    # which effectively swaps two comparison targets in this fake dataset
    assert_array_equal(mappedr.targets, cchunks)
    assert_array_equal(mappedr.chunks, clabels)

def test_samplesgroup_mapper_test_order_occurrence():
    data = np.arange(8)[:, None]
    ds = dataset_wizard(samples=data,
                        targets=[1, 0]*4,
                        chunks=[0]*4 + [1]*4)

    m = mean_group_sample(['targets', 'chunks'], order='occurrence')
    assert_true('order=' in repr(m))

    mds = ds.get_mapped(m)

    assert_array_equal(mds.sa.targets, [1, 0] * 2)
    assert_array_equal(mds.sa.chunks, [0]*2 + [1]*2)
    assert_array_equal(mds.samples[:, 0], [1, 2, 5, 6])

    # and if we ordered as 'uattrs' (default)
    m = mean_group_sample(['targets', 'chunks'])
    assert_false('order=' in repr(m))
    mds = ds.get_mapped(m)

    assert_array_equal(mds.sa.targets, [0, 1] * 2)
    assert_array_equal(mds.sa.chunks, [0]*2 + [1]*2)
    assert_array_equal(mds.samples[:, 0], [2, 1, 6, 5])

def test_featuregroup_mapper():
    ds = Dataset(np.arange(24).reshape(3,8))
    ds.fa['roi'] = [0, 1] * 4
    # just to check
    ds.sa['chunks'] = np.arange(3)

    # correct results
    csamples = [[3, 4], [11, 12], [19, 20]]
    croi = [0, 1]
    cchunks = np.arange(3)

    m = mean_group_feature(['roi'])
    mds = m.forward(ds)
    assert_equal(mds.shape, (3, 2))
    assert_array_equal(mds.samples, csamples)
    assert_array_equal(mds.fa.roi, np.unique([0, 1] * 4))
    # FAs should simply remain the same
    assert_array_equal(mds.sa.chunks, np.arange(3))

    # now without grouping
    m = mean_feature()
    # forwarding just the samples should yield the same result
    assert_array_equal(m.forward(ds.samples),
                       m.forward(ds).samples)

    # And when operating on a dataset with >1D samples, then operate
    # only across "features", i.e. 1st dimension
    ds = Dataset(np.arange(24).reshape(3,2,2,2))
    mapped = ds.get_mapped(m)
    assert_array_equal(m.forward(ds.samples),
                       mapped.samples)
    assert_array_equal(mapped.samples.shape, (3, 2, 2))
    assert_array_equal(mapped.samples, np.mean(ds.samples, axis=1))
    # and still could map back? ;) not ATM, so just to ensure consistency
    assert_raises(NotImplementedError,
                  mapped.a.mapper.reverse, mapped.samples)
    # but it should also work with standard 2d sample arrays
    ds = Dataset(np.arange(24).reshape(3,8))
    mapped = ds.get_mapped(m)
    assert_array_equal(mapped.samples.shape, (3, 1))


def test_fxmapper():
    origdata = np.arange(24).reshape(3,8)
    ds = Dataset(origdata.copy())
    ds.samples *= -1

    # test a mapper that doesn't change the shape
    # it shouldn't mapper along with axis it is applied
    m_s = FxMapper('samples', np.absolute)
    m_f = FxMapper('features', np.absolute)
    a_m = absolute_features()
    assert_array_equal(m_s.forward(ds), origdata)
    assert_array_equal(a_m.forward(ds), origdata)
    assert_array_equal(m_s.forward(ds), m_f.forward(ds))


def test_features01():
    # TODO: might be worth creating appropriate factory
    #       help in mappers/fx
    aov = OneWayAnova(
        postproc=FxMapper('features',
                          lambda x: x / x.max(),
                          attrfx=None))
    f = aov(datasets['uni2small'])
    ok_((f.samples != 1.0).any())
    ok_(f.samples.max() == 1.0)

@sweepargs(f=dir(np))
def test_fx_native_calls(f):
    import inspect

    ds = datasets['uni2small']
    if f in ['size', 'rollaxis']:
        # really not appropriate ones here to test
        return
    try:
        f_ = getattr(np, f)
        if 'axis' != inspect.getargs(f_.__code__).args[1]:
            # if 'axis' is not the 2nd arg -- skip
            return
    except:
        return

    # so we got a function which has 'axis' arugment
    for naxis in (0, 1): # check on both axes
        for do_group in (False, True): # test with
                                       # groupping and without
            kwargs = dict(attrfx='merge')
            if do_group:
                if naxis == 0:
                    kwargs['uattrs'] = ('targets', 'chunks')
                else:
                    kwargs['uattrs'] = ('nonbogus_targets',)

            axis = ('samples', 'features')[naxis]
            def custom(data):
                """So we could enforce apply_along_axis
                """
                # always 0 since it would be the job for apply_along_axis
                return f_(data, axis=0)
            try:
                m2 = FxMapper(axis, custom, **kwargs)
                dsm2 = ds.get_mapped(m2)
            except Exception, e:
                # We assume that our previous implementation should work ;-)
                continue

            m1 = FxMapper(axis, f_, **kwargs)
            dsm1 = ds.get_mapped(m1)

            assert_objectarray_equal(dsm1.samples, dsm2.samples)
            assert_objectarray_equal(dsm1.targets, dsm2.targets)
            assert_objectarray_equal(dsm1.chunks, dsm2.chunks)
            assert_objectarray_equal(dsm1.fa.nonbogus_targets, dsm2.fa.nonbogus_targets)

def test_uniquemerge2literal():
    from mvpa2.mappers.fx import _uniquemerge2literal
    assert_equal(_uniquemerge2literal(range(3)), ['0+1+2'])
    assert_equal(_uniquemerge2literal(np.arange(6).reshape(2,3)), ['[0 1 2]+[3 4 5]'])
    assert_array_equal(_uniquemerge2literal([[2,3,4]]), [[2, 3, 4]])
    assert_array_equal(_uniquemerge2literal([[2,3,4],[2,3,4]]), [[2, 3, 4]])
    assert_equal(_uniquemerge2literal([2,2,2]), [2])
    assert_array_equal(_uniquemerge2literal(['L1', 'L1']), ['L1'])
