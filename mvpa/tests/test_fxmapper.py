# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SampleGroup mapper"""


import numpy as np
from mvpa.mappers.fx import *
from mvpa.datasets.base import dataset_wizard, Dataset

from mvpa.testing.tools import *


def test_samplesgroup_mapper():
    data = np.arange(24).reshape(8,3)
    labels = [0, 1] * 4
    chunks = np.repeat(np.array((0,1)),4)

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
    from mvpa.testing.datasets import datasets
    from mvpa.measures.anova import OneWayAnova
    # TODO: might be worth creating appropriate factory
    #       help in mappers/fx
    aov = OneWayAnova(
        postproc=FxMapper('features',
                          lambda x: x / x.max(),
                          attrfx=None))
    f = aov(datasets['uni2small'])
    ok_((f.samples != 1.0).any())
    ok_(f.samples.max() == 1.0)
