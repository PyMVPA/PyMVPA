# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SampleGroup mapper"""


import numpy as N
from mvpa.mappers.fx import *
from mvpa.datasets.base import dataset_wizard, Dataset

from mvpa.testing.tools import assert_equal, assert_raises,  assert_array_equal

def test_samplesgroup_mapper():
    data = N.arange(24).reshape(8,3)
    labels = [0, 1] * 4
    chunks = N.repeat(N.array((0,1)),4)

    # correct results
    csamples = [[3, 4, 5], [6, 7, 8], [15, 16, 17], [18, 19, 20]]
    clabels = [0, 1, 0, 1]
    cchunks = [0, 0, 1, 1]

    ds = dataset_wizard(samples=data, targets=labels, chunks=chunks)
    # add some feature attribute -- just to check
    ds.fa['checker'] = N.arange(3)
    ds.init_origids('samples')

    m = mean_group_sample(['targets', 'chunks'])
    mds = m.forward(ds)
    assert_array_equal(mds.samples, csamples)
    # FAs should simply remain the same
    assert_array_equal(mds.fa.checker, N.arange(3))

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


def test_featuregroup_mapper():
    ds = Dataset(N.arange(24).reshape(3,8))
    ds.fa['roi'] = [0, 1] * 4
    # just to check
    ds.sa['chunks'] = N.arange(3)

    # correct results
    csamples = [[3, 4], [11, 12], [19, 20]]
    croi = [0, 1]
    cchunks = N.arange(3)

    m = mean_group_feature(['roi'])
    mds = m.forward(ds)
    assert_equal(mds.shape, (3, 2))
    assert_array_equal(mds.samples, csamples)
    assert_array_equal(mds.fa.roi, N.unique([0, 1] * 4))
    # FAs should simply remain the same
    assert_array_equal(mds.sa.chunks, N.arange(3))


def test_fxmapper():
    origdata = N.arange(24).reshape(3,8)
    ds = Dataset(origdata.copy())
    ds.samples *= -1

    # test a mapper that doesn't change the shape
    # it shouldn't mapper along with axis it is applied
    m_s = FxMapper('samples', N.absolute)
    m_f = FxMapper('features', N.absolute)
    a_m = absolute_features()
    assert_array_equal(m_s.forward(ds), origdata)
    assert_array_equal(a_m.forward(ds), origdata)
    assert_array_equal(m_s.forward(ds), m_f.forward(ds))
