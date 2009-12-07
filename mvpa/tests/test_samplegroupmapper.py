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
from numpy.testing import assert_array_equal
from nose.tools import assert_equal
from mvpa.mappers.samplegroup import SampleGroupMapper
from mvpa.datasets.base import dataset

def test_samplesgroup_mapper():
    data = N.arange(24).reshape(8,3)
    labels = [0, 1] * 4
    chunks = N.repeat(N.array((0,1)),4)

    # correct results
    csamples = [[3, 4, 5], [6, 7, 8], [15, 16, 17], [18, 19, 20]]
    clabels = [0, 1, 0, 1]
    cchunks = [0, 0, 1, 1]

    ds = dataset(samples=data, labels=labels, chunks=chunks)
    ds.init_origids('samples')

    # default behavior
    m = SampleGroupMapper(['labels', 'chunks'])

    # train mapper first
    m.train(ds)
    assert_array_equal(m.forward(ds).samples, csamples)

    # directly apply to dataset
    # using untrained mapper!
    mapped = ds.get_mapped(SampleGroupMapper(['labels', 'chunks']))

    assert_equal(mapped.nsamples, 4)
    assert_equal(mapped.nfeatures, 3)
    assert_array_equal(mapped.samples, csamples)
    assert_array_equal(mapped.labels, clabels)
    assert_array_equal(mapped.chunks, cchunks)
    # make sure origids get regenerated
    assert_array_equal([s.count('+') for s in mapped.sa.origids], [1] * 4)
