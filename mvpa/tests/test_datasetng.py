'''Tests for the dataset implementation'''
import numpy as N

from numpy.testing import assert_array_equal
from nose.tools import ok_

from mvpa.datasets.base import _Dataset as Dataset


def test_initSimple():
    samples = N.arange(12).reshape((4,3))
    labels = range(4)
    chunks = [1, 1, 2, 2]

    ds = Dataset.initSimple(samples, labels, chunks)

    assert_array_equal(ds.samples, samples)
    ok_(ds.sa.labels == labels)
    ok_(ds.sa.chunks == chunks)

    # XXX but why is this puking, or rather do we want to keep cluttering the
    # interface like this -- I'd prefer having it all inside the collection.
    ok_(ds.labels == labels)
    ok_(ds.chunks == chunks)

    ok_(sorted(ds.sa.names) == ['chunks', 'labels'])
