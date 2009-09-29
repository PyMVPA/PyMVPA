'''Tests for the dataset implementation'''
import numpy as N

from numpy.testing import assert_array_equal
from nose.tools import ok_

from mvpa.datasets.base import Dataset


def test_from_labeled():
    samples = N.arange(12).reshape((4,3))
    labels = range(4)
    chunks = [1, 1, 2, 2]

    ds = Dataset.from_labeled(samples, labels, chunks)

    ## XXX stuff that needs thought:

    # ds.reset() << I guess we don't want that, but it is there.

    # ds.sa (empty) has this in the public namespace:
    #   add, get, getvalue, isKnown, isSet, items, listing, name, names
    #   owner, remove, reset, setvalue, whichSet
    # maybe we need some form of leightweightCollection?

    assert_array_equal(ds.samples, samples)
    ok_(ds.sa.labels == labels)
    ok_(ds.sa.chunks == chunks)

    # same should work for shortcuts
    ok_(ds.labels == labels)
    ok_(ds.chunks == chunks)

    ok_(sorted(ds.sa.names) == ['chunks', 'labels'])
