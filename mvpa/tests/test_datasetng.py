'''Tests for the dataset implementation'''
import numpy as N

from numpy.testing import assert_array_equal
from nose.tools import ok_, assert_raises

from mvpa.datasets.base import Dataset
from mvpa.mappers.array import DenseArrayMapper


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

    # there is not necessarily a mapper present
    ok_(not ds.a.isKnown('mapper'))


def test_basic_datamapping():
    samples = N.arange(24).reshape((4,3,2))

    # cannot handle 3d samples without a mapper
    assert_raises(ValueError, Dataset, samples)

    ds = Dataset.from_unlabeled(samples,
            mapper=DenseArrayMapper(shape=samples.shape[1:]))

    # mapper should end up in the dataset
    ok_(ds.a.isKnown('mapper') == ds.a.isSet('mapper') == True)

    # check correct mapping
    ok_(ds.nsamples == 4)
    ok_(ds.nfeatures == 6)

