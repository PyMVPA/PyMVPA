# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for the dataset implementation'''

import numpy as N
import random

from numpy.testing import assert_array_equal
from nose.tools import ok_, assert_raises, assert_false, assert_equal

from mvpa.datasets.base import dataset, Dataset
from mvpa.mappers.array import DenseArrayMapper
from mvpa.misc.data_generators import normalFeatureDataset
import mvpa.support.copy as copy
from mvpa.misc.exceptions import DatasetError
from mvpa.misc.attributes import SampleAttribute

from tests_warehouse import *

def test_from_basic():
    samples = N.arange(12).reshape((4, 3))
    labels = range(4)
    chunks = [1, 1, 2, 2]

    ds = Dataset.from_basic(samples, labels, chunks)

    ## XXX stuff that needs thought:

    # ds.reset() << I guess we don't want that, but it is there.

    # ds.sa (empty) has this in the public namespace:
    #   add, get, getvalue, isKnown, isSet, items, listing, name, names
    #   owner, remove, reset, setvalue, whichSet
    # maybe we need some form of leightweightCollection?

    assert_array_equal(ds.samples, samples)
    assert_array_equal(ds.sa.labels, labels)
    assert_array_equal(ds.sa.chunks, chunks)

    # same should work for shortcuts
    assert_array_equal(ds.labels, labels)
    assert_array_equal(ds.chunks, chunks)

    ok_(sorted(ds.sa.names) == ['chunks', 'labels', 'origids'])

    # there is not necessarily a mapper present
    ok_(not ds.a.isKnown('mapper'))

    # has to complain about misshaped samples attributes
    assert_raises(DatasetError, Dataset.from_basic, samples, labels + labels)

    # check that we actually have attributes of the expected type
    ok_(isinstance(ds.sa['labels'], SampleAttribute))


def test_from_masked():
    ds = Dataset.from_masked(samples=[range(5)], labels=1, chunks=1)
    # simple sequence has to be a single pattern
    assert_equal(ds.nsamples, 1)
    # check correct pattern layout (1x5)
    assert_array_equal(ds.samples, [[0, 1, 2, 3, 4]])

    # check for single label and origin
    assert_array_equal(ds.labels, [1])
    assert_array_equal(ds.chunks, [1])

    # now try adding pattern with wrong shape
    assert_raises(DatasetError, ds.__iadd__,
                  Dataset.from_masked(N.ones((2,3)), labels=1, chunks=1))

    # now add two real patterns
    ds += Dataset.from_masked(N.random.standard_normal((2, 5)),
                              labels=2, chunks=2)
    assert_equal(ds.nsamples, 3)
    assert_array_equal(ds.labels, [1, 2, 2])
    assert_array_equal(ds.chunks, [1, 2, 2])

    # test unique class labels
    ds += Dataset.from_masked(N.random.standard_normal((2, 5)),
                              labels=3, chunks=5)
    assert_array_equal(ds.sa['labels'].unique, [1, 2, 3])

    # test wrong attributes length
    assert_raises(DatasetError, Dataset.from_masked,
                  N.random.standard_normal((4,2,3,4)), labels=[1, 2, 3],
                  chunks=2)
    assert_raises(DatasetError, Dataset.from_masked,
                  N.random.standard_normal((4,2,3,4)), labels=[1, 2, 3, 4],
                  chunks=[2, 2, 2])


def test_shape_conversion():
    ds = Dataset.from_masked(N.arange(24).reshape((2, 3, 4)),
                             labels=1, chunks=1)
    assert_equal(ds.nsamples, 2)
    assert_equal(ds.samples.shape, (2, 12))
    assert_array_equal(ds.samples, [range(12), range(12, 24)])


def test_samples_shape():
    ds = Dataset.from_masked(N.ones((10, 2, 3, 4)), labels=1, chunks=1)
    ok_(ds.samples.shape == (10, 24))


def test_basic_datamapping():
    samples = N.arange(24).reshape((4, 3, 2))

    # cannot handle 3d samples without a mapper
    assert_raises(DatasetError, Dataset, samples)

    ds = Dataset.from_basic(samples,
            mapper=DenseArrayMapper(shape=samples.shape[1:]))

    # mapper should end up in the dataset
    ok_(ds.a.isKnown('mapper') == ds.a.isSet('mapper') == True)

    # check correct mapping
    ok_(ds.nsamples == 4)
    ok_(ds.nfeatures == 6)


def test_ds_shallowcopy():
    # lets use some instance of somewhat evolved dataset
    ds = normalFeatureDataset()
    # SHALLOW copy the beast
    ds_ = copy.copy(ds)
    # verify that we have the same data
    assert_array_equal(ds.samples, ds_.samples)
    assert_array_equal(ds.labels, ds_.labels)
    assert_array_equal(ds.chunks, ds_.chunks)

    # modify and see that we actually DO change the data in both
    ds_.samples[0, 0] = 1234
    assert_array_equal(ds.samples, ds_.samples)
    assert_array_equal(ds.labels, ds_.labels)
    assert_array_equal(ds.chunks, ds_.chunks)

    ds_.sa.labels[0] = 'ab'
    ds_.sa.chunks[0] = 234
    assert_array_equal(ds.samples, ds_.samples)
    assert_array_equal(ds.labels, ds_.labels)
    assert_array_equal(ds.chunks, ds_.chunks)
    ok_(ds.sa.labels[0] == 'ab')
    ok_(ds.sa.chunks[0] == 234)

    # XXX implement me
    #ok_(N.any(ds.uniquelabels != ds_.uniquelabels))
    #ok_(N.any(ds.uniquechunks != ds_.uniquechunks))


def test_ds_deepcopy():
    # lets use some instance of somewhat evolved dataset
    ds = normalFeatureDataset()
    # Clone the beast
    ds_ = copy.deepcopy(ds)
    # verify that we have the same data
    assert_array_equal(ds.samples, ds_.samples)
    assert_array_equal(ds.labels, ds_.labels)
    assert_array_equal(ds.chunks, ds_.chunks)

    # modify and see if we don't change data in the original one
    ds_.samples[0, 0] = 1234
    ok_(N.any(ds.samples != ds_.samples))
    assert_array_equal(ds.labels, ds_.labels)
    assert_array_equal(ds.chunks, ds_.chunks)

    ds_.sa.labels = N.hstack(([123], ds_.labels[1:]))
    ok_(N.any(ds.samples != ds_.samples))
    ok_(N.any(ds.labels != ds_.labels))
    assert_array_equal(ds.chunks, ds_.chunks)

    ds_.sa.chunks = N.hstack(([1234], ds_.chunks[1:]))
    ok_(N.any(ds.samples != ds_.samples))
    ok_(N.any(ds.labels != ds_.labels))
    ok_(N.any(ds.chunks != ds_.chunks))

    # XXX implement me
    #ok_(N.any(ds.uniquelabels != ds_.uniquelabels))
    #ok_(N.any(ds.uniquechunks != ds_.uniquechunks))


def test_mergeds():
    data0 = Dataset.from_basic(N.ones((5, 5)), labels=1)
    data1 = Dataset.from_basic(N.ones((5, 5)), labels=1, chunks=1)
    data2 = Dataset.from_basic(N.ones((3, 5)), labels=2, chunks=1)
    data3 = Dataset.from_basic(N.ones((4, 5)), labels=2)

    # cannot merge if there are attributes missing in one of the datasets
    assert_raises(DatasetError, data1.__iadd__, data0)

    merged = data1 + data2

    ok_( merged.nfeatures == 5 )
    l12 = [1]*5 + [2]*3
    l1 = [1]*8
    ok_((merged.labels == l12).all())
    ok_((merged.chunks == l1).all())

    data1 += data2

    ok_(data1.nfeatures == 5)
    ok_((data1.labels == l12).all())
    ok_((data1.chunks == l1).all())

    # we need the same samples attributes in both datasets
    assert_raises(DatasetError, data2.__iadd__, data3)


def test_mergeds2():
    """Test composition of new datasets by addition of existing ones
    """
    data = dataset([range(5)], labels=1, chunks=1)

    assert_array_equal(data.UL, [1])

    # simple sequence has to be a single pattern
    assert_equal(data.nsamples, 1)
    # check correct pattern layout (1x5)
    assert_array_equal(data.samples, [[0, 1, 2, 3, 4]])

    # check for single labels and origin
    assert_array_equal(data.labels, [1])
    assert_array_equal(data.chunks, [1])

    # now try adding pattern with wrong shape
    assert_raises(DatasetError,
                  data.__iadd__,
                  dataset(N.ones((2,3)), labels=1, chunks=1))

    # now add two real patterns
    dss = datasets['uni2large'].samples
    data += dataset(dss[:2, :5], labels=2, chunks=2)
    assert_equal(data.nfeatures, 5)
    assert_array_equal(data.labels, [1, 2, 2])
    assert_array_equal(data.chunks, [1, 2, 2])

    # test automatic origins
    data += dataset(dss[3:5, :5], labels=3, chunks=[0, 1])
    assert_array_equal(data.chunks, [1, 2, 2, 0, 1])

    # test unique class labels
    assert_array_equal(data.UL, [1, 2, 3])

    # test wrong label length
    assert_raises(DatasetError, dataset, dss[:4, :5], labels=[ 1, 2, 3 ],
                                         chunks=2)

    # test wrong origin length
    assert_raises(DatasetError, dataset, dss[:4, :5], labels=[ 1, 2, 3, 4 ],
                                         chunks=[ 2, 2, 2 ])


def test_combined_samplesfeature_selection():
    data = dataset(N.arange(20).reshape((4, 5)),
                   labels=[1,2,3,4],
                   chunks=[5,6,7,8])

    ok_(data.nsamples == 4)
    ok_(data.nfeatures == 5)
    sel = data[[0, 3], [1, 2]]
    ok_(sel.nsamples == 2)
    ok_(sel.nfeatures == 2)
    assert_array_equal(sel.labels, [1, 4])
    assert_array_equal(sel.chunks, [5, 8])
    assert_array_equal(sel.samples, [[1, 2], [16, 17]])

    # should yield the same result if done sequentially
    sel2 = data[:, [1, 2]]
    sel2 = sel2[[0, 3]]
    assert_array_equal(sel.samples, sel2.samples)
    ok_(sel2.nsamples == 2)
    ok_(sel2.nfeatures == 2)

    assert_raises(ValueError, data.__getitem__, (1, 2, 3))

    # test correct behavior when selecting just single rows/columns
    single = data[0]
    ok_(single.nsamples == 1)
    ok_(single.nfeatures == 5)
    assert_array_equal(single.samples, [[0, 1, 2, 3, 4]])
    single = data[:, 0]
    ok_(single.nsamples == 4)
    ok_(single.nfeatures == 1)
    assert_array_equal(single.samples, [[0], [5], [10], [15]])
    single = data[1, 1]
    ok_(single.nsamples == 1)
    ok_(single.nfeatures == 1)
    assert_array_equal(single.samples, [[6]])



def test_labelpermutation_randomsampling():
    ds  = Dataset.from_basic(N.ones((5, 1)),     labels=range(5), chunks=1)
    ds += Dataset.from_basic(N.ones((5, 1)) + 1, labels=range(5), chunks=2)
    ds += Dataset.from_basic(N.ones((5, 1)) + 2, labels=range(5), chunks=3)
    ds += Dataset.from_basic(N.ones((5, 1)) + 3, labels=range(5), chunks=4)
    ds += Dataset.from_basic(N.ones((5, 1)) + 4, labels=range(5), chunks=5)

    # XXX put me back
    ok_(ds.get_nsamples_per_attr('labels') == {0:5, 1:5, 2:5, 3:5, 4:5})
    sample = ds.random_samples(2)
    ok_(sample.get_nsamples_per_attr('labels').values() == [ 2, 2, 2, 2, 2 ])
    ok_((ds.sa['chunks'].unique == range(1, 6)).all())

    # keep the orig labels
    orig_labels = ds.labels.copy()

    # also keep the orig dataset, but SHALLOW copy and leave everything
    # else as a view!
    ods = copy.copy(ds)

    ds.permute_labels()
    # some permutation should have happened
    assert_false((ds.labels == orig_labels).all())

    # but the original dataset should be uneffected
    assert_array_equal(ods.labels, orig_labels)

    # samples are really shared
    ds.samples[0, 0] = 123456
    assert_array_equal(ds.samples, ods.samples)

    # and other samples attributes too
    ds.chunks[0] = 9876
    assert_array_equal(ds.chunks, ods.chunks)


def test_feature2coord():
    origdata = N.random.standard_normal((10, 2, 4, 3, 5))
    data = Dataset.from_masked(origdata, labels=2, chunks=2)

    def random_coord(shape):
        return [random.sample(range(size), 1)[0] for size in shape]

    # check 100 random coord2feature transformations
    for i in xrange(100):
        # choose random coord
        c = random_coord((2, 4, 3, 5))
        # tranform to feature_id
        id_ = data.a.mapper.getOutId(c)

        # compare data from orig array (selected by coord)
        # and data from pattern array (selected by feature id)
        orig = origdata[:, c[0], c[1], c[2], c[3]]
        pat = data.samples[:, id_]

        assert_array_equal(orig, pat)


def test_coord2feature():
    origdata = N.random.standard_normal((10, 2, 4, 3, 5))
    data = Dataset.from_masked(origdata, labels=2, chunks=2)

    def random_coord(shape):
        return [random.sample(range(size), 1)[0] for size in shape]

    for id_ in xrange(data.nfeatures):
        # transform to coordinate
        c = data.a.mapper.getInId(id_)
        assert_equal(len(c), 4)

        # compare data from orig array (selected by coord)
        # and data from pattern array (selected by feature id)
        orig = origdata[:, c[0], c[1], c[2], c[3]]
        pat = data.samples[:, id_]

        assert_array_equal(orig, pat)


def test_masked_featureselection():
    origdata = N.random.standard_normal((10, 2, 4, 3, 5))
    data = Dataset.from_masked(origdata, labels=2, chunks=2)

    unmasked = data.samples.copy()

    # default must be no mask
    ok_(data.nfeatures == 120)
    ok_(data.a.mapper.getOutSize() == 120)

    # check that full mask uses all features
    # this uses auto-mapping of selection arrays in __getitem__
    sel = data[:, N.ones((2, 4, 3, 5), dtype='bool')]
    ok_(sel.nfeatures == data.samples.shape[1])
    ok_(data.nfeatures == 120)

    # check partial array mask
    partial_mask = N.zeros((2, 4, 3, 5), dtype='bool')
    partial_mask[0, 0, 2, 2] = 1
    partial_mask[1, 2, 2, 0] = 1

    sel = data[:, partial_mask]
    ok_(sel.nfeatures == 2)
    ok_(sel.a.mapper.getMask().shape == (2, 4, 3, 5))

    # check that feature selection does not change source data
    ok_(data.nfeatures == 120)
    assert_equal(data.a.mapper.getOutSize(), 120)

    # check selection with feature list
    sel = data[:, [0, 37, 119]]
    ok_(sel.nfeatures == 3)

    # check size of the masked samples
    ok_(sel.samples.shape == (10, 3))

    # check that the right features are selected
    assert_array_equal(unmasked[:, [0, 37, 119]], sel.samples)


def test_origmask_extraction():
    origdata = N.random.standard_normal((10, 2, 4, 3))
    data = Dataset.from_masked(origdata, labels=2, chunks=2)

    # check with custom mask
    sel = data[:, 5]
    ok_(sel.samples.shape[1] == 1)
    origmask = sel.a.mapper.getMask()
    ok_(origmask[0, 1, 2] == True)
    ok_(origmask.shape == (2, 4, 3))


def test_feature_masking():
    mask = N.zeros((5, 3), dtype='bool')
    mask[2, 1] = True
    mask[4, 0] = True
    data = Dataset.from_masked(N.arange(60).reshape((4, 5, 3)),
                               labels=1, chunks=1, mask=mask)

    # check simple masking
    ok_(data.nfeatures == 2)
    ok_(data.a.mapper.getOutId((2, 1)) == 0 and
        data.a.mapper.getOutId((4,0)) == 1 )
    assert_raises(ValueError, data.a.mapper.getOutId, (2, 3))
    ok_(data.a.mapper.getMask().shape == (5, 3))
    ok_(tuple(data.a.mapper.getInId(1)) == (4, 0))

    # selection should be idempotent
    ok_(data[:, mask].nfeatures == data.nfeatures)
    # check that correct feature get selected
    assert_array_equal(data[:, 1].samples[:, 0], [12, 27, 42, 57])
    ok_(tuple(data[:, 1].a.mapper.getInId(0)) == (4, 0))
    ok_(data[:, 1].a.mapper.getMask().sum() == 1)

    # previous selections should not affect the original mapper
    ok_(data.a.mapper.getOutId((2, 1)) == 0 and
        data.a.mapper.getOutId((4, 0)) == 1 )

    # check sugarings
    # XXX put me back
    #self.failUnless(N.all(data.I == data.origids))
    assert_array_equal(data.C, data.chunks)
    assert_array_equal(data.UC, N.unique(data.chunks))
    assert_array_equal(data.L, data.labels)
    assert_array_equal(data.UL, N.unique(data.labels))
    assert_array_equal(data.S, data.samples)
    assert_array_equal(data.O, data.mapper.reverse(data.samples))


def test_origid_handling():
    ds = dataset(N.atleast_2d(N.arange(35)).T)
    ok_(ds.nsamples == 35)
    assert_equal(len(N.unique(ds.sa.origids)), 35)
    selector = [3, 7, 10, 15]
    subds = ds[selector]
    assert_array_equal(subds.sa.origids, ds.sa.origids[selector])
