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
import shutil
import tempfile
import os

from numpy.testing import assert_array_equal
from nose.tools import ok_, assert_raises, assert_false, assert_equal, \
        assert_true

from mvpa.base.externals import versions, exists
from mvpa.base.types import is_datasetlike
from mvpa.base.dataset import DatasetError, vstack, hstack
from mvpa.mappers.flatten import mask_mapper
from mvpa.datasets.base import dataset, Dataset
from mvpa.misc.data_generators import normalFeatureDataset
import mvpa.support.copy as copy
from mvpa.base.collections import SampleAttributesCollection, \
        FeatureAttributesCollection, DatasetAttributesCollection, \
        ArrayCollectable, SampleAttribute

from tests_warehouse import *

class myarray(N.ndarray):
    pass


# TODO Urgently need test to ensure that multidimensional samples and feature
#      attributes work and adjust docs to mention that we support such
def test_from_basic():
    samples = N.arange(12).reshape((4, 3)).view(myarray)
    labels = range(4)
    chunks = [1, 1, 2, 2]

    ds = Dataset.from_basic(samples, labels, chunks)
    ds.init_origids('both')
    first = ds.sa.origids
    # now do again and check that they get regenerated
    ds.init_origids('both')
    assert_false(first is ds.sa.origids)
    assert_array_equal(first, ds.sa.origids)

    ok_(is_datasetlike(ds))
    ok_(not is_datasetlike(labels))

    # array subclass survives
    ok_(isinstance(ds.samples, myarray))

    ## XXX stuff that needs thought:

    # ds.sa (empty) has this in the public namespace:
    #   add, get, getvalue, has_key, is_set, items, listing, name, names
    #   owner, remove, reset, setvalue, which_set
    # maybe we need some form of leightweightCollection?

    assert_array_equal(ds.samples, samples)
    assert_array_equal(ds.sa.labels, labels)
    assert_array_equal(ds.sa.chunks, chunks)

    # same should work for shortcuts
    assert_array_equal(ds.labels, labels)
    assert_array_equal(ds.chunks, chunks)

    ok_(sorted(ds.sa.keys()) == ['chunks', 'labels', 'origids'])
    ok_(sorted(ds.fa.keys()) == ['origids'])
    # add some more
    ds.a['random'] = 'blurb'

    # check stripping attributes from a copy
    cds = ds.copy() # full copy
    ok_(sorted(cds.sa.keys()) == ['chunks', 'labels', 'origids'])
    ok_(sorted(cds.fa.keys()) == ['origids'])
    ok_(sorted(cds.a.keys()) == ['random'])
    cds = ds.copy(sa=[], fa=[], a=[]) # plain copy
    ok_(cds.sa.keys() == [])
    ok_(cds.fa.keys() == [])
    ok_(cds.a.keys() == [])
    cds = ds.copy(sa=['labels'], fa=None, a=['random']) # partial copy
    ok_(cds.sa.keys() == ['labels'])
    ok_(cds.fa.keys() == ['origids'])
    ok_(cds.a.keys() == ['random'])

    # there is not necessarily a mapper present
    ok_(not ds.a.has_key('mapper'))

    # has to complain about misshaped samples attributes
    assert_raises(ValueError, Dataset.from_basic, samples, labels + labels)

    # check that we actually have attributes of the expected type
    ok_(isinstance(ds.sa['labels'], ArrayCollectable))

    # the dataset will take care of not adding stupid stuff
    assert_raises(ValueError, ds.sa.__setitem__, 'stupid', N.arange(3))
    assert_raises(ValueError, ds.fa.__setitem__, 'stupid', N.arange(4))
    # or change proper attributes to stupid shapes
    try:
        ds.sa.labels = N.arange(3)
    except ValueError:
        pass
    else:
        ok_(False, msg="Assigning value with improper shape to attribute "
                       "did not raise exception.")


def test_labelschunks_access():
    samples = N.arange(12).reshape((4, 3)).view(myarray)
    labels = range(4)
    chunks = [1, 1, 2, 2]
    ds = Dataset.from_basic(samples, labels, chunks)

    # array subclass survives
    ok_(isinstance(ds.samples, myarray))

    assert_array_equal(ds.labels, labels)
    assert_array_equal(ds.chunks, chunks)

    # moreover they should point to the same thing
    ok_(ds.labels is ds.sa.labels)
    ok_(ds.labels is ds.sa['labels'].value)
    ok_(ds.chunks is ds.sa.chunks)
    ok_(ds.chunks is ds.sa['chunks'].value)

    # assignment should work at all levels including 1st
    ds.labels = chunks
    assert_array_equal(ds.labels, chunks)
    ok_(ds.labels is ds.sa.labels)
    ok_(ds.labels is ds.sa['labels'].value)


def test_from_masked():
    ds = Dataset.from_masked(samples=N.atleast_2d(N.arange(5)).view(myarray),
                             labels=1, chunks=1)
    # simple sequence has to be a single pattern
    assert_equal(ds.nsamples, 1)
    # array subclass survives
    ok_(isinstance(ds.samples, myarray))

    # check correct pattern layout (1x5)
    assert_array_equal(ds.samples, [[0, 1, 2, 3, 4]])

    # check for single label and origin
    assert_array_equal(ds.labels, [1])
    assert_array_equal(ds.chunks, [1])

    # now try adding pattern with wrong shape
    assert_raises(DatasetError, ds.append,
                  Dataset.from_masked(N.ones((2,3)), labels=1, chunks=1))

    # now add two real patterns
    ds.append(Dataset.from_masked(N.random.standard_normal((2, 5)),
                                  labels=2, chunks=2))
    assert_equal(ds.nsamples, 3)
    assert_array_equal(ds.labels, [1, 2, 2])
    assert_array_equal(ds.chunks, [1, 2, 2])

    # test unique class labels
    ds.append(Dataset.from_masked(N.random.standard_normal((2, 5)),
                                  labels=3, chunks=5))
    assert_array_equal(ds.sa['labels'].unique, [1, 2, 3])

    # test wrong attributes length
    assert_raises(ValueError, Dataset.from_masked,
                  N.random.standard_normal((4,2,3,4)), labels=[1, 2, 3],
                  chunks=2)
    assert_raises(ValueError, Dataset.from_masked,
                  N.random.standard_normal((4,2,3,4)), labels=[1, 2, 3, 4],
                  chunks=[2, 2, 2])

    # no test one that is using from_masked
    ds = datasets['3dlarge']
    for a in ds.sa:
        assert_equal(len(ds.sa[a].value), len(ds))
    for a in ds.fa:
        assert_equal(len(ds.fa[a].value), ds.nfeatures)


def test_shape_conversion():
    ds = Dataset.from_masked(N.arange(24).reshape((2, 3, 4)).view(myarray),
                             labels=1, chunks=1)
    # array subclass survives
    ok_(isinstance(ds.samples, myarray))

    assert_equal(ds.nsamples, 2)
    assert_equal(ds.samples.shape, (2, 12))
    assert_array_equal(ds.samples, [range(12), range(12, 24)])


def test_multidim_attrs():
    samples = N.arange(24).reshape(2, 3, 4)
    # have a dataset with two samples -- mapped from 2d into 1d
    # but have 2d labels and 3d chunks -- whatever that is
    ds = Dataset.from_masked(samples.copy(),
                             labels=samples.copy(),
                             chunks=N.random.normal(size=(2,10,4,2)))
    assert_equal(ds.nsamples, 2)
    assert_equal(ds.nfeatures, 12)
    assert_equal(ds.sa.labels.shape, (2,3,4))
    assert_equal(ds.sa.chunks.shape, (2,10,4,2))

    # try slicing
    subds = ds[0]
    assert_equal(subds.nsamples, 1)
    assert_equal(subds.nfeatures, 12)
    assert_equal(subds.sa.labels.shape, (1,3,4))
    assert_equal(subds.sa.chunks.shape, (1,10,4,2))

    # add multidim feature attr
    fattr = ds.mapper.forward(samples)
    assert_equal(fattr.shape, (2,12))
    # should puke -- first axis is #samples
    assert_raises(ValueError, ds.fa.__setitem__, 'moresamples', fattr)
    # but that should be fine
    ds.fa['moresamples'] = fattr.T
    assert_equal(ds.fa.moresamples.shape, (12,2))



def test_samples_shape():
    ds = Dataset.from_masked(N.ones((10, 2, 3, 4)), labels=1, chunks=1)
    ok_(ds.samples.shape == (10, 24))

    # what happens to 1D samples
    ds = Dataset(N.arange(5))
    assert_equal(ds.shape, (5, 1))
    assert_equal(ds.nfeatures, 1)


def test_basic_datamapping():
    samples = N.arange(24).reshape((4, 3, 2)).view(myarray)

    ds = Dataset.from_basic(samples,
            mapper=mask_mapper(shape=samples.shape[1:]))

    # array subclass survives
    ok_(isinstance(ds.samples, myarray))

    # mapper should end up in the dataset
    ok_(ds.a.has_key('mapper'))

    # check correct mapping
    ok_(ds.nsamples == 4)
    ok_(ds.nfeatures == 6)


def test_ds_shallowcopy():
    # lets use some instance of somewhat evolved dataset
    ds = normalFeatureDataset()
    ds.samples = ds.samples.view(myarray)

    # SHALLOW copy the beast
    ds_ = copy.copy(ds)
    # verify that we have the same data
    assert_array_equal(ds.samples, ds_.samples)
    assert_array_equal(ds.labels, ds_.labels)
    assert_array_equal(ds.chunks, ds_.chunks)

    # array subclass survives
    ok_(isinstance(ds_.samples, myarray))


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
    ds.samples = ds.samples.view(myarray)
    # Clone the beast
    ds_ = ds.copy()
    # array subclass survives
    ok_(isinstance(ds_.samples, myarray))

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
    data0.fa['one'] = N.ones(5)
    data1 = Dataset.from_basic(N.ones((5, 5)), labels=1, chunks=1)
    data1.fa['one'] = N.zeros(5)
    data2 = Dataset.from_basic(N.ones((3, 5)), labels=2, chunks=1)
    data3 = Dataset.from_basic(N.ones((4, 5)), labels=2)
    data4 = Dataset.from_basic(N.ones((2, 5)), labels=3, chunks=2)
    data4.fa['test'] = N.arange(5)

    # cannot merge if there are attributes missing in one of the datasets
    assert_raises(DatasetError, data1.append, data0)

    merged = data1.copy()
    merged.append(data2)

    ok_( merged.nfeatures == 5 )
    l12 = [1]*5 + [2]*3
    l1 = [1]*8
    ok_((merged.labels == l12).all())
    ok_((merged.chunks == l1).all())

    data_append = data1.copy()
    data_append.append(data2)

    ok_(data_append.nfeatures == 5)
    ok_((data_append.labels == l12).all())
    ok_((data_append.chunks == l1).all())

    #
    # appending
    #

    # we need the same samples attributes in both datasets
    assert_raises(DatasetError, data2.append, data3)

    #
    # vstacking
    #
    assert_raises(ValueError, vstack, (data0, data1, data2, data3))
    datasets = (data1, data2, data4)
    merged = vstack(datasets)
    assert_equal(merged.shape,
                 (N.sum([len(ds) for ds in datasets]), data1.nfeatures))
    assert_true('test' in merged.fa)
    assert_array_equal(merged.sa.labels, [1]*5 + [2]*3 + [3]*2)

    #
    # hstacking
    #
    assert_raises(ValueError, hstack, datasets)
    datasets = (data0, data1)
    merged = hstack(datasets)
    assert_equal(merged.shape,
                 (len(data1), N.sum([ds.nfeatures for ds in datasets])))
    assert_true('chunks' in merged.sa)
    assert_array_equal(merged.fa.one, [1]*5 + [0]*5)


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
                  data.append,
                  dataset(N.ones((2,3)), labels=1, chunks=1))

    # now add two real patterns
    dss = datasets['uni2large'].samples
    data.append(dataset(dss[:2, :5], labels=2, chunks=2))
    assert_equal(data.nfeatures, 5)
    assert_array_equal(data.labels, [1, 2, 2])
    assert_array_equal(data.chunks, [1, 2, 2])

    # test automatic origins
    data.append(dataset(dss[3:5, :5], labels=3, chunks=[0, 1]))
    assert_array_equal(data.chunks, [1, 2, 2, 0, 1])

    # test unique class labels
    assert_array_equal(data.UL, [1, 2, 3])

    # test wrong label length
    assert_raises(ValueError, dataset, dss[:4, :5], labels=[ 1, 2, 3 ],
                                         chunks=2)

    # test wrong origin length
    assert_raises(ValueError, dataset, dss[:4, :5], labels=[ 1, 2, 3, 4 ],
                                         chunks=[ 2, 2, 2 ])


def test_combined_samplesfeature_selection():
    data = dataset(N.arange(20).reshape((4, 5)).view(myarray),
                   labels=[1,2,3,4],
                   chunks=[5,6,7,8])

    # array subclass survives
    ok_(isinstance(data.samples, myarray))

    ok_(data.nsamples == 4)
    ok_(data.nfeatures == 5)
    sel = data[[0, 3], [1, 2]]
    ok_(sel.nsamples == 2)
    ok_(sel.nfeatures == 2)
    assert_array_equal(sel.labels, [1, 4])
    assert_array_equal(sel.chunks, [5, 8])
    assert_array_equal(sel.samples, [[1, 2], [16, 17]])
    # array subclass survives
    ok_(isinstance(sel.samples, myarray))


    # should yield the same result if done sequentially
    sel2 = data[:, [1, 2]]
    sel2 = sel2[[0, 3]]
    assert_array_equal(sel.samples, sel2.samples)
    ok_(sel2.nsamples == 2)
    ok_(sel2.nfeatures == 2)
    # array subclass survives
    ok_(isinstance(sel.samples, myarray))


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
    # array subclass survives
    ok_(isinstance(single.samples, myarray))



def test_labelpermutation_randomsampling():
    ds  = Dataset.from_basic(N.ones((5, 1)),     labels=range(5), chunks=1)
    ds.append(Dataset.from_basic(N.ones((5, 1)) + 1, labels=range(5), chunks=2))
    ds.append(Dataset.from_basic(N.ones((5, 1)) + 2, labels=range(5), chunks=3))
    ds.append(Dataset.from_basic(N.ones((5, 1)) + 3, labels=range(5), chunks=4))
    ds.append(Dataset.from_basic(N.ones((5, 1)) + 4, labels=range(5), chunks=5))
    # use subclass for testing if it would survive
    ds.samples = ds.samples.view(myarray)

    ok_(ds.get_nsamples_per_attr('labels') == {0:5, 1:5, 2:5, 3:5, 4:5})
    sample = ds.random_samples(2)
    ok_(sample.get_nsamples_per_attr('labels').values() == [ 2, 2, 2, 2, 2 ])
    ok_((ds.sa['chunks'].unique == range(1, 6)).all())

    # keep the orig labels
    orig_labels = ds.labels[:]

    # also keep the orig dataset, but SHALLOW copy and leave everything
    # else as a view!
    ods = copy.copy(ds)

    ds.permute_labels()
    # some permutation should have happened
    assert_false((ds.labels == orig_labels).all())

    # but the original dataset should be uneffected
    assert_array_equal(ods.labels, orig_labels)
    # array subclass survives
    ok_(isinstance(ods.samples, myarray))

    # samples are really shared
    ds.samples[0, 0] = 123456
    assert_array_equal(ds.samples, ods.samples)

    # and other samples attributes too
    ds.chunks[0] = 9876
    assert_array_equal(ds.chunks, ods.chunks)


def test_masked_featureselection():
    origdata = N.random.standard_normal((10, 2, 4, 3, 5)).view(myarray)
    data = Dataset.from_masked(origdata, labels=2, chunks=2)

    unmasked = data.samples.copy()
    # array subclass survives
    ok_(isinstance(data.samples, myarray))

    # default must be no mask
    ok_(data.nfeatures == 120)
    ok_(data.a.mapper.forward1(origdata[0]).shape == (120,))

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

    # check that feature selection does not change source data
    ok_(data.nfeatures == 120)
    ok_(data.a.mapper.forward1(origdata[0]).shape == (120,))

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


def test_feature_masking():
    mask = N.zeros((5, 3), dtype='bool')
    mask[2, 1] = True
    mask[4, 0] = True
    data = Dataset.from_masked(N.arange(60).reshape((4, 5, 3)),
                               labels=1, chunks=1, mask=mask)

    # check simple masking
    ok_(data.nfeatures == 2)

    # selection should be idempotent
    ok_(data[:, mask].nfeatures == data.nfeatures)
    # check that correct feature get selected
    assert_array_equal(data[:, 1].samples[:, 0], [12, 27, 42, 57])
    # XXX put back when coord -> fattr is implemented
    #ok_(tuple(data[:, 1].a.mapper.getInId(0)) == (4, 0))
    ok_(data[:, 1].a.mapper.forward1(mask).shape == (1,))

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
    ds.init_origids('both')
    ok_(ds.nsamples == 35)
    assert_equal(len(N.unique(ds.sa.origids)), 35)
    assert_equal(len(N.unique(ds.fa.origids)), 1)
    selector = [3, 7, 10, 15]
    subds = ds[selector]
    assert_array_equal(subds.sa.origids, ds.sa.origids[selector])

    # Now if we request new origids if they are present we could
    # expect different behavior
    assert_raises(ValueError, subds.init_origids, 'both', mode='raises')
    sa_origids = subds.sa.origids.copy()
    fa_origids = subds.fa.origids.copy()
    for s in ('both', 'samples', 'features'):
        assert_raises(RuntimeError, subds.init_origids, s, mode='raise')
        subds.init_origids(s, mode='existing')
        # we should have the same origids as before
        assert_array_equal(subds.sa.origids, sa_origids)
        assert_array_equal(subds.fa.origids, fa_origids)

    # Lets now change, which should be default behavior
    subds.init_origids('both')
    assert_equal(len(sa_origids), len(subds.sa.origids))
    assert_equal(len(fa_origids), len(subds.fa.origids))
    # values should change though
    ok_((sa_origids != subds.sa.origids).any())
    ok_((fa_origids != subds.fa.origids).any())

def test_idhash():
    ds = dataset(N.arange(12).reshape((4, 3)),
                 labels=1, chunks=1)
    origid = ds.idhash
    #XXX BUG -- no assurance that labels would become an array... for now -- do manually
    ds.labels = N.array([3, 1, 2, 3])   # change all labels
    ok_(origid != ds.idhash,
                    msg="Changing all labels should alter dataset's idhash")

    origid = ds.idhash

    z = ds.labels[1]
    assert_equal(origid, ds.idhash,
                 msg="Accessing shouldn't change idhash")
    z = ds.chunks
    assert_equal(origid, ds.idhash,
                 msg="Accessing shouldn't change idhash")
    z[2] = 333
    ok_(origid != ds.idhash,
        msg="Changing value in attribute should change idhash")

    origid = ds.idhash
    ds.samples[1, 1] = 1000
    ok_(origid != ds.idhash,
        msg="Changing value in data should change idhash")

    origid = ds.idhash
    orig_labels = ds.labels #.copy()
    ds.permute_labels()
    ok_(origid != ds.idhash,
        msg="Permutation also changes idhash")

    ds.labels = orig_labels
    ok_(origid == ds.idhash,
        msg="idhash should be restored after reassigning orig labels")


def test_arrayattributes():
    samples = N.arange(12).reshape((4, 3))
    labels = range(4)
    chunks = [1, 1, 2, 2]
    ds = dataset(samples, labels, chunks)

    for a in (ds.samples, ds.labels, ds.chunks):
        ok_(isinstance(a, N.ndarray))

    ds.labels = labels
    ok_(isinstance(ds.labels, N.ndarray))

    ds.chunks = chunks
    ok_(isinstance(ds.chunks, N.ndarray))


def test_repr():
    attr_repr = "SampleAttribute(name='TestAttr', doc='my own test', " \
                                "value=array([0, 1, 2, 3, 4]), length=None)"
    sattr = SampleAttribute(name='TestAttr', doc='my own test',
                            value=N.arange(5))
    # check precise formal representation
    ok_(repr(sattr) == attr_repr)
    # check that it actually works as a Python expression
    from numpy import array
    eattr = eval(repr(sattr))
    ok_(repr(eattr), attr_repr)

    # should also work for a simple dataset
    ds = datasets['uni2small']
    ds_repr = repr(ds)
    ok_(repr(eval(ds_repr)) == ds_repr)


def is_bsr(x):
    """Helper function to check if instance is bsr_matrix if such is
    avail at all
    """
    import scipy.sparse as sparse
    return hasattr(sparse, 'bsr_matrix') and isinstance(x, sparse.bsr_matrix)

def test_other_samples_dtypes():
    import scipy.sparse as sparse
    dshape = (4, 3)
    # test for ndarray, custom ndarray-subclass, matrix,
    # and all sparse matrix types we know
    stypes = [N.arange(N.prod(dshape)).reshape(dshape),
              N.arange(N.prod(dshape)).reshape(dshape).view(myarray),
              N.matrix(N.arange(N.prod(dshape)).reshape(dshape)),
              sparse.csc_matrix(N.arange(N.prod(dshape)).reshape(dshape)),
              sparse.csr_matrix(N.arange(N.prod(dshape)).reshape(dshape))]
    if hasattr(sparse, 'bsr_matrix'):
        stypes += [
              # BSR cannot be sliced, but is more efficient for sparse
              # arithmetic operations than CSC pr CSR
              sparse.bsr_matrix(N.arange(N.prod(dshape)).reshape(dshape))]
              # LIL and COO are best for constructing matrices, not for
              # doing something with them
              #sparse.lil_matrix(N.arange(N.prod(dshape)).reshape(dshape)),
              #sparse.coo_matrix(N.arange(N.prod(dshape)).reshape(dshape)),
              # DOK doesn't allow duplicates and is bad at array-like slicing
              #sparse.dok_matrix(N.arange(N.prod(dshape)).reshape(dshape)),
              # DIA only has diagonal storage and cannot be sliced
              #sparse.dia_matrix(N.arange(N.prod(dshape)).reshape(dshape))]

    # it needs to have .shape (the only way to get len(sparse))
    for s in stypes:
        ds = Dataset(s)
        # nothing happended to the original dtype
        assert_equal(type(s), type(ds.samples))
        # no shape change
        assert_equal(ds.shape, dshape)
        assert_equal(ds.nsamples, dshape[0])
        assert_equal(ds.nfeatures, dshape[1])

        # sparse doesn't work like an array
        if sparse.isspmatrix(ds.samples):
            assert_raises(RuntimeError, N.mean, ds)
        else:
            # need to convert results, since matrices return matrices
            assert_array_equal(N.mean(ds, axis=0),
                               N.array(N.mean(ds.samples, axis=0)).squeeze())

        # select subset and see what happens
        # bsr type doesn't support first axis slicing
        if is_bsr(s):
            assert_raises(NotImplementedError, ds.__getitem__, [0])
        elif versions['scipy'] <= '0.6.0' and sparse.isspmatrix(ds.samples):
            assert_raises(IndexError, ds.__getitem__, [0])
        else:
            sel = ds[1:3]
            assert_equal(sel.shape, (2, dshape[1]))
            assert_equal(type(s), type(sel.samples))
            if sparse.isspmatrix(sel.samples):
                assert_array_equal(sel.samples[1].todense(),
                                   ds.samples[2].todense())
            else:
                assert_array_equal(sel.samples[1],
                                   ds.samples[2])

        # feature selection
        if is_bsr(s):
            assert_raises(NotImplementedError, ds.__getitem__, (slice(None), 0))
        elif versions['scipy'] <= '0.6.0' and sparse.isspmatrix(ds.samples):
            assert_raises(IndexError, ds.__getitem__, (slice(None), 0))
        else:
            sel = ds[:, 1:3]
            assert_equal(sel.shape, (dshape[0], 2))
            assert_equal(type(s), type(sel.samples))
            if sparse.isspmatrix(sel.samples):
                assert_array_equal(sel.samples[:,1].todense(),
                        ds.samples[:,2].todense())
            else:
                assert_array_equal(sel.samples[:,1],
                        ds.samples[:,2])


        # what we don't do
        class voodoo:
            dtype = 'fancy'
        # voodoo
        assert_raises(ValueError, Dataset, voodoo())
        # crippled
        assert_raises(ValueError, Dataset, N.array(5))

        # things that might behave in surprising ways
        # lists -- first axis is samples, hence single feature
        ds = Dataset(range(5))
        assert_equal(ds.nfeatures, 1)
        assert_equal(ds.shape, (5, 1))
        # arrays of objects
        data = N.array([{},{}])
        ds = Dataset(data)
        assert_equal(ds.shape, (2, 1))
        assert_equal(ds.nsamples, 2)
        # Nothing to index, hence no features
        assert_equal(ds.nfeatures, 1)


def test_h5py_io():
    if not exists('h5py'):
        return

    tempdir = tempfile.mkdtemp()

    # store random dataset to file
    ds = datasets['3dlarge']
    ds.save(os.path.join(tempdir, 'plain.hdf5'))

    # reload and check for identity
    ds2 = Dataset.from_hdf5(os.path.join(tempdir, 'plain.hdf5'))
    assert_array_equal(ds.samples, ds2.samples)
    for attr in ds.sa:
        assert_array_equal(ds.sa[attr].value, ds2.sa[attr].value)
    for attr in ds.fa:
        assert_array_equal(ds.fa[attr].value, ds2.fa[attr].value)
    assert_true(len(ds.a.mapper), 2)
    # since we have no __equal__ do at least some comparison
    assert_equal(repr(ds.a.mapper), repr(ds2.a.mapper))

    #cleanup temp dir
    shutil.rmtree(tempdir, ignore_errors=True)
