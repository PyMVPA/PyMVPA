# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for the dataset implementation'''

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets

import numpy as np
import shutil
import tempfile
import os


from mvpa2.base import cfg
from mvpa2.base.externals import versions
from mvpa2.base.types import is_datasetlike
from mvpa2.base.dataset import DatasetError, vstack, hstack, all_equal, \
                                stack_by_unique_feature_attribute, \
                                stack_by_unique_sample_attribute
from mvpa2.datasets.base import dataset_wizard, Dataset, HollowSamples
from mvpa2.misc.data_generators import normal_feature_dataset
from mvpa2.testing import reseed_rng
import mvpa2.support.copy as copy
from mvpa2.base.collections import \
     SampleAttributesCollection, FeatureAttributesCollection, \
     DatasetAttributesCollection, ArrayCollectable, SampleAttribute, \
     Collectable


class myarray(np.ndarray):
    pass


# TODO Urgently need test to ensure that multidimensional samples and feature
#      attributes work and adjust docs to mention that we support such
def test_from_wizard():
    samples = np.arange(12).reshape((4, 3)).view(myarray)
    labels = range(4)
    chunks = [1, 1, 2, 2]

    ds = Dataset(samples, sa={'targets': labels, 'chunks': chunks})
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
    assert_array_equal(ds.sa.targets, labels)
    assert_array_equal(ds.sa.chunks, chunks)

    # same should work for shortcuts
    assert_array_equal(ds.targets, labels)
    assert_array_equal(ds.chunks, chunks)

    ok_(sorted(ds.sa.keys()) == ['chunks', 'origids', 'targets'])
    ok_(sorted(ds.fa.keys()) == ['origids'])
    # add some more
    ds.a['random'] = 'blurb'

    # check stripping attributes from a copy
    cds = ds.copy() # full copy
    ok_(sorted(cds.sa.keys()) == ['chunks', 'origids', 'targets'])
    ok_(sorted(cds.fa.keys()) == ['origids'])
    ok_(sorted(cds.a.keys()) == ['random'])
    cds = ds.copy(sa=[], fa=[], a=[]) # plain copy
    ok_(cds.sa.keys() == [])
    ok_(cds.fa.keys() == [])
    ok_(cds.a.keys() == [])
    cds = ds.copy(sa=['targets'], fa=None, a=['random']) # partial copy
    ok_(cds.sa.keys() == ['targets'])
    ok_(cds.fa.keys() == ['origids'])
    ok_(cds.a.keys() == ['random'])

    # there is not necessarily a mapper present
    ok_(not ds.a.has_key('mapper'))

    # has to complain about misshaped samples attributes
    assert_raises(ValueError, Dataset.from_wizard, samples, labels + labels)

    # check that we actually have attributes of the expected type
    ok_(isinstance(ds.sa['targets'], ArrayCollectable))

    # the dataset will take care of not adding stupid stuff
    assert_raises(ValueError, ds.sa.__setitem__, 'stupid', np.arange(3))
    assert_raises(ValueError, ds.fa.__setitem__, 'stupid', np.arange(4))
    # or change proper attributes to stupid shapes
    try:
        ds.sa.targets = np.arange(3)
    except ValueError:
        pass
    else:
        ok_(False, msg="Assigning value with improper shape to attribute "
                       "did not raise exception.")


def test_labelschunks_access():
    samples = np.arange(12).reshape((4, 3)).view(myarray)
    labels = range(4)
    chunks = [1, 1, 2, 2]
    ds = Dataset.from_wizard(samples, labels, chunks)

    # array subclass survives
    ok_(isinstance(ds.samples, myarray))

    assert_array_equal(ds.targets, labels)
    assert_array_equal(ds.chunks, chunks)

    # moreover they should point to the same thing
    ok_(ds.targets is ds.sa.targets)
    ok_(ds.targets is ds.sa['targets'].value)
    ok_(ds.chunks is ds.sa.chunks)
    ok_(ds.chunks is ds.sa['chunks'].value)

    # assignment should work at all levels including 1st
    ds.targets = chunks
    assert_array_equal(ds.targets, chunks)
    ok_(ds.targets is ds.sa.targets)
    ok_(ds.targets is ds.sa['targets'].value)

    # test broadcasting
    # but not for plain scalars
    assert_raises(ValueError, ds.set_attr, 'sa.bc', 5)
    # and not for plain plain str
    assert_raises(TypeError, ds.set_attr, 'sa.bc', "mike")
    # but for any iterable of len == 1
    ds.set_attr('sa.bc', (5,))
    ds.set_attr('sa.dc', ["mike"])
    assert_array_equal(ds.sa.bc, [5] * len(ds))
    assert_array_equal(ds.sa.dc, ["mike"] * len(ds))


@reseed_rng()
def test_ex_from_masked():
    ds = Dataset.from_wizard(samples=np.atleast_2d(np.arange(5)).view(myarray),
                             targets=1, chunks=1)
    # simple sequence has to be a single pattern
    assert_equal(ds.nsamples, 1)
    # array subclass survives
    ok_(isinstance(ds.samples, myarray))

    # check correct pattern layout (1x5)
    assert_array_equal(ds.samples, [[0, 1, 2, 3, 4]])

    # check for single label and origin
    assert_array_equal(ds.targets, [1])
    assert_array_equal(ds.chunks, [1])

    # now try adding pattern with wrong shape
    assert_raises(ValueError, vstack,
                  (ds, Dataset.from_wizard(np.ones((2, 3)), targets=1, chunks=1)))

    # now add two real patterns
    ds = vstack((ds, Dataset.from_wizard(np.random.standard_normal((2, 5)),
                                         targets=2, chunks=2)))
    assert_equal(ds.nsamples, 3)
    assert_array_equal(ds.targets, [1, 2, 2])
    assert_array_equal(ds.chunks, [1, 2, 2])

    # test unique class labels
    ds = vstack((ds, Dataset.from_wizard(np.random.standard_normal((2, 5)),
                                         targets=3, chunks=5)))
    assert_array_equal(ds.sa['targets'].unique, [1, 2, 3])

    # test wrong attributes length
    assert_raises(ValueError, Dataset.from_wizard,
                  np.random.standard_normal((4, 2, 3, 4)), targets=[1, 2, 3],
                  chunks=2)
    assert_raises(ValueError, Dataset.from_wizard,
                  np.random.standard_normal((4, 2, 3, 4)), targets=[1, 2, 3, 4],
                  chunks=[2, 2, 2])

    # no test one that is using from_masked
    ds = datasets['3dlarge']
    for a in ds.sa:
        assert_equal(len(ds.sa[a].value), len(ds))
    for a in ds.fa:
        assert_equal(len(ds.fa[a].value), ds.nfeatures)


def test_shape_conversion():
    ds = Dataset.from_wizard(np.arange(24).reshape((2, 3, 4)).view(myarray),
                             targets=1, chunks=1)
    # array subclass survives
    ok_(isinstance(ds.samples, myarray))

    assert_equal(ds.nsamples, 2)
    assert_equal(ds.samples.shape, (2, 12))
    assert_array_equal(ds.samples, [range(12), range(12, 24)])


@reseed_rng()
def test_multidim_attrs():
    samples = np.arange(24).reshape(2, 3, 4)
    # have a dataset with two samples -- mapped from 2d into 1d
    # but have 2d labels and 3d chunks -- whatever that is
    ds = Dataset.from_wizard(samples.copy(),
                             targets=samples.copy(),
                             chunks=np.random.normal(size=(2, 10, 4, 2)))
    assert_equal(ds.nsamples, 2)
    assert_equal(ds.nfeatures, 12)
    assert_equal(ds.sa.targets.shape, (2, 3, 4))
    assert_equal(ds.sa.chunks.shape, (2, 10, 4, 2))

    # try slicing
    subds = ds[0]
    assert_equal(subds.nsamples, 1)
    assert_equal(subds.nfeatures, 12)
    assert_equal(subds.sa.targets.shape, (1, 3, 4))
    assert_equal(subds.sa.chunks.shape, (1, 10, 4, 2))

    # add multidim feature attr
    fattr = ds.mapper.forward(samples)
    assert_equal(fattr.shape, (2, 12))
    # should puke -- first axis is #samples
    assert_raises(ValueError, ds.fa.__setitem__, 'moresamples', fattr)
    # but that should be fine
    ds.fa['moresamples'] = fattr.T
    assert_equal(ds.fa.moresamples.shape, (12, 2))



def test_samples_shape():
    ds = Dataset.from_wizard(np.ones((10, 2, 3, 4)), targets=1, chunks=1)
    ok_(ds.samples.shape == (10, 24))

    # what happens to 1D samples
    ds = Dataset(np.arange(5))
    assert_equal(ds.shape, (5, 1))
    assert_equal(ds.nfeatures, 1)


def test_basic_datamapping():
    samples = np.arange(24).reshape((4, 3, 2)).view(myarray)

    ds = Dataset.from_wizard(samples)

    # array subclass survives
    ok_(isinstance(ds.samples, myarray))

    # mapper should end up in the dataset
    ok_(ds.a.has_key('mapper'))

    # check correct mapping
    ok_(ds.nsamples == 4)
    ok_(ds.nfeatures == 6)


def test_ds_shallowcopy():
    # lets use some instance of somewhat evolved dataset
    ds = normal_feature_dataset()
    ds.samples = ds.samples.view(myarray)

    # SHALLOW copy the beast
    ds_ = copy.copy(ds)
    # verify that we have the same data
    assert_array_equal(ds.samples, ds_.samples)
    assert_array_equal(ds.targets, ds_.targets)
    assert_array_equal(ds.chunks, ds_.chunks)

    # array subclass survives
    ok_(isinstance(ds_.samples, myarray))


    # modify and see that we actually DO change the data in both
    ds_.samples[0, 0] = 1234
    assert_array_equal(ds.samples, ds_.samples)
    assert_array_equal(ds.targets, ds_.targets)
    assert_array_equal(ds.chunks, ds_.chunks)

    ds_.sa.targets[0] = 'ab'
    ds_.sa.chunks[0] = 234
    assert_array_equal(ds.samples, ds_.samples)
    assert_array_equal(ds.targets, ds_.targets)
    assert_array_equal(ds.chunks, ds_.chunks)
    ok_(ds.sa.targets[0] == 'ab')
    ok_(ds.sa.chunks[0] == 234)

    # XXX implement me
    #ok_(np.any(ds.uniquetargets != ds_.uniquetargets))
    #ok_(np.any(ds.uniquechunks != ds_.uniquechunks))


def test_ds_deepcopy():
    # lets use some instance of somewhat evolved dataset
    ds = normal_feature_dataset()
    ds.samples = ds.samples.view(myarray)
    # Clone the beast
    ds_ = ds.copy()
    # array subclass survives
    ok_(isinstance(ds_.samples, myarray))

    # verify that we have the same data
    assert_array_equal(ds.samples, ds_.samples)
    assert_array_equal(ds.targets, ds_.targets)
    assert_array_equal(ds.chunks, ds_.chunks)

    # modify and see if we don't change data in the original one
    ds_.samples[0, 0] = 1234
    ok_(np.any(ds.samples != ds_.samples))
    assert_array_equal(ds.targets, ds_.targets)
    assert_array_equal(ds.chunks, ds_.chunks)

    ds_.sa.targets = np.hstack(([123], ds_.targets[1:]))
    ok_(np.any(ds.samples != ds_.samples))
    ok_(np.any(ds.targets != ds_.targets))
    assert_array_equal(ds.chunks, ds_.chunks)

    ds_.sa.chunks = np.hstack(([1234], ds_.chunks[1:]))
    ok_(np.any(ds.samples != ds_.samples))
    ok_(np.any(ds.targets != ds_.targets))
    ok_(np.any(ds.chunks != ds_.chunks))

    # XXX implement me
    #ok_(np.any(ds.uniquetargets != ds_.uniquetargets))
    #ok_(np.any(ds.uniquechunks != ds_.uniquechunks))

@sweepargs(dsp=datasets.items())
def test_ds_array(dsp):
    # When dataset
    dsname, ds = dsp
    if dsname != 'hollow':
        ok_(np.asarray(ds) is ds.samples,
            msg="Must have been the same on %s=%s" % dsp)
    else:
        ok_(np.asarray(ds) is not ds.samples,
            msg="Should have not been the same on %s=%s" % dsp)
    ok_(np.array(ds) is not ds.samples,
        msg="Copy should have been created on array(), %s=%s" % dsp)


def test_mergeds():
    data0 = Dataset.from_wizard(np.ones((5, 5)), targets=1)
    data0.fa['one'] = np.ones(5)
    data1 = Dataset.from_wizard(np.ones((5, 5)), targets=1, chunks=1)
    data1.fa['one'] = np.zeros(5)
    data2 = Dataset.from_wizard(np.ones((3, 5)), targets=2, chunks=1)
    data3 = Dataset.from_wizard(np.ones((4, 5)), targets=2)
    data4 = Dataset.from_wizard(np.ones((2, 5)), targets=3, chunks=2)
    data4.fa['test'] = np.arange(5)

    merged = vstack((data1.copy(), data2))

    ok_(merged.nfeatures == 5)
    l12 = [1] * 5 + [2] * 3
    l1 = [1] * 8
    ok_((merged.targets == l12).all())
    ok_((merged.chunks == l1).all())

    data_append = vstack((data1.copy(), data2))

    ok_(data_append.nfeatures == 5)
    ok_((data_append.targets == l12).all())
    ok_((data_append.chunks == l1).all())

    #
    # vstacking
    #
    if __debug__:
        # we need the same samples attributes in both datasets
        assert_raises(ValueError, vstack, (data2, data3))

        # tested only in __debug__
        assert_raises(ValueError, vstack, (data0, data1, data2, data3))
    datasets = (data1, data2, data4)
    merged = vstack(datasets)
    assert_equal(merged.shape,
                 (np.sum([len(ds) for ds in datasets]), data1.nfeatures))
    assert_true('test' in merged.fa)
    assert_array_equal(merged.sa.targets, [1] * 5 + [2] * 3 + [3] * 2)

    #
    # hstacking
    #
    assert_raises(ValueError, hstack, datasets)
    datasets = (data0, data1)
    merged = hstack(datasets)
    assert_equal(merged.shape,
                 (len(data1), np.sum([ds.nfeatures for ds in datasets])))
    assert_true('chunks' in merged.sa)
    assert_array_equal(merged.fa.one, [1] * 5 + [0] * 5)

def test_hstack():
    """Additional tests for hstacking of datasets
    """
    ds3d = datasets['3dsmall']
    nf1 = ds3d.nfeatures
    nf3 = 3 * nf1
    ds3dstacked = hstack((ds3d, ds3d, ds3d))
    ok_(ds3dstacked.nfeatures == nf3)
    for fav in ds3dstacked.fa.itervalues():
        v = fav.value
        ok_(len(v) == nf3)
        assert_array_equal(v[:nf1], v[nf1:2 * nf1])
        assert_array_equal(v[2 * nf1:], v[nf1:2 * nf1])

def test_stack_add_dataset_attributes():
    data0 = Dataset.from_wizard(np.ones((5, 5)), targets=1)
    data0.a['one'] = np.ones(2)
    data0.a['two'] = 2
    data0.a['three'] = 'three'
    data0.a['common'] = range(10)
    data0.a['array'] = np.arange(10)
    data1 = Dataset.from_wizard(np.ones((5, 5)), targets=1)
    data1.a['one'] = np.ones(3)
    data1.a['two'] = 3
    data1.a['four'] = 'four'
    data1.a['common'] = range(10)
    data1.a['array'] = np.arange(10)


    vstacker = lambda x: vstack((data0, data1), a=x)
    hstacker = lambda x: hstack((data0, data1), a=x)

    add_params = (1, None, 'unique', 'uniques', 'all', 'drop_nonunique')

    for stacker in (vstacker, hstacker):
        for add_param in add_params:
            if add_param == 'unique':
                assert_raises(DatasetError, stacker, add_param)
                continue

            r = stacker(add_param)

            if add_param == 1:
                assert_array_equal(data1.a.one, r.a.one)
                assert_equal(r.a.two, 3)
                assert_equal(r.a.four, 'four')
                assert_true('three' not in r.a.keys())
                assert_true('array' in r.a.keys())
            elif add_param == 'uniques':
                assert_equal(set(r.a.keys()),
                             set(['one', 'two', 'three',
                                  'four', 'common', 'array']))
                assert_equal(r.a.two, (2, 3))
                assert_equal(r.a.four, ('four',))
            elif add_param == 'all':
                assert_equal(set(r.a.keys()),
                             set(['one', 'two', 'three',
                                  'four', 'common', 'array']))
                assert_equal(r.a.two, (2, 3))
                assert_equal(r.a.three, ('three', None))
            elif add_param == 'drop_nonunique':
                assert_equal(set(r.a.keys()),
                             set(['common', 'three', 'four', 'array']))
                assert_equal(r.a.three, 'three')
                assert_equal(r.a.four, 'four')
                assert_equal(r.a.common, range(10))
                assert_array_equal(r.a.array, np.arange(10))


def test_unique_stack():
    data = Dataset(np.reshape(np.arange(24), (4, 6)),
                        sa=dict(x=[0, 1, 0, 1]),
                        fa=dict(y=[x for x in 'abccba']))

    sa_stack = stack_by_unique_sample_attribute(data, 'x')
    assert_equal(sa_stack.shape, (2, 12))
    assert_array_equal(sa_stack.fa.x, [0] * 6 + [1] * 6)
    assert_array_equal(sa_stack.fa.y, [x for x in 'abccbaabccba'])

    fa_stack = stack_by_unique_feature_attribute(data, 'y')
    assert_equal(fa_stack.shape, (12, 2))
    assert_array_equal(fa_stack.sa.x, [0, 1] * 6)
    assert_array_equal(fa_stack.sa.y, [y for y in 'aaaabbbbcccc'])
    #assert_array_equal(fa_stack.fa.y,[''])

    # check values match the fa or sa
    for i in xrange(4):
        for j in xrange(6):
            d = data[i, j]
            for k, other in enumerate((sa_stack, fa_stack)):
                msk = other.samples == d.samples
                ii, jj = np.nonzero(msk) # find matching indices in other

                o = other[ii, jj]
                coll = [o.fa, o.sa][k]

                assert_equal(coll.x, d.sa.x)
                assert_equal(coll.y, d.fa.y)

    ystacker = lambda y: lambda x: stack_by_unique_feature_attribute(x, y)
    assert_raises(KeyError, ystacker('z'), data)

    data.fa['z'] = [z for z in '123451']
    assert_raises(ValueError, ystacker('z'), data)

def test_mergeds2():
    """Test composition of new datasets by addition of existing ones
    """
    data = dataset_wizard([range(5)], targets=1, chunks=1)

    assert_array_equal(data.UT, [1])

    # simple sequence has to be a single pattern
    assert_equal(data.nsamples, 1)
    # check correct pattern layout (1x5)
    assert_array_equal(data.samples, [[0, 1, 2, 3, 4]])

    # check for single labels and origin
    assert_array_equal(data.targets, [1])
    assert_array_equal(data.chunks, [1])

    # now try adding pattern with wrong shape
    assert_raises(ValueError,
                  vstack,
                  (data, dataset_wizard(np.ones((2, 3)), targets=1, chunks=1)))

    # now add two real patterns
    dss = datasets['uni2large'].samples
    data = vstack((data, dataset_wizard(dss[:2, :5], targets=2, chunks=2)))
    assert_equal(data.nfeatures, 5)
    assert_array_equal(data.targets, [1, 2, 2])
    assert_array_equal(data.chunks, [1, 2, 2])

    # test automatic origins
    data = vstack((data, (dataset_wizard(dss[3:5, :5], targets=3, chunks=[0, 1]))))
    assert_array_equal(data.chunks, [1, 2, 2, 0, 1])

    # test unique class labels
    assert_array_equal(data.UT, [1, 2, 3])

    # test wrong label length
    assert_raises(ValueError, dataset_wizard, dss[:4, :5], targets=[ 1, 2, 3 ],
                                         chunks=2)

    # test wrong origin length
    assert_raises(ValueError, dataset_wizard, dss[:4, :5],
                  targets=[ 1, 2, 3, 4 ], chunks=[ 2, 2, 2 ])


def test_combined_samplesfeature_selection():
    data = dataset_wizard(np.arange(20).reshape((4, 5)).view(myarray),
                   targets=[1, 2, 3, 4],
                   chunks=[5, 6, 7, 8])

    # array subclass survives
    ok_(isinstance(data.samples, myarray))

    ok_(data.nsamples == 4)
    ok_(data.nfeatures == 5)
    sel = data[[0, 3], [1, 2]]
    ok_(sel.nsamples == 2)
    ok_(sel.nfeatures == 2)
    assert_array_equal(sel.targets, [1, 4])
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


@reseed_rng()
def test_labelpermutation_randomsampling():
    ds = vstack([Dataset.from_wizard(np.ones((5, 10)), targets=range(5), chunks=i)
                    for i in xrange(1, 6)])
    # assign some feature attributes
    ds.fa['roi'] = np.repeat(np.arange(5), 2)
    ds.fa['lucky'] = np.arange(10) % 2
    # use subclass for testing if it would survive
    ds.samples = ds.samples.view(myarray)

    ok_(ds.get_nsamples_per_attr('targets') == {0:5, 1:5, 2:5, 3:5, 4:5})
    sample = ds.random_samples(2)
    ok_(sample.get_nsamples_per_attr('targets').values() == [ 2, 2, 2, 2, 2 ])
    ok_((ds.sa['chunks'].unique == range(1, 6)).all())

@reseed_rng()
def test_masked_featureselection():
    origdata = np.random.standard_normal((10, 2, 4, 3, 5)).view(myarray)
    data = Dataset.from_wizard(origdata, targets=2, chunks=2)

    unmasked = data.samples.copy()
    # array subclass survives
    ok_(isinstance(data.samples, myarray))

    # default must be no mask
    ok_(data.nfeatures == 120)
    ok_(data.a.mapper.forward1(origdata[0]).shape == (120,))

    # check that full mask uses all features
    # this uses auto-mapping of selection arrays in __getitem__
    sel = data[:, np.ones((2, 4, 3, 5), dtype='bool')]
    ok_(sel.nfeatures == data.samples.shape[1])
    ok_(data.nfeatures == 120)

    # check partial array mask
    partial_mask = np.zeros((2, 4, 3, 5), dtype='bool')
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


@reseed_rng()
def test_origmask_extraction():
    origdata = np.random.standard_normal((10, 2, 4, 3))
    data = Dataset.from_wizard(origdata, targets=2, chunks=2)

    # check with custom mask
    sel = data[:, 5]
    ok_(sel.samples.shape[1] == 1)


@reseed_rng()
def test_feature_masking():
    mask = np.zeros((5, 3), dtype='bool')
    mask[2, 1] = True
    mask[4, 0] = True
    data = Dataset.from_wizard(np.arange(60).reshape((4, 5, 3)),
                               targets=1, chunks=1, mask=mask)

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
    #self.assertTrue(np.all(data.I == data.origids))
    assert_array_equal(data.C, data.chunks)
    assert_array_equal(data.UC, np.unique(data.chunks))
    assert_array_equal(data.T, data.targets)
    assert_array_equal(data.UT, np.unique(data.targets))
    assert_array_equal(data.S, data.samples)
    assert_array_equal(data.O, data.mapper.reverse(data.samples))


def test_origid_handling():
    ds = dataset_wizard(np.atleast_2d(np.arange(35)).T)
    ds.init_origids('both')
    ok_(ds.nsamples == 35)
    assert_equal(len(np.unique(ds.sa.origids)), 35)
    assert_equal(len(np.unique(ds.fa.origids)), 1)
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
    ds = dataset_wizard(np.arange(12).reshape((4, 3)),
                 targets=1, chunks=1)
    origid = ds.idhash
    #XXX BUG -- no assurance that labels would become an array... for now -- do manually
    ds.targets = np.array([3, 1, 2, 3])   # change all labels
    ok_(origid != ds.idhash,
                    msg="Changing all targets should alter dataset's idhash")

    origid = ds.idhash

    z = ds.targets[1]
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
    orig_labels = ds.targets #.copy()
    ds.sa.targets = range(len(ds))
    ok_(origid != ds.idhash,
        msg="Chaging attribute also changes idhash")

    ds.targets = orig_labels
    ok_(origid == ds.idhash,
        msg="idhash should be restored after reassigning orig targets")


def test_arrayattributes():
    samples = np.arange(12).reshape((4, 3))
    labels = range(4)
    chunks = [1, 1, 2, 2]
    ds = dataset_wizard(samples, labels, chunks)

    for a in (ds.samples, ds.targets, ds.chunks):
        ok_(isinstance(a, np.ndarray))

    ds.targets = labels
    ok_(isinstance(ds.targets, np.ndarray))

    ds.chunks = chunks
    ok_(isinstance(ds.chunks, np.ndarray))

    # we should allow assigning somewhat more complex
    # iterables -- use ndarray of dtype object then
    # and possibly spit out a warning
    ds.sa['complex_list'] = [[], [1], [1, 2], []]
    ok_(ds.sa.complex_list.dtype == object)

    # but incorrect length should still fail
    assert_raises(ValueError, ds.sa.__setitem__,
                  'complex_list2', [[], [1], [1, 2]])


def test_repr():
    attr_repr = "SampleAttribute(name='TestAttr', doc='my own test', " \
                                "value=array([0, 1, 2, 3, 4]), length=None)"
    sattr = SampleAttribute(name='TestAttr', doc='my own test',
                            value=np.arange(5))
    # check precise formal representation
    ok_(repr(sattr) == attr_repr)
    # check that it actually works as a Python expression
    from numpy import array
    eattr = eval(repr(sattr))
    ok_(repr(eattr), attr_repr)

    # should also work for a simple dataset
    # Does not work due to bug in numpy:
    #  python -c "from numpy import *; print __version__; r=repr(array(['s', None])); print r; eval(r)"
    # would give "array([s, None], dtype=object)" without '' around s
    #ds = datasets['uni2small']
    ds = Dataset([[0, 1]],
                 a={'dsa1': 'v1'},
                 sa={'targets': [0]},
                 fa={'targets': ['b', 'n']})
    ds_repr = repr(ds)
    cfg_repr = cfg.get('datasets', 'repr', 'full')
    if cfg_repr == 'full':
        try:
            ok_(repr(eval(ds_repr)) == ds_repr)
        except SyntaxError, e:
            raise AssertionError, "%r cannot be evaluated" % ds_repr
    elif cfg_repr == 'str':
        ok_(str(ds) == ds_repr)
    else:
        raise AssertionError('Unknown kind of datasets.repr configuration %r'
                             % cfg_repr)

def test_str():
    args = (np.arange(12, dtype=np.int8).reshape((4, 3)),
             range(4),
             [1, 1, 2, 2])
    for iargs in range(1, len(args)):
        ds = dataset_wizard(*(args[:iargs]))
        ds_s = str(ds)
        ok_(ds_s.startswith('<Dataset: 4x3@int8'))
        ok_(ds_s.endswith('>'))

def is_bsr(x):
    """Helper function to check if instance is bsr_matrix if such is
    avail at all
    """
    import scipy.sparse as sparse
    return hasattr(sparse, 'bsr_matrix') and isinstance(x, sparse.bsr_matrix)

def test_other_samples_dtypes():
    skip_if_no_external('scipy')
    import scipy.sparse as sparse
    dshape = (4, 3)
    # test for ndarray, custom ndarray-subclass, matrix,
    # and all sparse matrix types we know
    stypes = [np.arange(np.prod(dshape)).reshape(dshape),
              np.arange(np.prod(dshape)).reshape(dshape).view(myarray),
              np.matrix(np.arange(np.prod(dshape)).reshape(dshape)),
              sparse.csc_matrix(np.arange(np.prod(dshape)).reshape(dshape)),
              sparse.csr_matrix(np.arange(np.prod(dshape)).reshape(dshape))]
    if hasattr(sparse, 'bsr_matrix'):
        stypes += [
              # BSR cannot be sliced, but is more efficient for sparse
              # arithmetic operations than CSC pr CSR
              sparse.bsr_matrix(np.arange(np.prod(dshape)).reshape(dshape))]
              # LIL and COO are best for constructing matrices, not for
              # doing something with them
              #sparse.lil_matrix(np.arange(np.prod(dshape)).reshape(dshape)),
              #sparse.coo_matrix(np.arange(np.prod(dshape)).reshape(dshape)),
              # DOK doesn't allow duplicates and is bad at array-like slicing
              #sparse.dok_matrix(np.arange(np.prod(dshape)).reshape(dshape)),
              # DIA only has diagonal storage and cannot be sliced
              #sparse.dia_matrix(np.arange(np.prod(dshape)).reshape(dshape))]

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
            assert_raises(RuntimeError, np.mean, ds)
        else:
            # need to convert results, since matrices return matrices
            assert_array_equal(np.mean(ds, axis=0),
                               np.array(np.mean(ds.samples, axis=0)).squeeze())

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
                assert_array_equal(sel.samples[:, 1].todense(),
                        ds.samples[:, 2].todense())
            else:
                assert_array_equal(sel.samples[:, 1],
                        ds.samples[:, 2])


        # what we don't do
        class voodoo:
            dtype = 'fancy'
        # voodoo
        assert_raises(ValueError, Dataset, voodoo())
        # crippled
        assert_raises(ValueError, Dataset, np.array(5))

        # things that might behave in surprising ways
        # lists -- first axis is samples, hence single feature
        ds = Dataset(range(5))
        assert_equal(ds.nfeatures, 1)
        assert_equal(ds.shape, (5, 1))
        # arrays of objects
        data = np.array([{}, {}])
        ds = Dataset(data)
        assert_equal(ds.shape, (2, 1))
        assert_equal(ds.nsamples, 2)
        # Nothing to index, hence no features
        assert_equal(ds.nfeatures, 1)


@sweepargs(ds=datasets.values() + [
    Dataset(np.array([None], dtype=object)),
    dataset_wizard(np.arange(3), targets=['a', 'bc', 'd'], chunks=np.arange(3)),
    dataset_wizard(np.arange(4), targets=['a', 'bc', 'a', 'bc'], chunks=[1, 1, 2, 2]),
    ])
def test_dataset_summary(ds):
    s = ds.summary()
    ok_(s.startswith(str(ds)[1:-1])) # we strip surrounding '<...>'
    # TODO: actual test of what was returned; to do that properly
    #       RF the summary() so it is a dictionary

    summaries = []
    if 'targets' in ds.sa:
        summaries += ['Sequence statistics']
        if 'chunks' in ds.sa:
            summaries += ['Summary for targets', 'Summary for chunks']

    # By default we should get all kinds of summaries
    if not 'Number of unique targets >' in s:
        for summary in summaries:
            ok_(summary in s)

    # If we give "wrong" targets_attr we should see none of summaries
    s2 = ds.summary(targets_attr='bogus')
    for summary in summaries:
        ok_(not summary in s2)

@nodebug(['ID_IN_REPR', 'MODULE_IN_REPR'])
@with_tempfile(suffix='.hdf5')
def test_h5py_io(dsfile):
    skip_if_no_external('h5py')

    # store random dataset to file
    ds = datasets['3dlarge']
    ds.save(dsfile)

    # reload and check for identity
    ds2 = Dataset.from_hdf5(dsfile)
    assert_array_equal(ds.samples, ds2.samples)
    for attr in ds.sa:
        assert_array_equal(ds.sa[attr].value, ds2.sa[attr].value)
    for attr in ds.fa:
        assert_array_equal(ds.fa[attr].value, ds2.fa[attr].value)
    assert_true(len(ds.a.mapper), 2)

    # since we have no __equal__ do at least some comparison
    assert_equal(repr(ds.a.mapper), repr(ds2.a.mapper))

    if __debug__:
        # debug mode needs special test as it enhances the repr output
        # with module info and id() appendix for objects
        #
        # INCORRECT slicing (:-1) since without any hash it results in
        # empty list -- moreover we seems of not reporting ids with #
        # any longer
        #
        #assert_equal('#'.join(repr(ds.a.mapper).split('#')[:-1]),
        #             '#'.join(repr(ds2.a.mapper).split('#')[:-1]))
        pass


def test_all_equal():
    # all these values are supposed to be different from each other
    # but equal to themselves
    a = np.random.normal(size=(10, 10)) + 1000.
    b = np.zeros((10, 10))
    c = np.zeros(10)
    d = np.zeros(11)
    e = 0
    f = None
    g = True
    h = ''
    i = 'a'

    values = [a, b, c, d, e, f, g, h, i]
    for ii, v in enumerate(values):
        for jj, w in enumerate(values):
            assert_equal(all_equal(v, w), ii == jj)

    # ensure that this function behaves like the 
    # standard python '==' comparator for singulars
    singulars = [0, None, True, False, '', 1, 'a']
    for v in singulars:
        for w in singulars:
            assert_equal(all_equal(v, w), v == w)


def test_hollow_samples():
    sshape = (10, 5)
    ds = Dataset(HollowSamples(sshape, dtype=int),
                 sa={'targets': np.tile(['one', 'two'], sshape[0] / 2)})
    assert_equal(ds.shape, sshape)
    assert_equal(ds.samples.dtype, int)
    # should give us features [1,3] and samples [2,3,5]
    mds = ds[[2, 3, 5], 1::2]
    assert_array_equal(mds.samples.sid, [2, 3, 5])
    assert_array_equal(mds.samples.fid, [1, 3])
    assert_equal(mds.shape, (3, 2))
    assert_equal(ds.samples.dtype, mds.samples.dtype)
    # orig should stay pristine
    assert_equal(ds.samples.dtype, int)
    assert_equal(ds.shape, sshape)

def test_assign_sa():
    # https://github.com/PyMVPA/PyMVPA/issues/149
    ds = Dataset(np.arange(6).reshape((2,-1)), sa=dict(targets=range(2)))
    ds.sa['task'] = ds.sa['targets']
    # so it should be a new collectable now
    assert_equal(ds.sa['task'].name, 'task')
    assert_equal(ds.sa['targets'].name, 'targets') # this lead to issue reported in 149
    assert('task' in ds.sa.keys())
    assert('targets' in ds.sa.keys())
    ds1 = ds[:, 1]
    assert('task' in ds1.sa.keys())
    assert('targets' in ds1.sa.keys()) # issue reported in 149
    assert_equal(ds1.sa['task'].name, 'task')
    assert_equal(ds1.sa['targets'].name,'targets')
