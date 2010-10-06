# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for generators."""


import numpy as np

from mvpa.testing.tools import ok_, assert_array_equal, assert_true, \
        assert_false, assert_equal, assert_raises, assert_almost_equal

from mvpa.datasets import dataset_wizard
from mvpa.generators.splitters import Splitter
from mvpa.base.node import ChainNode
from mvpa.generators.partition import OddEvenPartitioner
from mvpa.generators.permutation import AttributePermutator
from mvpa.generators.resampling import Balancer
from mvpa.misc.support import get_nelements_per_value


def give_data():
    # 100x10, 10 chunks, 4 targets
    return dataset_wizard(np.random.normal(size=(100,10)),
                          targets=[ i%4 for i in range(100) ],
                          chunks=[ i/10 for i in range(100)])


def test_splitter():
    ds = give_data()
    # split with defaults
    spl1 = Splitter('chunks')
    assert_raises(NotImplementedError, spl1, ds)

    splits = list(spl1.generate(ds))
    assert_equal(len(splits), len(ds.sa['chunks'].unique))

    for split in splits:
        # it should have perform basic slicing!
        assert_true(split.samples.base is ds.samples)
        assert_equal(len(split.sa['chunks'].unique), 1)
        assert_true('lastsplit' in split.a)
    assert_true(splits[-1].a.lastsplit)

    # now again, more customized
    spl2 = Splitter('targets', attr_values = [0,1,1,2,3,3,3], count=4,
                   noslicing=True)
    splits = list(spl2.generate(ds))
    assert_equal(len(splits), 4)
    for split in splits:
        # it should NOT have perform basic slicing!
        assert_false(split.samples.base is ds.samples)
        assert_equal(len(split.sa['targets'].unique), 1)
        assert_equal(len(split.sa['chunks'].unique), 10)
    assert_true(splits[-1].a.lastsplit)

    # two should be identical
    assert_array_equal(splits[1].samples, splits[2].samples)

    # now go wild and split by feature attribute
    ds.fa['roi'] = np.repeat([0,1], 5)
    # splitter should auto-detect that this is a feature attribute
    spl3 = Splitter('roi')
    splits = list(spl3.generate(ds))
    assert_equal(len(splits), 2)
    for split in splits:
        assert_true(split.samples.base is ds.samples)
        assert_equal(len(split.fa['roi'].unique), 1)
        assert_equal(split.shape, (100, 5))

    # and finally test chained splitters
    cspl = ChainNode([spl2, spl3, spl1])
    splits = list(cspl.generate(ds))
    # 4 target splits and 2 roi splits each and 10 chunks each
    assert_equal(len(splits), 80)


def test_partitionmapper():
    ds = give_data()
    oep = OddEvenPartitioner()
    parts = list(oep.generate(ds))
    assert_equal(len(parts), 2)
    for i, p in enumerate(parts):
        assert_array_equal(p.sa['partitions'].unique, [1, 2])
        assert_equal(p.a.partitions_set, i)
        assert_equal(len(p), len(ds))


def test_attrpermute():
    ds = give_data()
    ds.sa['ids'] = range(len(ds))
    pristine_data = ds.samples.copy()
    permutation = AttributePermutator(['targets', 'ids'], assure=True)
    pds = permutation(ds)
    # should not touch the data
    assert_array_equal(pristine_data, pds.samples)
    # even keep the very same array
    assert_true(pds.samples.base is ds.samples)
    # there is no way that it can be the same attribute
    assert_false(np.all(pds.sa.ids == ds.sa.ids))
    # ids should reflect permutation setup
    assert_array_equal(pds.sa.targets, ds.sa.targets[pds.sa.ids])
    # other attribute should remain intact
    assert_array_equal(pds.sa.chunks, ds.sa.chunks)

    # now chunk-wise permutation
    permutation = AttributePermutator('ids', limit='chunks')
    pds = permutation(ds)
    # first ten should remain first ten
    assert_false(np.any(pds.sa.ids[:10] > 9))

    # same thing, but only permute single chunk
    permutation = AttributePermutator('ids', limit={'chunks': 3})
    pds = permutation(ds)
    # one chunk should change
    assert_false(np.any(pds.sa.ids[30:40] > 39))
    assert_false(np.any(pds.sa.ids[30:40] < 30))
    # the rest not
    assert_array_equal(pds.sa.ids[:30], range(30))

    # or a list of chunks
    permutation = AttributePermutator('ids', limit={'chunks': [3,4]})
    pds = permutation(ds)
    # two chunks should change
    assert_false(np.any(pds.sa.ids[30:50] > 49))
    assert_false(np.any(pds.sa.ids[30:50] < 30))
    # the rest not
    assert_array_equal(pds.sa.ids[:30], range(30))

    # and now try generating more permutations
    nruns = 2
    permutation = AttributePermutator(['targets', 'ids'], assure=True, count=nruns)
    pds = list(permutation.generate(ds))
    assert_equal(len(pds), nruns)
    for p in pds:
        assert_false(np.all(p.sa.ids == ds.sa.ids))

    # permute feature attrs
    ds.fa['ids'] = range(ds.shape[1])
    permutation = AttributePermutator('fa.ids', assure=True)
    pds = permutation(ds)
    assert_false(np.all(pds.fa.ids == ds.fa.ids))


def test_balancer():
    ds = give_data()
    # only mark the selection in an attribute
    bal = Balancer()
    res = bal(ds)
    # we get a new dataset, with shared samples
    assert_false(ds is res)
    assert_true(ds.samples is res.samples.base)
    # should kick out 2 samples in each chunk of 10
    assert_almost_equal(np.mean(res.sa.balanced_set), 0.8)
    # same as above, but actually apply the selection
    bal = Balancer(apply_selection=True, count=5)
    # just run it once
    res = bal(ds)
    # we get a new dataset, with shared samples
    assert_false(ds is res)
    # should kick out 2 samples in each chunk of 10
    assert_equal(len(res), int(0.8 * len(ds)))
    # now use it as a generator
    dses = list(bal.generate(ds))
    assert_equal(len(dses), 5)
    # with limit
    bal = Balancer(limit={'chunks': 3}, apply_selection=True)
    res = bal(ds)
    assert_equal(res.sa['chunks'].unique, (3,))
    assert_equal(get_nelements_per_value(res.sa.targets).values(),
                 [2] * 4)
    # fixed amount
    bal = Balancer(amount=1, limit={'chunks': 3}, apply_selection=True)
    res = bal(ds)
    assert_equal(get_nelements_per_value(res.sa.targets).values(),
                 [1] * 4)
    # fraction
    bal = Balancer(amount=0.499, limit=None, apply_selection=True)
    res = bal(ds)
    assert_array_equal(
            np.round(np.array(get_nelements_per_value(ds.sa.targets).values()) * 0.5),
            np.array(get_nelements_per_value(res.sa.targets).values()))
    # check on feature attribute
    ds.fa['one'] = np.tile([1,2], 5)
    ds.fa['chk'] = np.repeat([1,2], 5)
    bal = Balancer(attr='one', amount=2, limit='chk', apply_selection=True)
    res = bal(ds)
    assert_equal(get_nelements_per_value(res.fa.one).values(),
                 [4] * 2)
