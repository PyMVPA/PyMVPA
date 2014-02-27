# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for generators."""

import itertools
import numpy as np

from mvpa2.testing.tools import ok_, assert_array_equal, assert_true, \
        assert_false, assert_equal, assert_raises, assert_almost_equal, \
        reseed_rng, assert_not_equal

from mvpa2.datasets import dataset_wizard, Dataset
from mvpa2.generators.splitters import Splitter
from mvpa2.base.node import ChainNode
from mvpa2.generators.partition import OddEvenPartitioner, NFoldPartitioner, \
     ExcludeTargetsCombinationsPartitioner
from mvpa2.generators.permutation import AttributePermutator
from mvpa2.generators.base import  Repeater, Sifter
from mvpa2.generators.resampling import Balancer
from mvpa2.misc.support import get_nelements_per_value


def give_data():
    # 100x10, 10 chunks, 4 targets
    return dataset_wizard(np.random.normal(size=(100,10)),
                          targets=[ i%4 for i in range(100) ],
                          chunks=[ i//10 for i in range(100)])


@reseed_rng()
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


@reseed_rng()
def test_partitionmapper():
    ds = give_data()
    oep = OddEvenPartitioner()
    parts = list(oep.generate(ds))
    assert_equal(len(parts), 2)
    for i, p in enumerate(parts):
        assert_array_equal(p.sa['partitions'].unique, [1, 2])
        assert_equal(p.a.partitions_set, i)
        assert_equal(len(p), len(ds))


@reseed_rng()
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

    # verify that implausible assure=True would not work
    permutation = AttributePermutator('targets', limit='ids', assure=True)
    assert_raises(RuntimeError, permutation, ds)

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
    permutation = AttributePermutator(['targets', 'ids'],
                                      assure=True, count=nruns)
    pds = list(permutation.generate(ds))
    assert_equal(len(pds), nruns)
    for p in pds:
        assert_false(np.all(p.sa.ids == ds.sa.ids))

    # permute feature attrs
    ds.fa['ids'] = range(ds.shape[1])
    permutation = AttributePermutator('fa.ids', assure=True)
    pds = permutation(ds)
    assert_false(np.all(pds.fa.ids == ds.fa.ids))

    # now chunk-wise uattrs strategy (reassignment)
    permutation = AttributePermutator('targets', limit='chunks',
                                      strategy='uattrs', assure=True)
    pds = permutation(ds)
    # Due to assure above -- we should have changed things
    assert_not_equal(zip(ds.targets), zip(pds.targets))
    # in each chunk we should have unique remappings
    for c in ds.UC:
        chunk_idx = ds.C == c
        otargets, ptargets = ds.targets[chunk_idx], pds.sa.targets[chunk_idx]
        # we still have the same targets
        assert_equal(set(ptargets), set(otargets))
        # we have only 1-to-1 mappings
        assert_true(len(set(zip(otargets, ptargets))), len(set(otargets)))

    ds.sa['odds'] = ds.sa.ids % 2
    # test combinations
    permutation = AttributePermutator(['targets', 'odds'], limit='chunks',
                                       strategy='uattrs', assure=True)
    pds = permutation(ds)
    # Due to assure above -- we should have changed things
    assert_not_equal(zip(ds.targets,   ds.sa.odds),
                     zip(pds.targets, pds.sa.odds))
    # In each chunk we should have unique remappings
    for c in ds.UC:
        chunk_idx = ds.C == c
        otargets, ptargets = ds.targets[chunk_idx], pds.sa.targets[chunk_idx]
        oodds, podds = ds.sa.odds[chunk_idx], pds.sa.odds[chunk_idx]
        # we still have the same targets
        assert_equal(set(ptargets), set(otargets))
        assert_equal(set(oodds), set(podds))
        # at the end we have the same mapping
        assert_equal(set(zip(otargets, oodds)), set(zip(ptargets, podds)))

@reseed_rng()
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
    # same but include all offlimit samples
    bal = Balancer(limit={'chunks': 3}, include_offlimit=True,
                   apply_selection=True)
    res = bal(ds)
    assert_array_equal(res.sa['chunks'].unique, range(10))
    # chunk three still balanced, but the rest is not, i.e. all samples included
    assert_equal(get_nelements_per_value(res[res.sa.chunks == 3].sa.targets).values(),
                 [2] * 4)
    assert_equal(get_nelements_per_value(res.sa.chunks).values(),
                 [10, 10, 10, 8, 10, 10, 10, 10, 10, 10])
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


def test_repeater():
    reps = 4
    r = Repeater(reps, space='OMG')
    dsl = [ds for ds in r.generate(Dataset([0,1]))]
    assert_equal(len(dsl), reps)
    for i, ds in enumerate(dsl):
        assert_equal(ds.a.OMG, i)

def test_sifter():
    # somewhat duplicating the doctest
    ds = Dataset(samples=np.arange(8).reshape((4,2)),
                 sa={'chunks':   [ 0 ,  1 ,  2 ,  3 ],
                     'targets':  ['c', 'c', 'p', 'p']})
    for sift_targets_definition in (['c', 'p'],
                                    dict(uvalues=['c', 'p'])):
        par = ChainNode([NFoldPartitioner(cvtype=2, attr='chunks'),
                         Sifter([('partitions', 2),
                                 ('targets', sift_targets_definition)])
                         ])
        dss = list(par.generate(ds))
        assert_equal(len(dss), 4)
        for ds_ in dss:
            testing = ds[ds_.sa.partitions == 2]
            assert_array_equal(np.unique(testing.sa.targets), ['c', 'p'])
            # and we still have both targets  present in training
            training = ds[ds_.sa.partitions == 1]
            assert_array_equal(np.unique(training.sa.targets), ['c', 'p'])

def test_sifter_with_balancing():
    # extended previous test which was already
    # "... somewhat duplicating the doctest"
    ds = Dataset(samples=np.arange(12).reshape((-1, 2)),
                 sa={'chunks':   [ 0 ,  1 ,  2 ,  3 ,  4,   5 ],
                     'targets':  ['c', 'c', 'c', 'p', 'p', 'p']})

    # Without sifter -- just to assure that we do get all of them
    # i.e. 6*5*4*3/(4!) = 15
    par = ChainNode([NFoldPartitioner(cvtype=4, attr='chunks')])
    assert_equal(len(list(par.generate(ds))), 15)

    # so we will take 4 chunks out of available 7, but would care only
    # about those partitions where we have balanced number of 'c' and 'p'
    # entries
    assert_raises(ValueError,
                  lambda x: list(Sifter([('targets', dict(wrong=1))]).generate(x)),
                  ds)

    par = ChainNode([NFoldPartitioner(cvtype=4, attr='chunks'),
                     Sifter([('partitions', 2),
                             ('targets',
                              dict(uvalues=['c', 'p'],
                                   balanced=True))])
                     ])
    dss = list(par.generate(ds))
    # print [ x[x.sa.partitions==2].sa.targets for x in dss ]
    assert_equal(len(dss), 9)
    for ds_ in dss:
        testing = ds[ds_.sa.partitions == 2]
        assert_array_equal(np.unique(testing.sa.targets), ['c', 'p'])
        # and we still have both targets  present in training
        training = ds[ds_.sa.partitions == 1]
        assert_array_equal(np.unique(training.sa.targets), ['c', 'p'])

def test_exclude_targets_combinations():
    partitioner = ChainNode([NFoldPartitioner(),
                             ExcludeTargetsCombinationsPartitioner(
                                 k=2,
                                 targets_attr='targets',
                                 space='partitions')],
                            space='partitions')
    from mvpa2.misc.data_generators import normal_feature_dataset
    ds = normal_feature_dataset(snr=0., nlabels=4, perlabel=3, nchunks=3,
                                nonbogus_features=[0,1,2,3], nfeatures=4)
    partitions = list(partitioner.generate(ds))
    assert_equal(len(partitions), 3 * 6)
    splitter = Splitter('partitions')
    combs = []
    comb_chunks = []
    for p in partitions:
        trds, teds = list(splitter.generate(p))[:2]
        comb = tuple(np.unique(teds.targets))
        combs.append(comb)
        comb_chunks.append(comb + tuple(np.unique(teds.chunks)))
    assert_equal(len(set(combs)), 6)         # just 6 possible combinations of 2 out of 4
    assert_equal(len(set(comb_chunks)), 3*6) # all unique


def test_exclude_targets_combinations_subjectchunks():
    partitioner = ChainNode([NFoldPartitioner(attr='subjects'),
                             ExcludeTargetsCombinationsPartitioner(
                                 k=1,
                                 targets_attr='chunks',
                                 space='partitions')],
                            space='partitions')
    # targets do not need even to be defined!
    ds = Dataset(np.arange(18).reshape(9, 2),
                 sa={'chunks': np.arange(9) // 3,
                     'subjects': np.arange(9) % 3})
    dss = list(partitioner.generate(ds))
    assert_equal(len(dss), 9)

    testing_subjs, testing_chunks = [], []
    for ds_ in dss:
        testing_partition = ds_.sa.partitions == 2
        training_partition = ds_.sa.partitions == 1
        # must be scalars -- so implicit test here
        # if not -- would be error
        testing_subj = np.asscalar(np.unique(ds_.sa.subjects[testing_partition]))
        testing_subjs.append(testing_subj)
        testing_chunk = np.asscalar(np.unique(ds_.sa.chunks[testing_partition]))
        testing_chunks.append(testing_chunk)
        # and those must not appear for training
        ok_(not testing_subj in ds_.sa.subjects[training_partition])
        ok_(not testing_chunk in ds_.sa.chunks[training_partition])
    # and we should have gone through all chunks/subjs pairs
    testing_pairs = set(zip(testing_subjs, testing_chunks))
    assert_equal(len(testing_pairs), 9)
    # yoh: equivalent to set(itertools.product(range(3), range(3))))
    #      but .product is N/A for python2.5
    assert_equal(testing_pairs, set(zip(*np.where(np.ones((3,3))))))
