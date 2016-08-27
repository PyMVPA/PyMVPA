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
from time import time

from mvpa2.testing.tools import ok_, assert_array_equal, assert_true, \
        assert_false, assert_equal, assert_raises, assert_almost_equal, \
        reseed_rng, assert_not_equal, assert_in, assert_not_in
from mvpa2.testing.tools import assert_warnings
from mvpa2.testing.tools import assert_datasets_equal

from mvpa2.datasets import dataset_wizard, Dataset
from mvpa2.generators.splitters import Splitter
from mvpa2.base.node import ChainNode
from mvpa2.generators.partition import OddEvenPartitioner, NFoldPartitioner, \
     ExcludeTargetsCombinationsPartitioner, FactorialPartitioner
from mvpa2.generators.permutation import AttributePermutator
from mvpa2.generators.base import  Repeater, Sifter
from mvpa2.generators.resampling import Balancer
from mvpa2.misc.data_generators import normal_feature_dataset
from mvpa2.misc.support import get_nelements_per_value


def give_data():
    # 100x10, 10 chunks, 4 targets
    return dataset_wizard(np.random.normal(size=(100, 10)),
                          targets=[i % 4 for i in range(100)],
                          chunks=[i//10 for i in range(100)])


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

    # Was about to use borrowkwargs but didn't work out . Test doesn't hurt
    doc = AttributePermutator.__init__.__doc__
    assert_in('limit : ', doc)
    assert_not_in('collection : ', doc)

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
    def assert_all_different_permutations(pds):
        assert_equal(len(pds), nruns)
        for i, p in enumerate(pds):
            assert_false(np.all(p.sa.ids == ds.sa.ids))
            for p_ in pds[i+1:]:
                assert_false(np.all(p.sa.ids == p_.sa.ids))

    permutation = AttributePermutator(['targets', 'ids'],
                                      assure=True, count=nruns)
    pds = list(permutation.generate(ds))
    assert_all_different_permutations(pds)

    # if we provide seeding, and generate, it should also return different datasets
    permutation = AttributePermutator(['targets', 'ids'],
                                      count=nruns, rng=1)
    pds1 = list(permutation.generate(ds))
    assert_all_different_permutations(pds)

    # but if we regenerate -- should all be the same to before
    pds2 = list(permutation.generate(ds))
    assert_equal(len(pds1), len(pds2))
    for p1, p2 in zip(pds1, pds2):
        assert_datasets_equal(p1, p2)

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
    ds.sa['ids'] = np.arange(len(ds))  # some sa to ease tracking of samples

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

    # if we rerun again, it would be a different selection
    res2 = bal(ds)
    assert_true(np.any(res.sa.ids != bal(ds).sa.ids))

    # but if we create a balancer providing seed rng int,
    # should be identical results
    bal = Balancer(apply_selection=True, count=5, rng=1)
    assert_false(np.any(bal(ds).sa.ids != bal(ds).sa.ids))

    # But results should differ if we use .generate to produce those multiple
    # balanced datasets
    b = Balancer(apply_selection=True, count=3, rng=1)
    balanced = list(b.generate(ds))
    assert_false(all(balanced[0].sa.ids == balanced[1].sa.ids))
    assert_false(all(balanced[0].sa.ids == balanced[2].sa.ids))
    assert_false(all(balanced[1].sa.ids == balanced[2].sa.ids))

    # And should be exactly the same
    for ds_a, ds_b in zip(balanced, b.generate(ds)):
        assert_datasets_equal(ds_a, ds_b)

    # Contribution by Chris Markiewicz
    # And interleaving __call__ and generator fetches
    gen1 = b.generate(ds)
    gen2 = b.generate(ds)

    seq1, seq2, seq3 = [], [], []

    for i in xrange(3):
        seq1.append(gen1.next())
        seq2.append(gen2.next())
        seq3.append(b(ds))

    # Produces expected sequences

    for i in xrange(3):
        assert_datasets_equal(balanced[i], seq1[i])
        assert_datasets_equal(balanced[i], seq2[i])

    # And all __call__s return the same result
    ds_a = seq3[0]
    for ds_b in seq3[1:]:
        assert_array_equal(ds_a.sa.ids, ds_b.sa.ids)

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
    ds.fa['one'] = np.tile([1, 2], 5)
    ds.fa['chk'] = np.repeat([1, 2], 5)
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


def test_permute_chunks():

    def is_sorted(x):
        return np.array_equal(np.sort(x), x)

    ds = give_data()
    # change targets labels
    # there is no target labels permuting within chunks,
    # assure = True would be error
    ds.sa['targets'] = range(len(ds.sa.targets))
    permutation = AttributePermutator(attr='targets',
                                      chunk_attr='chunks',
                                      strategy='chunks',
                                      assure=True)

    pds = permutation(ds)

    assert_false(is_sorted(pds.sa.targets))
    assert_true(np.array_equal(pds.samples, ds.samples))
    for chunk_id in np.unique(pds.sa.chunks):
        chunk_ds = pds[pds.sa.chunks == chunk_id]
        assert_true(is_sorted(chunk_ds.sa.targets))

    permutation = AttributePermutator(attr='targets',
                                      strategy='chunks')
    assert_raises(ValueError, permutation, ds)


def test_factorialpartitioner():
    # Test against sifter and chainmap implemented in test_usecases
    # -- code below copied from test_usecases --
    # Let's simulate the beast -- 6 categories total groupped into 3
    # super-ordinate, and actually without any 'superordinate' effect
    # since subordinate categories independent
    ds = normal_feature_dataset(nlabels=6,
                                snr=100,   # pure signal! ;)
                                perlabel=30,
                                nfeatures=6,
                                nonbogus_features=range(6),
                                nchunks=5)
    ds.sa['subord'] = ds.sa.targets.copy()
    ds.sa['superord'] = ['super%d' % (int(i[1])%3,)
                         for i in ds.targets]   # 3 superord categories
    # let's override original targets just to be sure that we aren't relying on them
    ds.targets[:] = 0

    # let's make two other datasets to test later
    # one superordinate category only
    ds_1super = ds.copy()
    ds_1super.sa['superord'] = ['super1' for i in ds_1super.targets]

    # one superordinate category has only one subordinate
    #ds_unbalanced = ds.copy()
    #nsuper1 = np.sum(ds_unbalanced.sa.superord == 'super1')
    #mask_superord = ds_unbalanced.sa.superord == 'super1'
    #uniq_subord = np.unique(ds_unbalanced.sa.subord[mask_superord])
    #ds_unbalanced.sa.subord[mask_superord] = [uniq_subord[0] for i in range(nsuper1)]
    ds_unbalanced = Dataset(range(4), sa={'subord': [0, 0, 1, 2],
                                          'superord': [1, 1, 2, 2]})

    npart = ChainNode([
        ## so we split based on superord
        NFoldPartitioner(len(ds.sa['superord'].unique),
                         attr='subord'),
        ## so it should select only those splits where we took 1 from
        ## each of the superord categories leaving things in balance
        Sifter([('partitions', 2),
                ('superord',
                 { 'uvalues': ds.sa['superord'].unique,
                   'balanced': True})
                ]),
    ], space='partitions')

    def partition(partitioner, ds_=ds):
        return [p.sa.partitions for p in partitioner.generate(ds_)]

    # now the new implementation
    # common kwargs
    factkw = dict(partitioner=NFoldPartitioner(attr='subord'), attr='superord')

    fpart = FactorialPartitioner(**factkw)
    p_npart = partition(npart)
    p_fpart = partition(fpart)

    assert_array_equal(np.sort(p_npart), np.sort(p_fpart))

    fpart2 = FactorialPartitioner(count=2, selection_strategy='first', **factkw)
    p_fpart2 = partition(fpart2)
    assert_equal(len(p_fpart), 8)
    assert_equal(len(p_fpart2), 2)
    assert_array_equal(p_fpart[:2], p_fpart2)

    # 1 equidistant -- should be the first one
    fpart1 = FactorialPartitioner(count=1, **factkw)
    p_fpart1 = partition(fpart1)
    assert_equal(len(p_fpart1), 1)
    assert_array_equal(p_fpart[:1], p_fpart1)

    # 2 equidistant
    fpart2 = FactorialPartitioner(count=2, **factkw)
    p_fpart2 = partition(fpart2)
    assert_equal(len(p_fpart2), 2)
    assert_array_equal(p_fpart[::4], p_fpart2)

    # without count -- should be all of them in original order
    fpartr = FactorialPartitioner(selection_strategy='random', **factkw)
    assert_array_equal(p_fpart, partition(fpartr))

    # but if with a count we should get some selection
    fpartr2 = FactorialPartitioner(selection_strategy='random', count=2, **factkw)
    # Let's generate a number of random selections:
    rand2_partitions = [partition(fpartr2) for i in xrange(10)]
    for p in rand2_partitions:
        assert_equal(len(p), 2)
    # majority of them must be different
    assert len(set([tuple(map(tuple, x)) for x in rand2_partitions])) >= 5

    # now let's check it behaves correctly if we have only one superord class
    nfold = NFoldPartitioner(attr='subord')
    p_nfold = partition(nfold, ds_1super)
    p_fpart = partition(fpart, ds_1super)
    assert_array_equal(np.sort(p_nfold), np.sort(p_fpart))

    # smoke test for unbalanced subord classes
    warning_msg = 'One or more superordinate attributes do not have the same '\
                  'number of subordinate attributes. This could yield to '\
                  'unbalanced partitions.'
    with assert_warnings([(RuntimeWarning, warning_msg)]):
        p_fpart = partition(fpart, ds_unbalanced)

    p_unbalanced = [np.array([2, 2, 2, 1]), np.array([2, 2, 1, 2])]
    superord_unbalanced = [([2], [1, 1, 2]), ([2], [1, 1, 2])]
    subord_unbalanced = [([2], [0, 0, 1]), ([1], [0, 0, 2])]

    for out_part, true_part, super_out, sub_out in \
            zip(p_fpart, p_unbalanced,
                superord_unbalanced, subord_unbalanced):
        assert_array_equal(out_part, true_part)
        assert_array_equal((ds_unbalanced[out_part == 1].sa.superord.tolist(),
                            ds_unbalanced[out_part == 2].sa.superord.tolist()),
                           super_out)
        assert_array_equal((ds_unbalanced[out_part == 1].sa.subord.tolist(),
                            ds_unbalanced[out_part == 2].sa.subord.tolist()),
                           sub_out)

    # now let's test on a dummy dataset
    ds_dummy = Dataset(range(4), sa={'subord': range(4),
                                     'superord': [1,2]*2})
    p_fpart = partition(fpart, ds_dummy)
    assert_array_equal(p_fpart,
                       [[2, 2, 1, 1],
                        [2, 1, 1, 2],
                        [1, 2, 2, 1],
                        [1, 1, 2, 2]])

def test_factorialpartitioner_big():
    # just to see that we can cope with relatively large datasets/numbers
    ds = normal_feature_dataset(nlabels=6,
                                perlabel=66,
                                nfeatures=2,
                                nchunks=11)

    # and now let's do factorial partitioner

    def partition(ds_=ds, **kwargs):
        partitioner = FactorialPartitioner(
            partitioner=NFoldPartitioner(attr='targets'),
            attr='chunks',
            **kwargs)
        return [p.sa.partitions for p in partitioner.generate(ds_)]

    # prohibitively large
    # print len(partition(ds))
    t0 = time()
    assert_equal(len(partition(ds, count=2, selection_strategy='first')), 2)
    # Those time limits are really a stretch. on a any reasonable box not too busy
    # should be done in fraction of a second, but allow to catch "naive"
    # implementation
    assert(time() - t0 < 3)

    assert_equal(len(partition(ds, count=2, selection_strategy='random')), 2)
    assert(time() - t0 < 3)
