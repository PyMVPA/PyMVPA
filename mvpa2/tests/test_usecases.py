# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for various use cases users reported mis-behaving"""

import unittest
import numpy as np

from mvpa2.testing.tools import ok_, assert_array_equal, assert_true, \
        assert_false, assert_equal, assert_not_equal, reseed_rng

@reseed_rng()
def _test_mcasey20120222():
    # http://lists.alioth.debian.org/pipermail/pkg-exppsy-pymvpa/2012q1/002034.html

    # This one is conditioned on allowing # of samples to be changed
    # by the mapper provided to MappedClassifier.  See
    # https://github.com/yarikoptic/PyMVPA/tree/_tent/allow_ch_nsamples

    import numpy as np
    from mvpa2.datasets.base import dataset_wizard
    from mvpa2.generators.partition import NFoldPartitioner
    from mvpa2.mappers.base import ChainMapper
    from mvpa2.mappers.svd import SVDMapper
    from mvpa2.mappers.fx import mean_group_sample
    from mvpa2.clfs.svm import LinearCSVMC
    from mvpa2.clfs.meta import MappedClassifier
    from mvpa2.measures.base import CrossValidation

    mapper = ChainMapper([mean_group_sample(['targets','chunks']),
                          SVDMapper()])
    clf = MappedClassifier(LinearCSVMC(), mapper)
    cvte = CrossValidation(clf, NFoldPartitioner(),
                           enable_ca=['repetition_results', 'stats'])

    ds = dataset_wizard(
        samples=np.arange(32).reshape((8, -1)),
        targets=[1, 1, 2, 2, 1, 1, 2, 2],
        chunks=[1, 1, 1, 1, 2, 2, 2, 2])

    errors = cvte(ds)


@reseed_rng()
def test_sifter_superord_usecase():
    from mvpa2.misc.data_generators import normal_feature_dataset
    from mvpa2.clfs.svm import LinearCSVMC            # fast one to use for tests
    from mvpa2.measures.base import CrossValidation

    from mvpa2.base.node import ChainNode
    from mvpa2.generators.partition import NFoldPartitioner
    from mvpa2.generators.base import  Sifter

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

    # and then do your normal where clf is space='superord'
    clf = LinearCSVMC(space='superord')
    cvte_regular = CrossValidation(clf, NFoldPartitioner(),
                                   errorfx=lambda p,t: np.mean(p==t))
    cvte_super = CrossValidation(clf, npart, errorfx=lambda p,t: np.mean(p==t))

    accs_regular = cvte_regular(ds)
    accs_super = cvte_super(ds)

    # With sifting we should get only 2^3 = 8 splits
    assert(len(accs_super) == 8)
    # I don't think that this would ever fail, so not marking it labile
    assert(np.mean(accs_regular) > .8)
    assert(np.mean(accs_super)   < .6)

def _test_edmund_chong_20120907():
    # commented out to avoid syntax warnings while compiling
    # from mvpa2.suite import *
    from mvpa2.testing.datasets import datasets
    repeater = Repeater(count=20)

    partitioner = ChainNode([NFoldPartitioner(cvtype=1),
                             Balancer(attr='targets',
                                      count=1, # for real data > 1
                                      limit='partitions',
                                      apply_selection=True
                                      )],
                            space='partitions')

    clf = LinearCSVMC() #choice of classifier
    permutator = AttributePermutator('targets', limit={'partitions': 1},
                                     count=1)
    null_cv = CrossValidation(
        clf,
        ChainNode([partitioner, permutator], space=partitioner.get_space()),
        errorfx=mean_mismatch_error)
    distr_est = MCNullDist(repeater, tail='left', measure=null_cv,
                           enable_ca=['dist_samples'])
    cvte = CrossValidation(clf, partitioner,
                           errorfx=mean_mismatch_error,
                           null_dist=distr_est,
                           enable_ca=['stats'])
    errors = cvte(datasets['uni2small'])


def test_chained_crossvalidation_searchlight():
    from mvpa2.clfs.gnb import GNB
    from mvpa2.clfs.meta import MappedClassifier
    from mvpa2.generators.partition import NFoldPartitioner
    from mvpa2.mappers.base import ChainMapper
    from mvpa2.mappers.base import Mapper
    from mvpa2.measures.base import CrossValidation
    from mvpa2.measures.searchlight import sphere_searchlight
    from mvpa2.testing.datasets import datasets

    dataset = datasets['3dlarge'].copy()
    dataset.fa['voxel_indices'] = dataset.fa.myspace
    sample_clf = GNB()              # fast and deterministic

    class ZScoreFeaturesMapper(Mapper):
        """Very basic mapper which would take care about standardizing
        all features within each sample separately
        """
        def _forward_data(self, data):
            return (data - np.mean(data, axis=1)[:, None])/np.std(data, axis=1)[:, None]

    # only do partial to save time
    sl_kwargs = dict(radius=2, center_ids=[3, 50])
    clf_mapped = MappedClassifier(sample_clf, ZScoreFeaturesMapper())
    cv = CrossValidation(clf_mapped, NFoldPartitioner())
    sl = sphere_searchlight(cv, **sl_kwargs)
    results_mapped = sl(dataset)

    cv_chained = ChainMapper([ZScoreFeaturesMapper(auto_train=True),
                              CrossValidation(sample_clf, NFoldPartitioner())])
    sl_chained = sphere_searchlight(cv_chained, **sl_kwargs)
    results_chained = sl_chained(dataset)

    assert_array_equal(results_mapped, results_chained)

def test_gnbsearchlight_permutations():
    import mvpa2
    from mvpa2.base.node import ChainNode
    from mvpa2.clfs.gnb import GNB
    from mvpa2.generators.base import  Repeater
    from mvpa2.generators.partition import NFoldPartitioner, OddEvenPartitioner
    #import mvpa2.generators.permutation
    #reload(mvpa2.generators.permutation)
    from mvpa2.generators.permutation import AttributePermutator
    from mvpa2.testing.datasets import datasets
    from mvpa2.measures.base import CrossValidation
    from mvpa2.measures.gnbsearchlight import sphere_gnbsearchlight
    from mvpa2.measures.searchlight import sphere_searchlight
    from mvpa2.mappers.fx import mean_sample
    from mvpa2.misc.errorfx import mean_mismatch_error
    from mvpa2.clfs.stats import MCNullDist
    from mvpa2.testing.tools import assert_raises, ok_

    # mvpa2.debug.active = ['APERM', 'SLC'] #, 'REPM']
    # mvpa2.debug.metrics += ['pid']
    count = 10
    nproc = 1 + int(mvpa2.externals.exists('pprocess'))
    ds = datasets['3dsmall'].copy()
    ds.fa['voxel_indices'] = ds.fa.myspace

    slkwargs = dict(radius=3, space='voxel_indices',  enable_ca=['roi_sizes'],
                    center_ids=[1, 10, 70, 100])

    mvpa2.seed(mvpa2._random_seed)
    clf  = GNB()
    splt = NFoldPartitioner(cvtype=2, attr='chunks')

    repeater   = Repeater(count=count)
    permutator = AttributePermutator('targets', limit={'partitions': 1}, count=1)

    null_sl = sphere_gnbsearchlight(clf, ChainNode([splt, permutator], space=splt.get_space()),
                                    postproc=mean_sample(), errorfx=mean_mismatch_error,
                                    **slkwargs)

    distr_est = MCNullDist(repeater, tail='left', measure=null_sl,
                           enable_ca=['dist_samples'])
    sl = sphere_gnbsearchlight(clf, splt,
                               reuse_neighbors=True,
                               null_dist=distr_est, postproc=mean_sample(),
                               errorfx=mean_mismatch_error,
                               **slkwargs)
    assert_raises(NotImplementedError, sl, ds)
    # "ad-hoc searchlights can't handle yet varying targets across partitions"
    if False:
        # after above limitation is removed -- enable
        sl_map = sl(ds)
        sl_null_prob = sl.ca.null_prob.samples.copy()

    mvpa2.seed(mvpa2._random_seed)
    ### 'normal' Searchlight
    clf  = GNB()
    splt = NFoldPartitioner(cvtype=2, attr='chunks')
    repeater   = Repeater(count=count)
    permutator = AttributePermutator('targets', limit={'partitions': 1}, count=1)
    # rng=np.random) # to trigger failure since the same np.random state
    # would be reused across all pprocesses
    null_cv = CrossValidation(clf, ChainNode([splt, permutator], space=splt.get_space()),
                              postproc=mean_sample())
    null_sl_normal = sphere_searchlight(null_cv, nproc=nproc, **slkwargs)
    distr_est_normal = MCNullDist(repeater, tail='left', measure=null_sl_normal,
                           enable_ca=['dist_samples'])

    cv = CrossValidation(clf, splt, errorfx=mean_mismatch_error,
                         enable_ca=['stats'], postproc=mean_sample() )
    sl = sphere_searchlight(cv, nproc=nproc, null_dist=distr_est_normal, **slkwargs)
    sl_map_normal = sl(ds)
    sl_null_prob_normal = sl.ca.null_prob.samples.copy()

    # For every feature -- we should get some variance in estimates In
    # case of failure they are all really close to each other (up to
    # numerical precision), so variance will be close to 0
    assert_array_less(-np.var(distr_est_normal.ca.dist_samples.samples[0],
                              axis=1), -1e-5)
    for s in distr_est_normal.ca.dist_samples.samples[0]:
        ok_(len(np.unique(s)) > 1)

    # TODO: compare two results, although might become tricky with
    #       nproc=2 and absent way to control RNG across child processes
