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
    from mvpa2.testing.tools import assert_raises, ok_, assert_array_less

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
    if __debug__:                         # assert is done only without -O mode
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
    # rng=np.random.RandomState(0)) # to trigger failure since the same np.random state
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

def test_multiclass_pairs_svm_searchlight():
    from mvpa2.measures.searchlight import sphere_searchlight
    import mvpa2.clfs.meta
    #reload(mvpa2.clfs.meta)
    from mvpa2.clfs.meta import MulticlassClassifier

    from mvpa2.datasets import Dataset
    from mvpa2.clfs.svm import LinearCSVMC
    #import mvpa2.testing.datasets
    #reload(mvpa2.testing.datasets)
    from mvpa2.testing.datasets import datasets
    from mvpa2.generators.partition import NFoldPartitioner, OddEvenPartitioner
    from mvpa2.measures.base import CrossValidation

    from mvpa2.testing import ok_, assert_equal, assert_array_equal
    from mvpa2.sandbox.multiclass import get_pairwise_accuracies

    # Some parameters used in the test below
    nproc = 1 + int(mvpa2.externals.exists('pprocess'))
    ntargets = 4                                # number of targets
    npairs = ntargets*(ntargets-1)/2
    center_ids = [35, 55, 1]
    ds = datasets['3dsmall'].copy()

    # redefine C,T so we have a multiclass task
    nsamples = len(ds)
    ds.sa.targets = range(ntargets) * (nsamples//ntargets)
    ds.sa.chunks = np.arange(nsamples) // ntargets
    # and add some obvious signal where it is due
    ds.samples[:, 55] += 10*ds.sa.targets   # for all 4 targets
    ds.samples[:, 35] += 10*(ds.sa.targets % 2) # so we have conflicting labels
    # while 35 would still be just for 2 categories which would conflict

    mclf = MulticlassClassifier(LinearCSVMC(),
                                pass_attr=['sa.chunks', 'ca.raw_predictions_ds'],
                                enable_ca=['raw_predictions_ds'])

    label_pairs = mclf._get_binary_pairs(ds)

    def place_sa_as_samples(ds):
        # add a degenerate dimension for the hstacking in the searchlight
        ds.samples = ds.sa.raw_predictions_ds[:, None]
        ds.sa.pop('raw_predictions_ds')   # no need to drag the copy
        return ds

    mcv = CrossValidation(mclf, OddEvenPartitioner(), errorfx=None,
                          postproc=place_sa_as_samples)
    sl = sphere_searchlight(mcv, nproc=nproc, radius=2, space='myspace',
                            center_ids=center_ids)
    slmap = sl(ds)


    ok_('chunks' in slmap.sa)
    ok_('cvfolds' in slmap.sa)
    ok_('targets' in slmap.sa)
    # so for each SL we got all pairwise tests
    assert_equal(slmap.shape, (nsamples, len(center_ids), npairs))
    assert_array_equal(np.unique(slmap.sa.cvfolds), [0, 1])

    # Verify that we got right labels in each 'pair'
    # all searchlights should have the same set of labels for a given
    # pair of targets
    label_pairs_ = np.apply_along_axis(
        np.unique, 0,
        ## reshape slmap so we have only simple pairs in the columns
        np.reshape(slmap, (-1, npairs))).T

    # need to prep that list of pairs obtained from MulticlassClassifier
    # and since it is 1-vs-1, they all should be just pairs of lists of
    # 1 element so should work
    assert_equal(len(label_pairs_), npairs)
    assert_array_equal(np.squeeze(np.array(label_pairs)), label_pairs_)
    assert_equal(label_pairs_.shape, (npairs, 2))   # for this particular case


    out    = get_pairwise_accuracies(slmap)
    out123 = get_pairwise_accuracies(slmap, select=[1, 2, 3])

    assert_array_equal(np.unique(out123.T), np.arange(1, 4))   # so we got at least correct targets
    # test that we extracted correct accuracies
    # First 3 in out.T should have category 0, so skip them and compare otherwise
    assert_array_equal(out.samples[3:], out123.samples)

    ok_(np.all(out.samples[:, 1] == 1.), "This was with super-strong result")
