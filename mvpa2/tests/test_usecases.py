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
