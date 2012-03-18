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

