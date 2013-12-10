# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA classifier cross-validation"""

from mvpa2.testing.tools import assert_equal, ok_, assert_array_equal

from mvpa2.base.node import ChainNode
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.generators.permutation import AttributePermutator
from mvpa2.measures.base import CrossValidation

from mvpa2.testing import *
from mvpa2.testing.datasets import pure_multivariate_signal, get_mv_pattern
from mvpa2.testing.clfs import *

class CrossValidationTests(unittest.TestCase):


    def test_simple_n_minus_one_cv(self):
        data = get_mv_pattern(3)
        data.init_origids('samples')

        self.assertTrue( data.nsamples == 120 )
        self.assertTrue( data.nfeatures == 2 )
        self.assertTrue(
            (data.sa.targets == \
                [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0] * 6).all())
        self.assertTrue(
            (data.sa.chunks == \
                [k for k in range(1, 7) for i in range(20)]).all())
        assert_equal(len(np.unique(data.sa.origids)), data.nsamples)

        cv = CrossValidation(sample_clf_nl, NFoldPartitioner(),
                enable_ca=['stats', 'training_stats'])
#                               'samples_error'])

        results = cv(data)
        self.assertTrue((results.samples < 0.2).all() and (results.samples >= 0.0).all())

        # TODO: test accessibility of {training_,}stats{,s} of
        # CrossValidatedTransferError

        # not yet implemented, and no longer this way
        #self.assertTrue(isinstance(cv.ca.samples_error, dict))
        #self.assertTrue(len(cv.ca.samples_error) == data.nsamples)
        ## one value for each origid
        #assert_array_equal(sorted(cv.ca.samples_error.keys()),
        #                   sorted(data.sa.origids))
        #for k, v in cv.ca.samples_error.iteritems():
        #    self.assertTrue(len(v) == 1)


    def test_noise_classification(self):
        # get a dataset with a very high SNR
        data = get_mv_pattern(10)

        # do crossval with default errorfx and 'mean' combiner
        cv = CrossValidation(sample_clf_nl, NFoldPartitioner())

        # must return a scalar value
        result = cv(data)
        # must be perfect
        self.assertTrue((result.samples < 0.05).all())

        # do crossval with permuted regressors
        cv = CrossValidation(sample_clf_nl,
                        ChainNode([NFoldPartitioner(),
                            AttributePermutator('targets', count=10)],
                                  space='partitions'))
        results = cv(data)

        # results must not be the same
        self.assertTrue(len(np.unique(results.samples))>1)

        # must be at chance level
        pmean = np.array(results).mean()
        self.assertTrue( pmean < 0.58 and pmean > 0.42 )


    def test_unpartitioned_cv(self):
        data = get_mv_pattern(10)
        # only one big chunk
        data.sa.chunks[:] = 1
        cv = CrossValidation(sample_clf_nl, NFoldPartitioner())
        # need to fail, because it can't be split into training and testing
        assert_raises(ValueError, cv, data)


def suite():  # pragma: no cover
    return unittest.makeSuite(CrossValidationTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

