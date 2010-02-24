# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA classifier cross-validation"""

from mvpa.testing.tools import assert_equal, ok_, assert_array_equal

from mvpa.datasets.splitters import NFoldSplitter
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.clfs.transerror import TransferError

from mvpa.testing import *
from mvpa.testing.datasets import pure_multivariate_signal, get_mv_pattern
from mvpa.testing.clfs import *

class CrossValidationTests(unittest.TestCase):


    def test_simple_n_minus_one_cv(self):
        data = get_mv_pattern(3)
        data.init_origids('samples')

        self.failUnless( data.nsamples == 120 )
        self.failUnless( data.nfeatures == 2 )
        self.failUnless(
            (data.sa.targets == \
                [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0] * 6).all())
        self.failUnless(
            (data.sa.chunks == \
                [k for k in range(1, 7) for i in range(20)]).all())
        assert_equal(len(np.unique(data.sa.origids)), data.nsamples)

        transerror = TransferError(sample_clf_nl)
        cv = CrossValidatedTransferError(
                transerror,
                NFoldSplitter(cvtype=1),
                enable_ca=['confusion', 'training_confusion',
                               'samples_error'])

        results = cv(data)
        self.failUnless((results.samples < 0.2).all() and (results.samples >= 0.0).all())

        # TODO: test accessibility of {training_,}confusion{,s} of
        # CrossValidatedTransferError

        self.failUnless(isinstance(cv.ca.samples_error, dict))
        self.failUnless(len(cv.ca.samples_error) == data.nsamples)
        # one value for each origid
        assert_array_equal(sorted(cv.ca.samples_error.keys()),
                           sorted(data.sa.origids))
        for k, v in cv.ca.samples_error.iteritems():
            self.failUnless(len(v) == 1)


    def test_noise_classification(self):
        # get a dataset with a very high SNR
        data = get_mv_pattern(10)

        # do crossval with default errorfx and 'mean' combiner
        transerror = TransferError(sample_clf_nl)
        cv = CrossValidatedTransferError(transerror, NFoldSplitter(cvtype=1)) 

        # must return a scalar value
        result = cv(data)
        # must be perfect
        self.failUnless((result.samples < 0.05).all())

        # do crossval with permuted regressors
        cv = CrossValidatedTransferError(transerror,
                  NFoldSplitter(cvtype=1, permute=True, nrunspersplit=10) )
        results = cv(data)

        # must be at chance level
        pmean = np.array(results).mean()
        self.failUnless( pmean < 0.58 and pmean > 0.42 )


    def test_harvesting(self):
        # get a dataset with a very high SNR
        data = get_mv_pattern(10)
        # do crossval with default errorfx and 'mean' combiner
        transerror = TransferError(clfswh['linear'][0])
        cv = CrossValidatedTransferError(
                transerror,
                NFoldSplitter(cvtype=1),
                harvest_attribs=['transerror.clf.ca.training_time'])
        result = cv(data)
        ok_(cv.ca.harvested.has_key('transerror.clf.ca.training_time'))
        assert_equal(len(cv.ca.harvested['transerror.clf.ca.training_time']),
                     len(data.UC))



def suite():
    return unittest.makeSuite(CrossValidationTests)


if __name__ == '__main__':
    import runner

