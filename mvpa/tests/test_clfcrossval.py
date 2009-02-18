# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA classifier cross-validation"""

from mvpa.datasets.splitters import NFoldSplitter
from mvpa.datasets.meta import MetaDataset
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.clfs.transerror import TransferError

from tests_warehouse import *
from tests_warehouse import pureMultivariateSignal, getMVPattern
from tests_warehouse_clfs import *

class CrossValidationTests(unittest.TestCase):


    def testSimpleNMinusOneCV(self):
        data = getMVPattern(3)

        self.failUnless( data.nsamples == 120 )
        self.failUnless( data.nfeatures == 2 )
        self.failUnless(
            (data.labels == \
                [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0] * 6).all())
        self.failUnless(
            (data.chunks == \
                [k for k in range(1, 7) for i in range(20)]).all())

        transerror = TransferError(sample_clf_nl)
        cv = CrossValidatedTransferError(
                transerror,
                NFoldSplitter(cvtype=1),
                enable_states=['confusion', 'training_confusion',
                               'samples_error'])

        results = cv(data)
        self.failUnless( results < 0.2 and results >= 0.0 )

        # TODO: test accessibility of {training_,}confusion{,s} of
        # CrossValidatedTransferError

        self.failUnless(isinstance(cv.samples_error, dict))
        self.failUnless(len(cv.samples_error) == data.nsamples)
        # one value for each origid
        self.failUnless(sorted(cv.samples_error.keys()) == sorted(data.origids))
        for k, v in cv.samples_error.iteritems():
            self.failUnless(len(v) == 1)


    def testNoiseClassification(self):
        # get a dataset with a very high SNR
        data = getMVPattern(10)

        # do crossval with default errorfx and 'mean' combiner
        transerror = TransferError(sample_clf_nl)
        cv = CrossValidatedTransferError(transerror, NFoldSplitter(cvtype=1)) 

        # must return a scalar value
        result = cv(data)

        # must be perfect
        self.failUnless( result < 0.05 )

        # do crossval with permuted regressors
        cv = CrossValidatedTransferError(transerror,
                  NFoldSplitter(cvtype=1, permute=True, nrunspersplit=10) )
        results = cv(data)

        # must be at chance level
        pmean = N.array(results).mean()
        self.failUnless( pmean < 0.58 and pmean > 0.42 )


    def testHarvesting(self):
        # get a dataset with a very high SNR
        data = getMVPattern(10)

        # do crossval with default errorfx and 'mean' combiner
        transerror = TransferError(clfswh['linear'][0])
        cv = CrossValidatedTransferError(
                transerror,
                NFoldSplitter(cvtype=1),
                harvest_attribs=['transerror.clf.training_time'])
        result = cv(data)
        self.failUnless(cv.harvested.has_key('transerror.clf.training_time'))
        self.failUnless(len(cv.harvested['transerror.clf.training_time'])>1)


    def testNMinusOneCVWithMetaDataset(self):
        # simple datasets with decreasing SNR
        data = MetaDataset([getMVPattern(3), getMVPattern(2), getMVPattern(1)])

        self.failUnless( data.nsamples == 120 )
        self.failUnless( data.nfeatures == 6 )
        self.failUnless(
            (data.labels == \
                [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0] * 6).all())
        self.failUnless(
            (data.chunks == \
                [ k for k in range(1,7) for i in range(20) ] ).all() )

        transerror = TransferError(sample_clf_nl)
        cv = CrossValidatedTransferError(transerror,
                                         NFoldSplitter(cvtype=1),
                                         enable_states=['confusion',
                                                        'training_confusion'])

        results = cv(data)
        self.failUnless(results < 0.2 and results >= 0.0,
                        msg="We should generalize while working with "
                        "metadataset. Got %s error" % results)

        # TODO: test accessibility of {training_,}confusion{,s} of
        # CrossValidatedTransferError



def suite():
    return unittest.makeSuite(CrossValidationTests)


if __name__ == '__main__':
    import runner

