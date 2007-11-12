#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA classifier cross-validation"""

import unittest
import numpy as N

from mvpa.datasets.dataset import Dataset
from mvpa.clf.knn import kNN
from mvpa.datasets.nfoldsplitter import NFoldSplitter
from mvpa.algorithms.clfcrossval import ClfCrossValidation
from mvpa.misc.errorfx import MeanMatchErrorFx


def pureMultivariateSignal(nsamples, chunk, signal2noise = 1.5):
    """ Create a 2d dataset with a clear multivariate signal, but no
    univariate information.

    %%%%%%%%%
    % O % X %
    %%%%%%%%%
    % X % O %
    %%%%%%%%%
    """

    # start with noise
    data=N.random.normal(size=(4*nsamples,2))

    # add signal
    data[:2*nsamples,1] += signal2noise
    data[2*nsamples:4*nsamples,1] -= signal2noise
    data[:nsamples,0] -= signal2noise
    data[2*nsamples:3*nsamples,0] -= signal2noise
    data[nsamples:2+nsamples,0] += signal2noise
    data[3*nsamples:4*nsamples,0] += signal2noise

    # two conditions
    labels = [0 for i in xrange(nsamples)] \
             + [1 for i in xrange(nsamples)] \
             + [1 for i in xrange(nsamples)] \
             + [0 for i in xrange(nsamples)]
    labels = N.array(labels)

    return Dataset(samples=data, labels=labels, chunks=chunk)


class CrossValidationTests(unittest.TestCase):

    def getMVPattern(self, s2n):
        run1 = pureMultivariateSignal(5, 1, s2n)
        run2 = pureMultivariateSignal(5, 2, s2n)
        run3 = pureMultivariateSignal(5, 3, s2n)
        run4 = pureMultivariateSignal(5, 4, s2n)
        run5 = pureMultivariateSignal(5, 5, s2n)
        run6 = pureMultivariateSignal(5, 6, s2n)

        data = run1 + run2 + run3 + run4 + run5 + run6

        return data


    def testSimpleNMinusOneCV(self):
        data = self.getMVPattern(3)

        self.failUnless( data.nsamples == 120 )
        self.failUnless( data.nfeatures == 2 )
        self.failUnless(
            (data.labels == [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]*6 ).all() )
        self.failUnless(
            (data.chunks == \
                [ k for k in range(1,7) for i in range(20) ] ).all() )


        cv = ClfCrossValidation(
                kNN(),
                NFoldSplitter(cvtype=1))

        results = cv(data)
        self.failUnless( results > 0.8 and results <= 1.0 )


    def testNoiseClassification(self):
        # get a dataset with a very high SNR
        data = self.getMVPattern(10)

        # do crossval with default errorfx and 'mean' combiner
        cv = ClfCrossValidation(kNN(), NFoldSplitter(cvtype=1)) 

        # must return a scalar value
        result = cv(data)

        # must be perfect
        self.failUnless( result > 0.95 )

        # do crossval with permuted regressors
        cv = ClfCrossValidation(
                  kNN(),
                  NFoldSplitter(cvtype=1, permute=True, nrunsperfold=10) )
        results = cv(data)

        # must be at chance level
        pmean = N.array(results).mean()
        self.failUnless( pmean < 0.58 and pmean > 0.42 )


def suite():
    return unittest.makeSuite(CrossValidationTests)


if __name__ == '__main__':
    import test_runner

