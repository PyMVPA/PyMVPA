### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Unit tests for PyMVPA pattern handling
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    version 2 of the License, or (at your option) any later version.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import mvpa
import mvpa.knn as knn
import mvpa.svm as svm
import mvpa.stats
import unittest
import numpy
import scipy.stats as stat
import sys

def pureMultivariateSignal(patterns, origin, signal2noise = 1.5):
    """ Create a 2d dataset with a clear multivariate signal, but no
    univariate information.

    %%%%%%%%%
    % O % X %
    %%%%%%%%%
    % X % O %
    %%%%%%%%%
    """

    # start with noise
    data=numpy.random.normal(size=(4*patterns,2))

    # add signal
    data[:2*patterns,1] += signal2noise
    data[2*patterns:4*patterns,1] -= signal2noise
    data[:patterns,0] -= signal2noise
    data[2*patterns:3*patterns,0] -= signal2noise
    data[patterns:2+patterns,0] += signal2noise
    data[3*patterns:4*patterns,0] += signal2noise

    # two conditions
    regs = [0 for i in xrange(patterns)] \
        + [1 for i in xrange(patterns)] \
        + [1 for i in xrange(patterns)] \
        + [0 for i in xrange(patterns)]
    regs = numpy.array(regs)

    return mvpa.MVPAPattern(data, regs, origin)


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

        self.failUnless( data.npatterns == 120 )
        self.failUnless( data.nfeatures == 2 )
        self.failUnless(
            (data.reg == [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]*6 ).all() )
        self.failUnless(
            (data.origin == \
                [ k for k in range(1,7) for i in range(20) ] ).all() )


        cv = mvpa.CrossValidation( data, knn.kNN(), cvtype=1)
        perf = numpy.array(cv())
        self.failUnless( perf.mean() > 0.8 and perf.mean() <= 1.0 )
        self.failUnless( len( perf ) == 6 )

        # check the number of CV folds
        self.failUnless( cv.xvalpattern.getNCVFolds(1) == 6 )

        # check the number of available patterns per CV fold
        ntrainpats, ntestpats = cv.xvalpattern.getNPatternsPerCVFold(1)

        self.failUnless( ntrainpats.shape == (6,2) )
        self.failUnless( ntestpats.shape == (6,2) )
        self.failUnless( (ntrainpats ==\
                         numpy.array([ 50 for i in range(12) ]).reshape(6,2)
                         ).all() )
        self.failUnless( (ntestpats ==\
                         numpy.array([ 10 for i in range(12) ]).reshape(6,2)
                         ).all() )

    def testRegressorPermutation(self):
        data = self.getMVPattern(4)

        cv = mvpa.CrossValidation( data, knn.kNN(), cvtype=1)
        perf = numpy.array(cv())

        data.permutatedRegressors( True )

        perm_perf = numpy.array(cv())

        self.failUnless( perf.mean() > perm_perf.mean())


    def testCLFStatskNN(self):
        data_h = self.getMVPattern(1)
        data_l = self.getMVPattern(0.5)

        data_h_uv = data_h.selectFeatures([0])

        # it is difficult to find criteria for a correct CV
        # when using random input data
        # for now, CV level 1,2 and 3 are simply run w/o special tests
        clf = knn.kNN()
        for cv in (1,2,3):
            cv_h = mvpa.CrossValidation( data_h, clf, cvtype=cv )
            perf_h = numpy.array( cv_h() )

            cv_l = mvpa.CrossValidation( data_l, clf, cvtype=cv )
            perf_l = numpy.array( cv_l() )

            cv_h_uv = mvpa.CrossValidation( data_h_uv, clf, cvtype=cv )
            perf_h_uv = numpy.array( cv_h_uv() )

            #print perf_h.mean(), stat.ttest_1samp( perf_h, 0.5 )[1] < 0.05
            #print perf_l.mean(), stat.ttest_1samp( perf_l, 0.5 )[1] < 0.05
            #print perf_h_uv.mean(), stat.ttest_1samp( perf_h_uv, 0.5 )[1] <0.05


    def testCLFStatsSVM(self):
        data_h = self.getMVPattern(1)
        data_l = self.getMVPattern(0.5)

        data_h_uv = data_h.selectFeatures([0])

        # it is difficult to find criteria for a correct CV
        # when using random input data
        # for now, CV level 1,2 and 3 are simply run w/o special tests
        clf = svm.SVM()
        for cv in (1,2,3):
            cv_h = mvpa.CrossValidation( data_h, clf, cvtype=cv )
            perf_h = numpy.array( cv_h() )

            cv_l = mvpa.CrossValidation( data_l, clf, cvtype=cv )
            perf_l = numpy.array( cv_l() )

            cv_h_uv = mvpa.CrossValidation( data_h_uv, clf, cvtype=cv )
            perf_h_uv = numpy.array( cv_h_uv() )

            #print perf_h.mean(), stat.ttest_1samp( perf_h, 0.5 )[1]
            #print perf_l.mean(), stat.ttest_1samp( perf_l, 0.5 )[1]
            #print perf_h_uv.mean(), stat.ttest_1samp( perf_h_uv, 0.5 )[1]

            #print cv_h.contingencytbl, \
            #      mvpa.stats.chisquare(cv_h.contingencytbl)
            #print cv_l.contingencytbl, \
            #      mvpa.stats.chisquare(cv_l.contingencytbl)
            #print cv_h_uv.contingencytbl, \
            #      mvpa.stats.chisquare(cv_h_uv.contingencytbl)


    def testPatternSamples(self):
        data = self.getMVPattern(3)

        cv = mvpa.CrossValidation( data, knn.kNN(), cvtype=1, ncvfoldsamples = 2 )

        perf = numpy.array(cv())

        # check that each cvfold is done twice
        self.failUnless( len( perf ) == 12 )

        # check that all training and test patterns are used for sampling
        self.failUnless( cv.testsamplelog == [ None for i in range(12) ] )
        self.failUnless( cv.trainsamplelog == [ None for i in range(12) ] )

        # check total pattern number per reg
        self.failUnless( cv.xvalpattern.pattern.patperreg == [60,60] )

        # enable automatic training pattern sampling
        cv.xvalpattern.trainsamplesize = 'auto'

        cv()
        # check that still all patterns are used for training and test patterns
        # are not touched at all
        self.failUnless( cv.trainsamplelog == [ 50 for i in range(12) ] )
        self.failUnless( cv.testsamplelog == [ None for i in range(12) ] )

        # now do pattern and training sampling
        cv.xvalpattern.trainsamplesize = 28
        cv.xvalpattern.testsamplesize = 6

        cv()

        self.failUnless( cv.trainsamplelog == [ 28 for i in range(12) ] )
        self.failUnless( cv.testsamplelog == [ 6 for i in range(12) ] )

        # now check that you cannot get more pattern samples than are available
        cv.xvalpattern.testsamplesize = 11

        cv.setClassifier( knn.kNN() )
        self.failUnlessRaises( ValueError, cv )


    def testNoiseClassification(self):
        # get a dataset with a very high SNR
        data = self.getMVPattern(10)

        # do crossval
        clf = knn.kNN()
        perf = mvpa.CrossValidation( data, clf, cvtype=1,
                                      ncvfoldsamples=10 )()
        # must be perfect
        self.failUnless( numpy.array(perf).mean() == 1.0 )

        # do crossval with permutated regressors
        perf = mvpa.CrossValidation(
                    data, clf, cvtype=1,
                    ncvfoldsamples= 10 )(permutate = True)

        # must be at chance level
        pmean = numpy.array(perf).mean()
        self.failUnless( pmean < 0.58 and pmean > 0.42 )


def suite():
    return unittest.makeSuite(CrossValidationTests)


if __name__ == '__main__':
    unittest.main()

