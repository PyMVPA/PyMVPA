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


    def testMofNCombinations(self):
        self.failUnlessEqual( 
            mvpa.getUniqueLengthNCombinations( range(3), 1 ), [[0],[1],[2]] )
        self.failUnlessEqual( 
            mvpa.getUniqueLengthNCombinations( 
                        range(4), 2 ), 
                        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]] 
                        )
        self.failUnlessEqual( 
            mvpa.getUniqueLengthNCombinations( 
                        range(4), 3 ), [[0, 1, 2], [0, 1, 3], [0, 2, 3]] )


    def testSimpleNMinusOneCV(self):
        data = self.getMVPattern(3)

        self.failUnless( data.npatterns == 120 )
        self.failUnless( data.nfeatures == 2 )
        self.failUnless(
            (data.reg == [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]*6 ).all() )
        self.failUnless(
            (data.origin == \
                [ k for k in range(1,7) for i in range(20) ] ).all() )


        cv = mvpa.CrossValidation( data, mvpa.kNN )
        perf = numpy.array(cv.run(cvtype=1))
        self.failUnless( perf.mean() > 0.8 and perf.mean() <= 1.0 )
        self.failUnless( len( perf ) == 6 )


    def testCLFStatskNN(self):
        data_h = self.getMVPattern(1)
        data_l = self.getMVPattern(0.5)

        data_h_uv = data_h.selectFeatures([0])

        # it is difficult to find criteria for a correct CV
        # when using random input data
        # for now, CV level 1,2 and 3 are simply run w/o special tests
        for cv in (1,2,3):
            cv_h = mvpa.CrossValidation( data_h, mvpa.kNN )
            perf_h = numpy.array( cv_h.run( cvtype=cv ) )

            cv_l = mvpa.CrossValidation( data_l, mvpa.kNN )
            perf_l = numpy.array( cv_l.run( cvtype=cv ) )

            cv_h_uv = mvpa.CrossValidation( data_h_uv, mvpa.kNN )
            perf_h_uv = numpy.array( cv_h_uv.run( cvtype=cv ) )

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
        for cv in (1,2,3):
            cv_h = mvpa.CrossValidation( data_h, mvpa.SVM )
            perf_h = numpy.array( cv_h.run( cvtype=cv ) )

            cv_l = mvpa.CrossValidation( data_l, mvpa.SVM )
            perf_l = numpy.array( cv_l.run( cvtype=cv ) )

            cv_h_uv = mvpa.CrossValidation( data_h_uv, mvpa.SVM )
            perf_h_uv = numpy.array( cv_h_uv.run( cvtype=cv ) )

            print perf_h.mean(), stat.ttest_1samp( perf_h, 0.5 )[1] < 0.05
            print perf_l.mean(), stat.ttest_1samp( perf_l, 0.5 )[1] < 0.05
            print perf_h_uv.mean(), stat.ttest_1samp( perf_h_uv, 0.5 )[1] <0.05


    def testPatternSampling(self):
        data = self.getMVPattern(3)

        cv = mvpa.CrossValidation( data, mvpa.kNN )
        cv.ncvfoldsamples = 2

        perf = numpy.array(cv.run(cvtype=1))

        # check that each cvfold is done twice
        self.failUnless( len( perf ) == 12 )

        # check that all training and test patterns are used for sampling
        self.failUnless( cv.testsamplelog == [ None for i in range(12) ] )
        self.failUnless( cv.trainsamplelog == [ None for i in range(12) ] )

        # check total pattern number per reg
        self.failUnless( cv.pattern.patperreg == [60,60] )

        # enable automatic training pattern sampling
        cv.trainsamplecfg = 'auto'

        cv.run(cvtype=1)
        # check that still all patterns are used for training and test patterns
        # are not touched at all
        self.failUnless( cv.trainsamplelog == [ 50 for i in range(12) ] )
        self.failUnless( cv.testsamplelog == [ None for i in range(12) ] )

        # now do pattern and training sampling
        cv.trainsamplecfg = 28
        cv.testsamplecfg = 6

        cv.run(cvtype = 1)

        self.failUnless( cv.trainsamplelog == [ 28 for i in range(12) ] )
        self.failUnless( cv.testsamplelog == [ 6 for i in range(12) ] )

        # now check that you cannot get more pattern samples than are available
        cv.testsamplecfg = 11

        self.failUnlessRaises( ValueError, cv.run, cvtype = 1 )


    def testContingencyTbl(self):
        data = numpy.array([1,2,1,2,2,2,3,2,1], ndmin=2).T
        reg = numpy.array([1,1,1,2,2,2,3,3,3])

        cv = mvpa.CrossValidation( mvpa.MVPAPattern( data, reg ),
                                   mvpa.kNN )

        tbl = cv.makeContingencyTbl( cv.pattern.reg,
                                     numpy.array([1,2,1,2,2,2,3,2,1]) )

        # should be square matrix (len(reglabels) x len(reglabels)
        self.failUnless( tbl.shape == (3,3) )

        # check table content
        self.failUnless( (tbl == [[2,1,0],[0,3,0],[1,1,1]]).all() )


def suite():
    return unittest.makeSuite(CrossValidationTests)


if __name__ == '__main__':
    unittest.main()

