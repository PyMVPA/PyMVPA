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
        run1 = pureMultivariateSignal(5, 1, 1.5)
        run2 = pureMultivariateSignal(5, 2, 1.5)
        run3 = pureMultivariateSignal(5, 3, 1.5)
        run4 = pureMultivariateSignal(5, 4, 1.5)
        run5 = pureMultivariateSignal(5, 5, 1.5)
        run6 = pureMultivariateSignal(5, 6, 1.5)

        data = run1 + run2 + run3 + run4 + run5 + run6

        self.failUnless( data.npatterns == 120 )
        self.failUnless( data.nfeatures == 2 )
        self.failUnless(
            (data.reg == [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]*6 ).all() )
        self.failUnless( (data.origin == [ k for k in range(1,7) for i in range(20) ] ).all() )


        cv = mvpa.CrossValidation( data, mvpa.kNN )
        perf = numpy.array(cv.start(cv=1))
        print perf, perf.mean()
        perf = numpy.array(cv.start(cv=2))
        print perf, perf.mean()


def suite():
    return unittest.makeSuite(CrossValidationTests)


if __name__ == '__main__':
    unittest.main()

