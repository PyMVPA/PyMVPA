### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
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
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

import mvpa
import unittest
import numpy

class PatternTests(unittest.TestCase):

    def testSelectFeatures(self):
        # make random 4d data set
        data = mvpa.Patterns()
        data.addPatterns( numpy.random.standard_normal( (10,2,3,4) ), 1, 1 )
        # make full 3d mask
        mask = numpy.ones( (2,3,4) )
        # check 4d -> 2d with 3d mask
        selected = data.selectFeatures( mask )
        self.failUnlessEqual( selected.shape, (10, 24) )

        # now reduce mask
        mask[1,1,1] = 0
        # check the one feature is removed
        selected = data.selectFeatures( mask )
        self.failUnlessEqual( selected.shape, (10, 23) )

        # make random 2d data set
        data.clear()
        data.addPatterns( numpy.random.standard_normal( (10,5) ), 1, 1)
        # make full 1d mask
        mask = numpy.ones( (5) )
        # check 2d -> 2d with 1d mask
        selected = data.selectFeatures( mask )
        self.failUnlessEqual( selected.shape, (10, 5) )

        # make random 1d data set
        data.clear()
        data.addPatterns( numpy.random.standard_normal( 10), 1, 1 )
        # check 1d -> 2d
        selected = data.selectFeatures()
        self.failUnlessEqual( selected.shape, (10, 1) )


    def testZScoring(self):
        data = mvpa.Patterns()
        # dataset: mean=2, std=1
        pat = numpy.array( (0,1,3,4,2,2,3,1,1,3,3,1,2,2,2,2) )
        self.failUnlessEqual( pat.mean(), 2.0 )
        self.failUnlessEqual( pat.std(), 1.0 )
        data.addPatterns( pat, 1, 1)
        data.zscore()

        # check z-scoring
        self.failUnless( data.pattern == [-2,-1,1,2,0,0,1,-1,-1,1,1,-1,0,0,0,0] )


    def testPatternShape(self):
        data = mvpa.Patterns()
        data.addPatterns( numpy.ones((10,2,3,4)), 1, 1 )

        self.failUnless( data.getPatternShape() == (2,3,4) )

def suite():
    return unittest.makeSuite(PatternTests)


if __name__ == '__main__':
    unittest.main()

