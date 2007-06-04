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
import random

class PatternTests(unittest.TestCase):

    def testAddPatterns(self):
        data = mvpa.MVPAPattern(range(5), 1, 1)
        # simple sequence has to be a single pattern
        self.failUnlessEqual( data.npatterns, 1)
        # check correct pattern layout (1x5)
        self.failUnless( 
            (data.pattern == numpy.array([[0, 1, 2, 3, 4]])).all() )

        # check for single regressor and origin
        self.failUnless( (data.reg == numpy.array([1])).all() )
        self.failUnless( (data.origin == numpy.array([1])).all() )

        # now try adding pattern with wrong shape
        self.failUnlessRaises( ValueError, 
                               data.addPattern, numpy.ones((2,3)), 1, 1 )

        # now add two real patterns
        data.addPattern( numpy.random.standard_normal((2,5)), 2, 2 )
        self.failUnlessEqual( data.npatterns, 3 )
        self.failUnless( (data.reg == numpy.array([1,2,2]) ).all() )
        self.failUnless( (data.origin == numpy.array([1,2,2]) ).all() )


        # test wrong regressor length
        self.failUnlessRaises( ValueError,
                               mvpa.MVPAPattern,
                               numpy.random.standard_normal((4,2,3,4)),
                               [ 1, 2, 3 ],
                               2 )

        # test wrong origin length
        self.failUnlessRaises( ValueError,
                               mvpa.MVPAPattern,
                               numpy.random.standard_normal((4,2,3,4)),
                               [ 1, 2, 3, 4 ],
                               [ 2, 2, 2 ] )

    def testShapeConversion(self):
        data = mvpa.MVPAPattern( numpy.arange(24).reshape((2,3,4)),1,1 )
        self.failUnlessEqual(data.npatterns, 2)
        self.failUnless( data.origshape == (3,4) )
        self.failUnlessEqual( data.pattern.shape, (2,12) )
        self.failUnless( (data.pattern == 
                          numpy.array([range(12),range(12,24)] ) ).all() )


#    def testSelectFeatures(self):
#        # make random 4d data set
#        data = mvpa.Patterns()
#        data.addPatterns( numpy.random.standard_normal( (10,2,3,4) ), 1, 1 )
#        # make full 3d mask
#        mask = numpy.ones( (2,3,4) )
#        # check 4d -> 2d with 3d mask
#        selected = data.selectFeatures( mask )
#        self.failUnlessEqual( selected.shape, (10, 24) )
#
#        # now reduce mask
#        mask[1,1,1] = 0
#        # check the one feature is removed
#        selected = data.selectFeatures( mask )
#        self.failUnlessEqual( selected.shape, (10, 23) )
#
#        # make random 2d data set
#        data.clear()
#        data.addPatterns( numpy.random.standard_normal( (10,5) ), 1, 1)
#        # make full 1d mask
#        mask = numpy.ones( (5) )
#        # check 2d -> 2d with 1d mask
#        selected = data.selectFeatures( mask )
#        self.failUnlessEqual( selected.shape, (10, 5) )
#
#        # make random 1d data set
#        data.clear()
#        data.addPatterns( numpy.random.standard_normal( 10), 1, 1 )
#        # check 1d -> 2d
#        selected = data.selectFeatures()
#        self.failUnlessEqual( selected.shape, (10, 1) )
#
#
    def testZScoring(self):
        # dataset: mean=2, std=1
        pat = numpy.array( (0,1,3,4,2,2,3,1,1,3,3,1,2,2,2,2) )
        data = mvpa.MVPAPattern(pat.reshape((16,1)), 1, 1)
        self.failUnlessEqual( data.pattern.mean(), 2.0 )
        self.failUnlessEqual( data.pattern.std(), 1.0 )
        data.zscore()

        # check z-scoring
        check = numpy.array([-2,-1,1,2,0,0,1,-1,-1,1,1,-1,0,0,0,0],dtype='float64')
        self.failUnless( (data.pattern ==  check.reshape(16,1)).all() )


    def testPatternShape(self):
        data = mvpa.MVPAPattern(numpy.ones((10,2,3,4)), 1, 1 )
        self.failUnless( data.pattern.shape == (10,24) )
        self.failUnless( data.origshape == (2,3,4) )


    def testFeature2Coord(self):
        origdata = numpy.random.standard_normal((10,2,4,3,5))
        data = mvpa.MVPAPattern( origdata, 2, 2 )

        def randomCoord(shape):
            return [ random.sample(range(size),1)[0] for size in shape ]

        # check 100 random coord2feature transformations
        for i in xrange(100):
            # choose random coord
            c = randomCoord(data.origshape)
            # tranform to feature_id
            id = data.getFeatureId(c)

            # compare data from orig array (selected by coord)
            # and data from pattern array (selected by feature id)
            orig = origdata[:,c[0],c[1],c[2],c[3]]
            pat = data.pattern[:, id ]

            self.failUnless( (orig == pat).all() )


    def testCoord2Feature(self):
        origdata = numpy.random.standard_normal((10,2,4,3,5))
        data = mvpa.MVPAPattern( origdata, 2, 2 )

        def randomCoord(shape):
            return [ random.sample(range(size),1)[0] for size in shape ]

        for id in xrange(data.nfeatures):
            # transform to coordinate
            c = data.getCoordinate(id)
            self.failUnlessEqual(len(c), 4)

            # compare data from orig array (selected by coord)
            # and data from pattern array (selected by feature id)
            orig = origdata[:,c[0],c[1],c[2],c[3]]
            pat = data.pattern[:, id ]

            self.failUnless( (orig == pat).all() )


def suite():
    return unittest.makeSuite(PatternTests)


if __name__ == '__main__':
    unittest.main()

