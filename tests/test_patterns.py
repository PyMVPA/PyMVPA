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


        # test unique automatic origins
        data.addPattern( numpy.random.standard_normal((2,5)), 3 )
        self.failUnless( (data.origin == numpy.array([1,2,2,3,4]) ).all() )

        # test unique class labels
        self.failUnless( (data.reglabels == numpy.array([1,2,3]) ).all() )

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


    def testZScoring(self):
        # dataset: mean=2, std=1
        pat = numpy.array( (0,1,3,4,2,2,3,1,1,3,3,1,2,2,2,2) )
        data = mvpa.MVPAPattern(pat.reshape((16,1)), 1, 1)
        self.failUnlessEqual( data.pattern.mean(), 2.0 )
        self.failUnlessEqual( data.pattern.std(), 1.0 )
        data.zscore(origin=True)

        # check z-scoring
        check = numpy.array([-2,-1,1,2,0,0,1,-1,-1,1,1,-1,0,0,0,0],dtype='float64')
        self.failUnless( (data.pattern ==  check.reshape(16,1)).all() )

        data = mvpa.MVPAPattern(pat.reshape((16,1)), 1, 1)
        data.zscore(origin=False)
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


    def testFeatureSelection(self):
        origdata = numpy.random.standard_normal((10,2,4,3,5))
        data = mvpa.MVPAPattern( origdata, 2, 2 )

        unmasked = data.pattern.copy()

        # default must be no mask
        self.failUnless( data.nfeatures == 120 )

        # check that full mask uses all features
        sel = data.selectFeaturesByMask( numpy.ones((2,4,3,5)) )
        self.failUnless( sel.nfeatures == data.pattern.shape[1] )
        self.failUnless( data.nfeatures == 120 )

        # check partial array mask
        partial_mask = numpy.zeros((2,4,3,5), dtype='uint')
        partial_mask[0,0,2,2] = 1
        partial_mask[1,2,2,0] = 1
        sel = data.selectFeaturesByMask( partial_mask )
        self.failUnless( sel.nfeatures == 2 )
        self.failUnless( sel.getFeatureMask().shape == (2,4,3,5))

        # check selection with feature list
        sel = data.selectFeaturesById( [0,37,119] )
        self.failUnless(sel.nfeatures == 3)

        # check size of the masked patterns
        self.failUnless( sel.pattern.shape == (10,3) )

        # check that the right features are selected
        self.failUnless( (unmasked[:,[0,37,119]]==sel.pattern).all() )


    def testPatternSelection(self):
        origdata = numpy.random.standard_normal((10,2,4,3,5))
        data = mvpa.MVPAPattern( origdata, 2, 2 )

        self.failUnless( data.npatterns == 10 )

        # set single pattern to enabled
        sel=data.selectPatterns(5)
        self.failUnless( sel.npatterns == 1 )
        self.failUnless( data.npatterns == 10 )

        # check duplicate selections
        sel = data.selectPatterns([5,5])
        self.failUnless( sel.npatterns == 2 )
        self.failUnless( (sel.pattern[0] == sel.pattern[1]).all() )
        self.failUnless( len(sel.reg) == 2 )
        self.failUnless( len(sel.origin) == 2 )

        self.failUnless( sel.pattern.shape == (2,120) )

    def testCombinedPatternAndFeatureMasking(self):
        data = mvpa.MVPAPattern(
            numpy.arange( 20 ).reshape( (4,5) ), 1, 1 )

        self.failUnless( data.npatterns == 4 )
        self.failUnless( data.nfeatures == 5 )
        fsel = data.selectFeaturesById([1,2])
        fpsel = fsel.selectPatterns([0,3])
        self.failUnless( fpsel.npatterns == 2 )
        self.failUnless( fpsel.nfeatures == 2 )

        self.failUnless( (fpsel.pattern == [[1,2],[16,17]]).all() )


    def testOrigMaskExtraction(self):
        origdata = numpy.random.standard_normal((10,2,4,3))
        data = mvpa.MVPAPattern( origdata, 2, 2 )

        # check with custom mask
        sel = data.selectFeaturesById([5])
        self.failUnless( sel.pattern.shape[1] == 1 )
        origmask = sel.getFeatureMask()
        self.failUnless( origmask[0,1,2] == True )
        self.failUnless( origmask.shape == data.origshape == (2,4,3) )



    def testPatternMerge(self):
        data1 = mvpa.MVPAPattern( numpy.ones((5,5,1)), 1, 1 )
        data2 = mvpa.MVPAPattern( numpy.ones((3,5,1)), 2, 1 )

        merged = data1 + data2

        self.failUnless( merged.npatterns == 8 )
        self.failUnless( (merged.reg == [ 1,1,1,1,1,2,2,2]).all() )
        self.failUnless( (merged.origin == [ 1,1,1,1,1,1,1,1]).all() )

        data1 += data2

        self.failUnless( data1.npatterns == 8 )
        self.failUnless( (data1.reg == [ 1,1,1,1,1,2,2,2]).all() )
        self.failUnless( (data1.origin == [ 1,1,1,1,1,1,1,1]).all() )


    def testRegressorRandomizationAndSampling(self):
        data = mvpa.MVPAPattern( numpy.ones((5,1)), range(5), 1 )
        data.addPattern( numpy.ones((5,1))+1, range(5), 2 )
        data.addPattern( numpy.ones((5,1))+2, range(5), 3 )
        data.addPattern( numpy.ones((5,1))+3, range(5), 4 )
        data.addPattern( numpy.ones((5,1))+4, range(5), 5 )

        self.failUnless( data.patperreg == [ 5,5,5,5,5 ] )

        sample = data.getPatternSample( 2 )

        self.failUnless( sample.patperreg == [ 2,2,2,2,2 ] )

        self.failUnless( (data.originlabels == range(1,6)).all() )

        # store the old regs
        origregs = data.reg.copy()

        data.permutatedRegressors(True)

        self.failIf( (data.reg == origregs).all() )

        data.permutatedRegressors(False)

        self.failUnless( (data.reg == origregs).all() )

        # now try another object with the same data
        data2 = mvpa.MVPAPattern( data.pattern, data.reg, data.origin )

        # regressors are the same as the originals
        self.failUnless( (data2.reg == origregs).all() )

        # now permutate in the new object
        data2.permutatedRegressors( True )

        # must not affect the old one
        self.failUnless( (data.reg == origregs).all() )
        # but only the new one
        self.failIf( (data2.reg == origregs).all() )


    def testFeatureMasking(self):
        mask = numpy.zeros((5,3),dtype='bool')
        mask[2,1] = True; mask[4,0] = True
        data = mvpa.MVPAPattern(
            numpy.arange( 60 ).reshape( (4,5,3) ), 1, 1, mask=mask )

        # check simple masking
        self.failUnless( data.nfeatures == 2 )
        self.failUnless( data.getFeatureId( (2,1) ) == 0 
                     and data.getFeatureId( (4,0) ) == 1 )
        self.failUnlessRaises( ValueError, data.getFeatureId, (2,3) )
        self.failUnless( data.getFeatureMask().shape == (5,3) )
        self.failUnless( tuple(data.getCoordinate( 1 )) == (4,0) )

        # selection should be idempotent
        self.failUnless(data.selectFeaturesByMask( mask ).nfeatures == data.nfeatures )
        # check that correct feature get selected
        self.failUnless( (data.selectFeaturesById([1]).pattern[:,0] \
                          == numpy.array([12, 27, 42, 57]) ).all() )
        self.failUnless(tuple( data.selectFeaturesById([1]).getCoordinate(0) ) == (4,0) )
        self.failUnless( data.selectFeaturesById([1]).getFeatureMask().sum() == 1 )


    def testROIMasking(self):
        mask=numpy.array([i/6 for i in range(60)], dtype='int').reshape(6,10)
        data = mvpa.MVPAPattern(
            numpy.arange( 180 ).reshape( (3,6,10) ), 1, 1, mask=mask )

        self.failIf( data.getFeatureMask().dtype == 'bool' )
        # check that the 0 masked features get cut
        self.failUnless( data.nfeatures == 54 )
        self.failUnless( (data.pattern[:,0] == [6,66,126]).all() )
        self.failUnless( data.getFeatureMask().shape == (6,10) )

        featsel = data.selectFeaturesById([19])
        self.failUnless( (data.pattern[:,19] == featsel.pattern[:,0]).all() )

        # check single ROI selection works
        roisel = data.selectFeaturesByGroup([4])
        self.failUnless( (data.pattern[:,19] == roisel.pattern[:,1]).all() )

        # check dual ROI selection works (plus keep feature order)
        roisel = data.selectFeaturesByGroup([6,4])
        self.failUnless( (data.pattern[:,19] == roisel.pattern[:,1]).all() )
        self.failUnless( (data.pattern[:,32] == roisel.pattern[:,8]).all() )

        # check if feature coords can be recovered
        self.failUnless( (roisel.getCoordinate(8) == (3,8)).all() )


def suite():
    return unittest.makeSuite(PatternTests)


if __name__ == '__main__':
    unittest.main()

