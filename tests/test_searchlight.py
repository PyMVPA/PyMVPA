### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Unit tests for PyMVPA searchlight algorithm
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

import unittest
import mvpa
import mvpa.searchlight as sl
import mvpa.knn as knn
import numpy

class SearchlightTests(unittest.TestCase):

    def setUp(self):
        data = numpy.random.standard_normal(( 100, 3, 6, 6 ))
        reg = numpy.concatenate( ( numpy.repeat( 0, 50 ),
                                   numpy.repeat( 1, 50 ) ) )
        orig = numpy.repeat( range(5), 10 )
        origin = numpy.concatenate( (orig, orig) )
        self.pattern = mvpa.MVPAPattern( data, reg, origin )
        self.slight = sl.Searchlight( self.pattern,
                                      numpy.ones( (3, 6, 6) ),
                                      elementsize = (3,3,3),
                                      forcesphere = True )


    def testSearchlight(self):
        # check virgin results
        self.failUnless( (self.slight.perfmean == 0).all() )
        self.failUnless( (self.slight.perfvar == 0).all() )
        self.failUnless( (self.slight.chisquare == 0).all() )
        self.failUnless( (self.slight.chanceprob == 0).all() )
        self.failUnless( (self.slight.spheresize == 0).all() )

        self.failUnless( self.slight.ncvfolds == 5 )

        # run searchlight
        self.slight(knn.kNN, 3.0, verbose=False, k=5)

        # check that something happened
        self.failIf( (self.slight.perfmean == 0).all() )
        self.failIf( (self.slight.perfvar == 0).all() )
        self.failIf( (self.slight.chisquare == 0).all() )
        self.failIf( (self.slight.chanceprob == 0).all() )
        self.failIf( (self.slight.spheresize == 0).all() )


    def testOptimalSearchlight(self):
        test_radii = [3,6,9]
        osl = sl.OptimalSearchlight( self.slight,
                                     test_radii,
                                     knn.kNN,
                                     verbose = False,
                                     k=5 )

        # check that only valid radii are in bestradius array
        self.failUnless( 
            ( numpy.array([ i in test_radii for i in numpy.unique(osl.bestradius) ]) \
              == True ).all() )


    def testSphericalROIMaskGenerator(self):
        # make dummy mask
        mask = numpy.zeros((4,4,4))
        mask[2,2,2] = 1
        mask[1,1,1] = 1
        mask[2,1,1] = 1

        # generate ROI mask
        roi = sl.makeSphericalROIMask( mask, 1 )

        self.failUnless( mask.shape == roi.shape)
        self.failUnless( roi.dtype == 'int32' )


def suite():
    return unittest.makeSuite(SearchlightTests)


if __name__ == '__main__':
    unittest.main()

