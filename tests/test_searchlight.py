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
import mvpa.maskeddataset
import mvpa.searchlight as sl
import mvpa.knn as knn
import mvpa.svm as svm
import numpy

class SearchlightTests(unittest.TestCase):

    def setUp(self):
        data = numpy.random.standard_normal(( 100, 3, 6, 6 ))
        reg = numpy.concatenate( ( numpy.repeat( 0, 50 ),
                                   numpy.repeat( 1, 50 ) ) )
        orig = numpy.repeat( range(5), 10 )
        origin = numpy.concatenate( (orig, orig) )
        self.pattern = mvpa.maskeddataset.MaskedDataset( data, reg, origin )


    def testSearchlight(self):
        mask = numpy.zeros( (3, 6, 6) )
        mask[0,0,0] = 1
        mask[1,3,2] = 1
        slight = sl.Searchlight( self.pattern,
                                 mask,
                                 knn.kNN(k=5),
                                 elementsize = (3,3,3),
                                 forcesphere = True,
                                 verbose = False )

        # check virgin results
        self.failUnless( (slight.perfmean == 0).all() )
        self.failUnless( (slight.perfvar == 0).all() )
        self.failUnless( (slight.chisquare == 0).all() )
        self.failUnless( (slight.chanceprob == 0).all() )
        self.failUnless( (slight.spheresize == 0).all() )

        # run searchlight
        slight(3.0)

        # check that something happened
        self.failIf( (slight.perfmean == 0).all() )
        self.failIf( (slight.perfvar == 0).all() )
        self.failIf( (slight.chisquare == 0).all() )
        self.failIf( (slight.chanceprob == 0).all() )
        self.failIf( (slight.spheresize == 0).all() )


    def testOptimalSearchlight(self):
        slight = sl.Searchlight( self.pattern,
                                 numpy.ones((3,6,6)),
                                 svm.SVM(),
                                 elementsize = (3,3,3),
                                 forcesphere = True,
                                 verbose = False )
        test_radii = [3,6,9]
        clf = knn.kNN(k=5)
        osl = sl.OptimalSearchlight( slight,
                                     test_radii,
                                     verbose = False )
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

