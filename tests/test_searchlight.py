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

    def testSearchlight(self):
        data = numpy.random.standard_normal(( 100, 3, 6, 6 ))
        reg = numpy.concatenate( ( numpy.repeat( 0, 50 ),
                                   numpy.repeat( 1, 50 ) ) )
        orig = numpy.repeat( range(5), 10 )
        origin = numpy.concatenate( (orig, orig) )
        pattern = mvpa.MVPAPattern( data, reg, origin )
        slight = sl.Searchlight( pattern,
                                 numpy.ones( (3, 6, 6) ),
                                 3.0,
                                 elementsize = (3,3,3),
                                 forcesphere = True )

        # check virgin results
        self.failUnless( (slight.perfmean == 0).all() )
        self.failUnless( (slight.perfvar == 0).all() )
        self.failUnless( (slight.chisquare == 0).all() )
        self.failUnless( (slight.chanceprob == 0).all() )
        self.failUnless( (slight.spheresize == 0).all() )

        self.failUnless( slight.ncvfolds == 5 )

        # run searchlight
        slight.run(knn.kNN, verbose=True, k=5)

        # check that something happened
        self.failIf( (slight.perfmean == 0).all() )
        self.failIf( (slight.perfvar == 0).all() )
        self.failIf( (slight.chisquare == 0).all() )
        self.failIf( (slight.chanceprob == 0).all() )
        self.failIf( (slight.spheresize == 0).all() )


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

        # check whether earlier ROIs or not overwritten by
        # later ones
        self.failUnless( roi[2,1,1] == 1 )


def suite():
    return unittest.makeSuite(SearchlightTests)


if __name__ == '__main__':
    unittest.main()

