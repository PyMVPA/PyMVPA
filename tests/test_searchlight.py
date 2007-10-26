#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Unit tests for PyMVPA searchlight algorithm"""

import unittest
import mvpa.maskeddataset
import mvpa.searchlight as sl
import mvpa.knn as knn
import mvpa.svm as svm
import numpy as N

class SearchlightTests(unittest.TestCase):

    def setUp(self):
        data = N.random.standard_normal(( 100, 3, 6, 6 ))
        reg = N.concatenate( ( N.repeat( 0, 50 ),
                                   N.repeat( 1, 50 ) ) )
        orig = N.repeat( range(5), 10 )
        origin = N.concatenate( (orig, orig) )
        self.pattern = mvpa.maskeddataset.MaskedDataset( data, reg, origin )


    def testSearchlight(self):
        mask = N.zeros( (3, 6, 6) )
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
                                 N.ones((3,6,6)),
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
            ( N.array([ i in test_radii for i in N.unique(osl.bestradius) ]) \
              == True ).all() )


    def testSphericalROIMaskGenerator(self):
        # make dummy mask
        mask = N.zeros((4,4,4))
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

