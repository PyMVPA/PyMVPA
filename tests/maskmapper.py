#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Unit tests for PyMVPA mask mapper"""


from mvpa.datasets.maskmapper import *
from mvpa.datasets.metric import *
import unittest
import numpy as N

class MaskMapperTests(unittest.TestCase):

    def testForwardMaskMapper(self):
        mask = N.ones((3,2))
        map_ = MaskMapper(mask)

        # test shape reports
        self.failUnless( map_.dsshape == mask.shape )
        self.failUnless( map_.nfeatures == 6 )

        # test 1sample mapping
        self.failUnless( ( map_.forward( N.arange(6).reshape(3,2) ) \
                           == [0,1,2,3,4,5]).all() )

        # test 4sample mapping
        foursample = map_.forward( N.arange(24).reshape(4,3,2)) 
        self.failUnless( ( foursample \
                           == [[0,1,2,3,4,5],
                               [6,7,8,9,10,11],
                               [12,13,14,15,16,17],
                               [18,19,20,21,22,23]]).all() )

        # check incomplete masks
        mask[1,1] = 0
        map_ = MaskMapper(mask)
        self.failUnless( map_.nfeatures == 5 )
        self.failUnless( ( map_.forward( N.arange(6).reshape(3,2) ) \
                           == [0,1,2,4,5]).all() )

        # check that it doesn't accept wrong dataspace
        self.failUnlessRaises( ValueError,
                               map_.forward,
                               N.arange(4).reshape(2,2) )


    def testReverseMaskMapper(self):
        mask = N.ones((3,2))
        mask[1,1] = 0
        map_ = MaskMapper(mask)

        rmapped = map_(N.arange(1,6))
        self.failUnless( rmapped.shape == (3,2) )
        self.failUnless( rmapped[1,1] == 0 )
        self.failUnless( rmapped[2,1] == 5 )


        # check that it doesn't accept wrong dataspace
        self.failUnlessRaises( ValueError,
                               map_,
                               N.arange(6))

        rmapped2 = map_(N.arange(1,11).reshape(2,5))
        self.failUnless( rmapped2.shape == (2,3,2) )
        self.failUnless( rmapped2[0,1,1] == 0 )
        self.failUnless( rmapped2[1,1,1] == 0 )
        self.failUnless( rmapped2[0,2,1] == 5 )
        self.failUnless( rmapped2[1,2,1] == 10 )

    def testMaskMetricMapper(self):
        """ Test MaskMetricMapper
        """
        mask = N.ones((3,2))
        mask[1,1] = 0

        # take space with non-square elements
        neighborFinder = DescreteMetric([0.5, 2])
        map_ = MaskMetricMapper(mask, neighborFinder)

        # test getNeighbors
        # now it returns list of arrays
        #target = [N.array([0, 0]), N.array([0, 1]),
        #          N.array([1, 0]), N.array([2, 0])]
        #result = map_.getNeighborIn([0, 0], 2)
        #self.failUnless(N.array(map(lambda x,y:(x==y).all(), result, target)).all())

        # check by providing outId
        target = [0,1,2,3]
        result = map_.getNeighbors(0, 2.1)
        self.failUnless( result == target )


def suite():
    return unittest.makeSuite(MaskMapperTests)


if __name__ == '__main__':
    unittest.main()

