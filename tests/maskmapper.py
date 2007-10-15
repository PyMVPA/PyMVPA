### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Unit tests for PyMVPA mask mapper
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

from mvpa.maskmapper import *
import unittest
import numpy as N

class MaskMapperTests(unittest.TestCase):

    def testForwardMaskMapper(self):
        mask = N.ones((3,2))
        map = MaskMapper(mask)

        # test shape reports
        self.failUnless( map.dsshape == mask.shape )
        self.failUnless( map.nfeatures == 6 )

        # test 1sample mapping
        self.failUnless( ( map.forward( N.arange(6).reshape(3,2) ) \
                           == [0,1,2,3,4,5]).all() )

        # test 4sample mapping
        foursample = map.forward( N.arange(24).reshape(4,3,2)) 
        self.failUnless( ( foursample \
                           == [[0,1,2,3,4,5],
                               [6,7,8,9,10,11],
                               [12,13,14,15,16,17],
                               [18,19,20,21,22,23]]).all() )

        # check incomplete masks
        mask[1,1] = 0
        map = MaskMapper(mask)
        self.failUnless( map.nfeatures == 5 )
        self.failUnless( ( map.forward( N.arange(6).reshape(3,2) ) \
                           == [0,1,2,4,5]).all() )

        # check that it doesn't accept wrong dataspace
        self.failUnlessRaises( ValueError,
                               map.forward,
                               N.arange(4).reshape(2,2) )


    def testReverseMaskMapper(self):
        mask = N.ones((3,2))
        mask[1,1] = 0
        map = MaskMapper(mask)

        rmapped = map(N.arange(1,6))
        self.failUnless( rmapped.shape == (3,2) )
        self.failUnless( rmapped[1,1] == 0 )
        self.failUnless( rmapped[2,1] == 5 )


        # check that it doesn't accept wrong dataspace
        self.failUnlessRaises( ValueError,
                               map,
                               N.arange(6))

        rmapped2 = map(N.arange(1,11).reshape(2,5))
        self.failUnless( rmapped2.shape == (2,3,2) )
        self.failUnless( rmapped2[0,1,1] == 0 )
        self.failUnless( rmapped2[1,1,1] == 0 )
        self.failUnless( rmapped2[0,2,1] == 5 )
        self.failUnless( rmapped2[1,2,1] == 10 )


def suite():
    return unittest.makeSuite(MaskMapperTests)


if __name__ == '__main__':
    unittest.main()

