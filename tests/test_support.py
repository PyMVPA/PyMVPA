### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#
#    Unit tests for PyMVPA serial feature inclusion algorithm
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

import unittest
import numpy as np
import mvpa.support as support

class SupportFxTests(unittest.TestCase):

    def testTransformWithBoxcar(self):
        data = np.arange(10)
        sp = np.arange(10)

        # check if stupid thing don't work
        self.failUnlessRaises(ValueError, 
                              support.transformWithBoxcar,
                              data,
                              sp,
                              0 )

        # now do an identity transformation
        trans = support.transformWithBoxcar( data, sp, 1)
        self.failUnless( (trans == data).all() )

        # now check for illegal boxes
        self.failUnlessRaises( ValueError,
                               support.transformWithBoxcar,
                               data,
                               sp,
                               2 )

        # now something that should work
        sp = np.arange(9)
        trans = support.transformWithBoxcar( data, sp, 2)
        self.failUnless( ( trans == \
                           [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5] ).all() )


        # now test for proper data shape
        data = np.ones((10,3,4,2))
        sp = [ 2, 4, 3, 5 ]
        trans = support.transformWithBoxcar( data, sp, 4)
        self.failUnless( trans.shape == (4,3,4,2) )


    def testConfusionMatrix(self):
        data = np.array([1,2,1,2,2,2,3,2,1], ndmin=2).T
        reg = np.array([1,1,1,2,2,2,3,3,3])

        tbl = support.buildConfusionMatrix( np.unique(reg), reg, np.array([1,2,1,2,2,2,3,2,1]) )

        # should be square matrix (len(reglabels) x len(reglabels)
        self.failUnless( tbl.shape == (3,3) )

        # check table content
        self.failUnless( (tbl == [[2,1,0],[0,3,0],[1,1,1]]).all() )



def suite():
    return unittest.makeSuite(SupportFxTests)


if __name__ == '__main__':
    unittest.main()

