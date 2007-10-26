#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Unit tests for PyMVPA serial feature inclusion algorithm"""

import unittest
import numpy as N
import mvpa.support as support

class SupportFxTests(unittest.TestCase):

    def testTransformWithBoxcar(self):
        data = N.arange(10)
        sp = N.arange(10)

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
        sp = N.arange(9)
        trans = support.transformWithBoxcar( data, sp, 2)
        self.failUnless( ( trans == \
                           [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5] ).all() )


        # now test for proper data shape
        data = N.ones((10,3,4,2))
        sp = [ 2, 4, 3, 5 ]
        trans = support.transformWithBoxcar( data, sp, 4)
        self.failUnless( trans.shape == (4,3,4,2) )


    def testConfusionMatrix(self):
        data = N.array([1,2,1,2,2,2,3,2,1], ndmin=2).T
        reg = N.array([1,1,1,2,2,2,3,3,3])

        tbl = support.buildConfusionMatrix( N.unique(reg), reg, N.array([1,2,1,2,2,2,3,2,1]) )

        # should be square matrix (len(reglabels) x len(reglabels)
        self.failUnless( tbl.shape == (3,3) )

        # check table content
        self.failUnless( (tbl == [[2,1,0],[0,3,0],[1,1,1]]).all() )


    def testMofNCombinations(self):
        self.failUnlessEqual(
            support.getUniqueLengthNCombinations( range(3), 1 ), [[0],[1],[2]] )
        self.failUnlessEqual(
            support.getUniqueLengthNCombinations(
                        range(4), 2 ),
                        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
                        )
        self.failUnlessEqual(
            support.getUniqueLengthNCombinations(
                        range(4), 3 ), [[0, 1, 2], [0, 1, 3], [0, 2, 3]] )


def suite():
    return unittest.makeSuite(SupportFxTests)


if __name__ == '__main__':
    unittest.main()

