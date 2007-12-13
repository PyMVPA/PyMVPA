#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA serial feature inclusion algorithm"""

import unittest

import numpy as N

from mvpa.misc.support import *


class SupportFxTests(unittest.TestCase):

    def testTransformWithBoxcar(self):
        data = N.arange(10)
        sp = N.arange(10)

        # check if stupid thing don't work
        self.failUnlessRaises(ValueError, 
                              transformWithBoxcar,
                              data,
                              sp,
                              0 )

        # now do an identity transformation
        trans = transformWithBoxcar( data, sp, 1)
        self.failUnless( (trans == data).all() )

        # now check for illegal boxes
        self.failUnlessRaises( ValueError,
                               transformWithBoxcar,
                               data,
                               sp,
                               2 )

        # now something that should work
        sp = N.arange(9)
        trans = transformWithBoxcar( data, sp, 2)
        self.failUnless( ( trans == \
                           [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5] ).all() )


        # now test for proper data shape
        data = N.ones((10,3,4,2))
        sp = [ 2, 4, 3, 5 ]
        trans = transformWithBoxcar( data, sp, 4)
        self.failUnless( trans.shape == (4,3,4,2) )


    def testConfusionMatrix(self):
        data = N.array([1,2,1,2,2,2,3,2,1], ndmin=2).T
        reg = N.array([1,1,1,2,2,2,3,3,3])

        tbl = buildConfusionMatrix( N.unique(reg), reg, N.array([1,2,1,2,2,2,3,2,1]) )

        # should be square matrix (len(reglabels) x len(reglabels)
        self.failUnless( tbl.shape == (3,3) )

        # check table content
        self.failUnless( (tbl == [[2,1,0],[0,3,0],[1,1,1]]).all() )

        # check pretty print
        matrix = N.array( [ [100,900,1], [10,100,2], [1,0,0] ] )
        labels = ["s", "looong","nah"]
        s=pprintConfusionMatrix(labels=labels, matrix=matrix)

        # just a silly test to make sure that printing works
        self.failUnless(len(s)>100)


    def testMofNCombinations(self):
        self.failUnlessEqual(
            getUniqueLengthNCombinations( range(3), 1 ), [[0],[1],[2]] )
        self.failUnlessEqual(
            getUniqueLengthNCombinations(
                        range(4), 2 ),
                        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
                        )
        self.failUnlessEqual(
            getUniqueLengthNCombinations(
                        range(4), 3 ), [[0, 1, 2], [0, 1, 3], [0, 2, 3]] )


    def testBreakPoints(self):
        items_cont = [0, 0, 0, 1, 1, 1, 3, 3, 2]
        items_noncont = [0, 0, 1, 1, 0, 3, 2]
        self.failUnlessRaises(ValueError, getBreakPoints, items_noncont)
        self.failUnlessEqual(getBreakPoints(items_noncont, contiguous=False),
                             [0, 2, 4, 5, 6])
        self.failUnlessEqual(getBreakPoints(items_cont), [0, 3, 6, 8])
        self.failUnlessEqual(getBreakPoints(items_cont, contiguous=False),
                             [0, 3, 6, 8])



def suite():
    return unittest.makeSuite(SupportFxTests)


if __name__ == '__main__':
    import test_runner

