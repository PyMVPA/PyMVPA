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


    def testMapOverlap(self):
        mo = MapOverlap()

        maps = [[1,0,1,0],
                [1,0,0,1],
                [1,0,1,0]]

        overlap = mo(maps)

        self.failUnlessEqual(overlap, 1./len(maps[0]))
        self.failUnless((mo.overlap_map == [1,0,0,0]).all())
        self.failUnless((mo.spread_map == [0,0,1,1]).all())
        self.failUnless((mo.ovstats_map == [1,0,2./3,1./3]).all())

        mo = MapOverlap(overlap_threshold=0.5)
        overlap = mo(maps)
        self.failUnlessEqual(overlap, 2./len(maps[0]))
        self.failUnless((mo.overlap_map == [1,0,1,0]).all())
        self.failUnless((mo.spread_map == [0,0,0,1]).all())
        self.failUnless((mo.ovstats_map == [1,0,2./3,1./3]).all())


    def _testLoop(self):

        def genN(N=5, justi=False):
            for i in xrange(N):
                print "Y: ",i
                if justi:
                    yield i
                else:
                    yield i, "b" * i

        def caller(*args, **kwargs):
            """Accumulate me please"""
            print "ARGS: ", args
            print "KWARGS: ", kwargs

        def callerAB(a, b):
            return "%s:%s " %(a, b)

        class callerC(object):
            def __init__(self):
                self.i = 0

            def __call__(self, a, b):
                self.i += 1
                return a

        r1 = loop(genN, callerC(), attribs=["i"])
        r2 = loop(lambda :genN(justi=False), callerAB)
        print r1
        print r2

def suite():
    return unittest.makeSuite(SupportFxTests)


if __name__ == '__main__':
    import test_runner

