#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA transformers."""

import unittest
import numpy as N

from mvpa.misc.transformers import Absolute, OneMinus, RankOrder, UnitNorm, \
     ReverseRankOrder


class TransformerTests(unittest.TestCase):

    def setUp(self):
        self.d1 = N.array([ 1,  0, -1, -2, -3])
        self.d2 = N.array([ 2.3,  0, -1, 2, -30, 1])

    def testAbsolute(self):
        # generate 100 values (gaussian noise mean -1000 -> all negative)
        out = Absolute(N.random.normal(-1000, size=100))

        self.failUnless(out.min() >= 0)
        self.failUnless(len(out) == 100)

    def testAbsolute2(self):
        target = self.d1
        out = OneMinus(N.arange(5))
        self.failUnless((out == target).all())

    def testRankOrder(self):
        nelements = len(self.d2)
        out = RankOrder(self.d2)
        outr = ReverseRankOrder(self.d2)
        uout = N.unique(out)
        uoutr = N.unique(outr)
        self.failUnless((uout == N.arange(nelements)).all(),
                        msg="We should get all indexes. Got just %s" % uout)
        self.failUnless((uoutr == N.arange(nelements)).all(),
                        msg="We should get all indexes. Got just %s" % uoutr)
        self.failUnless((out+outr+1 == nelements).all())
        self.failUnless((out == [ 0,  3,  4,  1,  5,  2]).all())


    def testUnitNorm(self):
        out = UnitNorm(self.d2)
        self.failUnless(N.abs(N.sum(out*out)-1.0) < 1e-10)


def suite():
    return unittest.makeSuite(TransformerTests)


if __name__ == '__main__':
    import runner

