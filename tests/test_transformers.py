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

from mvpa.misc.transformers import Absolute, OneMinus, RankOrder, \
     ReverseRankOrder, L1Normed, L2Normed, OverAxis

from tests_warehouse import sweepargs, datasets

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

    def testL2Norm(self):
        out = L2Normed(self.d2)
        self.failUnless(N.abs(N.sum(out*out)-1.0) < 1e-10)

    def testL1Norm(self):
        out = L1Normed(self.d2)
        self.failUnless(N.abs(N.sum(N.abs(out))-1.0) < 1e-10)


    def testOverAxis(self):
        data = datasets['uni4large'].samples[:120,0].reshape((2,3,4,5))
        # Simple transformer/combiner which collapses across given
        # dimension, e.g. sum
        for axis in [None, 0, 1, 2]:
            oversum = OverAxis(N.sum, axis=axis)(data)
            sum_ = N.sum(data, axis=axis)
            self.failUnless(N.all(sum_ == oversum))

        # Transformer which doesn't modify dimensionality of the data
        data = data.reshape((6, -1))
        overnorm = OverAxis(L2Normed, axis=1)(data)
        self.failUnless(N.linalg.norm(overnorm)!=1.0)
        for d in overnorm:
            self.failUnless(N.abs(N.linalg.norm(d) - 1.0)<0.00001)

        overnorm = OverAxis(L2Normed, axis=0)(data)
        self.failUnless(N.linalg.norm(overnorm)!=1.0)
        for d in overnorm.T:
            self.failUnless(N.abs(N.linalg.norm(d) - 1.0)<0.00001)


def suite():
    return unittest.makeSuite(TransformerTests)


if __name__ == '__main__':
    import runner

