# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Boxcar mapper"""


import unittest
from mvpa.support.copy import deepcopy
import numpy as N

from mvpa.mappers.boxcar import BoxcarMapper
from mvpa.datasets import Dataset


class BoxcarMapperTests(unittest.TestCase):

    def testSimple(self):
        """Just the same tests as for transformWithBoxcar.

        Mention that BoxcarMapper doesn't apply function with each boxcar
        """
        data = N.arange(10)
        sp = N.arange(10)

        # check if stupid thing don't work
        self.failUnlessRaises(ValueError,
                              BoxcarMapper,
                              sp,
                              0 )

        # now do an identity transformation
        bcm = BoxcarMapper(sp, 1)
        trans = bcm(data)
        # ,0 is a feature below, so we get explicit 2D out of 1D
        self.failUnless( (trans[:,0] == data).all() )

        # now check for illegal boxes
        self.failUnlessRaises(ValueError,
                              BoxcarMapper(sp, 2),
                              data)

        # now something that should work
        sp = N.arange(9)
        bcm = BoxcarMapper(sp,2)
        trans = bcm(data)
        self.failUnless( (trans == N.vstack((N.arange(9),
                                             N.arange(9)+1)).T ).all() )


        # now test for proper data shape
        data = N.ones((10,3,4,2))
        sp = [ 2, 4, 3, 5 ]
        trans = BoxcarMapper(sp, 4)(data)
        self.failUnless( trans.shape == (4,4,3,4,2) )

        # test reverse
        data = N.arange(240).reshape(10, 3, 4, 2)
        sp = [ 2, 4, 3, 5 ]
        m = BoxcarMapper(sp, 2)
        mp = m.forward(data)
        self.failUnless(mp.shape == (4, 2, 3, 4, 2))

        # try full reconstruct
        mr = m.reverse(mp)
        # shape has to match
        self.failUnless(mr.shape == data.shape)
        # first two samples where not in any of the boxcars and cannot be
        # reconstructed
        self.failUnless(N.sum(mr[:2]) == 0)
        # same for the last ones
        self.failUnless(N.sum(mr[7:]) == 0)

        # check proper reconstruction of non-conflicting sample
        self.failUnless((mr[2].ravel() == N.arange(48, 72)).all())

        # check proper reconstruction of samples being part of multiple
        # mapped samples
        self.failUnless((mr[3].ravel() == N.arange(72, 96)).all())

        # test reverse of a single sample
        singlesample = N.arange(48).reshape(2, 3, 4, 2)
        self.failUnless((singlesample == m.reverse(singlesample)).all())
        # should not work for shape mismatch
        self.failUnlessRaises(ValueError, m.reverse, singlesample[0])

        # check broadcasting of 'raw' samples into proper boxcars on forward()
        bc = m.forward(N.arange(24).reshape(3, 4, 2))
        self.failUnless((bc ==
                         N.array(2 * [N.arange(24).reshape(3, 4, 2)])).all())


    def testIds(self):
        data = N.arange(20).reshape( (10,2) )
        bcm = BoxcarMapper([1, 4, 6], 3)
        trans = bcm(data)

        self.failUnlessEqual(bcm.isValidInId( [1] ), True)
        self.failUnlessEqual(bcm.isValidInId( [0,1] ), False)

        self.failUnlessEqual(bcm.isValidOutId( [1] ), True)
        self.failUnlessEqual(bcm.isValidOutId( [3] ), False)
        self.failUnlessEqual(bcm.isValidOutId( [0,1] ), True)
        self.failUnlessEqual(bcm.isValidOutId( [0,1,0] ), False)



def suite():
    return unittest.makeSuite(BoxcarMapperTests)


if __name__ == '__main__':
    import runner

