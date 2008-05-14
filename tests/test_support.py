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
from mvpa.datasets.splitter import NFoldSplitter
from mvpa.clfs.transerror import TransferError
from tests_warehouse import getMVPattern
from tests_warehouse_clfs import *


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
        trans = transformWithBoxcar(data, sp, 1)
        self.failUnless( (trans == data).all() )

        # now check for illegal boxes
        self.failUnlessRaises(ValueError,
                              transformWithBoxcar,
                              data,
                              sp,
                              2)

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


    def testHarvester(self):
        # do very simple list comprehension
        self.failUnlessEqual(
            [(-1)*i for i in range(5)],
            Harvester(xrange,
                      [HarvesterCall(lambda x: (-1)*x, expand_args=False)])
            (5))


        # do clf cross-validation on a dataset with a very high SNR
        cv = Harvester(NFoldSplitter(cvtype=1),
                       [HarvesterCall(TransferError(sample_clf_nl), argfilter=[1,0])])
        data = getMVPattern(10)
        err = N.array(cv(data))

        # has to be perfect
        self.failUnless((err < 0.1).all())
        self.failUnlessEqual(err.shape, (len(data.uniquechunks),))

        # now same stuff but two classifiers at once
        cv = Harvester(NFoldSplitter(cvtype=1),
                  [HarvesterCall(TransferError(sample_clf_nl), argfilter=[1,0]),
                   HarvesterCall(TransferError(sample_clf_nl), argfilter=[1,0])])
        err = N.array(cv(data))
        self.failUnlessEqual(err.shape, (2,len(data.uniquechunks)))

        # only one again, but this time remember confusion matrix
        cv = Harvester(NFoldSplitter(cvtype=1),
                  [HarvesterCall(TransferError(sample_clf_nl,
                                               enable_states=['confusion']),
                                 argfilter=[1,0], attribs=['confusion'])])
        res = cv(data)

        self.failUnless(isinstance(res, dict))
        self.failUnless(res.has_key('confusion') and res.has_key('result'))
        self.failUnless(len(res['result']) == len(data.uniquechunks))



def suite():
    return unittest.makeSuite(SupportFxTests)


if __name__ == '__main__':
    import runner

