#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA pattern handling"""

from mvpa.datasets.maskeddataset import MaskedDataset
from mvpa.datasets.splitter import NFoldSplitter, OddEvenSplitter, \
                                   NoneSplitter, HalfSplitter
import unittest
import numpy as N


class SplitterTests(unittest.TestCase):

    def setUp(self):
        self.data = \
            MaskedDataset(samples=N.random.normal(size=(100,10)),
                           labels=[ i%4 for i in range(100) ],
                            chunks=[ i/10 for i in range(100)])



    def testSimplestCVPatGen(self):
        # create the generator
        nfs = NFoldSplitter(cvtype=1)

        # now get the xval pattern sets One-Fold CV)
        xvpat = [ (train, test) for (train,test) in nfs(self.data) ]

        self.failUnless( len(xvpat) == 10 )

        for i,p in enumerate(xvpat):
            self.failUnless( len(p) == 2 )
            self.failUnless( p[0].nsamples == 90 )
            self.failUnless( p[1].nsamples == 10 )
            self.failUnless( p[1].chunks[0] == i )


    def testOddEvenSplit(self):
        oes = OddEvenSplitter()

        splits = [ (train, test) for (train, test) in oes(self.data) ]

        self.failUnless(len(splits) == 2)

        for i,p in enumerate(splits):
            self.failUnless( len(p) == 2 )
            self.failUnless( p[0].nsamples == 50 )
            self.failUnless( p[1].nsamples == 50 )

        self.failUnless((splits[0][1].uniquechunks == [1, 3, 5, 7, 9]).all())
        self.failUnless((splits[0][0].uniquechunks == [0, 2, 4, 6, 8]).all())
        self.failUnless((splits[1][0].uniquechunks == [1, 3, 5, 7, 9]).all())
        self.failUnless((splits[1][1].uniquechunks == [0, 2, 4, 6, 8]).all())

        # check if it works on pure odd and even chunk ids
        moresplits = [ (train, test) for (train, test) in oes(splits[0][0])]

        for split in moresplits:
            self.failUnless(split[0] != None)
            self.failUnless(split[1] != None)


    def testHalfSplit(self):
        hs = HalfSplitter()

        splits = [ (train, test) for (train, test) in hs(self.data) ]

        self.failUnless(len(splits) == 2)

        for i,p in enumerate(splits):
            self.failUnless( len(p) == 2 )
            self.failUnless( p[0].nsamples == 50 )
            self.failUnless( p[1].nsamples == 50 )

        self.failUnless((splits[0][1].uniquechunks == [0, 1, 2, 3, 4]).all())
        self.failUnless((splits[0][0].uniquechunks == [5, 6, 7, 8, 9]).all())
        self.failUnless((splits[1][1].uniquechunks == [5, 6, 7, 8, 9]).all())
        self.failUnless((splits[1][0].uniquechunks == [0, 1, 2, 3, 4]).all())

        # check if it works on pure odd and even chunk ids
        moresplits = [ (train, test) for (train, test) in hs(splits[0][0])]

        for split in moresplits:
            self.failUnless(split[0] != None)
            self.failUnless(split[1] != None)



    def testNoneSplitter(self):
        nos = NoneSplitter()

        splits = [ (train, test) for (train, test) in nos(self.data) ]

        self.failUnless(len(splits) == 1)
        self.failUnless(splits[0][0] == None)
        self.failUnless(splits[0][1].nsamples == 100)

        # test sampling tools
        nos = NoneSplitter(nrunspersplit=3,
                           nsecondsamples=10)
        splits = [ (train, test) for (train, test) in nos(self.data) ]

        self.failUnless(len(splits) == 3)
        for split in splits:
            self.failUnless(split[0] == None)
            self.failUnless(split[1].nsamples == 40)
            self.failUnless(split[1].samplesperlabel.values() == [10,10,10,10])


    def testLabelSplitter(self):
        oes = OddEvenSplitter(attr='labels')

        splits = [ (first, second) for (first, second) in oes(self.data) ]

        self.failUnless((splits[0][0].uniquelabels == [0,2]).all())
        self.failUnless((splits[0][1].uniquelabels == [1,3]).all())
        self.failUnless((splits[1][0].uniquelabels == [1,3]).all())
        self.failUnless((splits[1][1].uniquelabels == [0,2]).all())



def suite():
    return unittest.makeSuite(SplitterTests)


if __name__ == '__main__':
    import test_runner

