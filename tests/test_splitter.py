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
from mvpa.datasets.splitter import NFoldSplitter, OddEvenSplitter
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

        print splits



def suite():
    return unittest.makeSuite(SplitterTests)


if __name__ == '__main__':
    import test_runner

