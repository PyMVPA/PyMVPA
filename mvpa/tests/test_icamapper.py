# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA ICA mapper"""


import unittest
from mvpa.support.copy import deepcopy
import numpy as N
from mvpa.mappers.ica import ICAMapper

from mvpa.datasets import Dataset

class ICAMapperTests(unittest.TestCase):

    def setUp(self):
        # data: 40 sample feature line in 2d space (40x2; samples x features)
        samples = N.vstack([N.arange(40.) for i in range(2)]).T
        samples -= samples.mean()
        samples +=  N.random.normal(size=samples.shape, scale=0.1)
        self.ndlin = Dataset(samples=samples, labels=1, chunks=1)

        # data: 40 sample feature line in 50d space (40x50; samples x features)
        samples = N.vstack([N.arange(40.) for i in range(50)]).T
        samples -= samples.mean()
        samples +=  N.random.normal(size=samples.shape, scale=0.1)
        self.largefeat = Dataset(samples=samples, labels=1, chunks=1)

        self.pm = ICAMapper()


    def testSimpleICA(self):
        # train
        self.pm.train(self.ndlin)

        self.failUnlessEqual(self.pm.proj.shape, (2, 2))

        # now project data into ICA space
        p = self.pm.forward(self.ndlin.samples)

        self.failUnlessEqual(p.shape, (40, 2))

        # check that the mapped data can be fully recovered by 'reverse()'
        self.failUnless(N.abs(self.pm.reverse(p) - self.ndlin.samples).mean() \
                        < 0.0001)


#    def testAutoOptimzeICA(self):
#        # train
#        self.pm.train(self.largefeat)
#
 #       self.failUnlessEqual(self.pm.proj.shape, (50, 40))
#
 #       # now project data into ICA space
 ##       p = self.pm.forward(self.largefeat.samples)
#
#        self.failUnless(p.shape[1] == 40)
#        print self.pm.proj
#        print self.pm.recon
#        print p

#        P.scatter(p[:20,0], p[:20,1],color='green')
#        P.scatter(p[20:,0], p[20:,1], color='red')
#        P.show()
#        self.failUnless(N.abs(self.pm.reverse(p) - self.largefeat.samples).mean() \
 #                       < 0.0001)

def suite():
    return unittest.makeSuite(ICAMapperTests)


if __name__ == '__main__':
    import runner

