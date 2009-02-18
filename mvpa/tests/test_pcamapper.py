# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA PCA mapper"""


import unittest
from mvpa.support.copy import deepcopy
import numpy as N
from mvpa.mappers.pca import PCAMapper

from mvpa.datasets import Dataset


class PCAMapperTests(unittest.TestCase):

    def setUp(self):
        # data: 40 sample feature line in 20d space (40x20; samples x features)
        self.ndlin = Dataset(samples=N.concatenate(
                        [N.arange(40) for i in range(20)]).reshape(20,-1).T, labels=1, chunks=1)

        # data: 10 sample feature line in 40d space
        #       (10x40; samples x features)
        self.largefeat = Dataset(samples=N.concatenate(
                        [N.arange(10) for i in range(40)]).reshape(40,-1).T, labels=1, chunks=1)

        self.pm = PCAMapper()


    def testSimplePCA(self):
        # train PCA
        self.pm.train(self.ndlin)

        self.failUnlessEqual(self.pm.mix.shape, (20, 20))

        # now project data into PCA space
        p = self.pm.forward(self.ndlin.samples)

        # only first eigenvalue significant
        self.failUnless(self.pm.sv[:1] > 1.0)
        self.failUnless((self.pm.sv[1:] < 0.0001).all())

        # only variance of first component significant
        var = p.var(axis=0)

        # test that only one component has variance
        self.failUnless(var[:1] > 1.0)
        self.failUnless((var[1:] < 0.0001).all())

        # check that the mapped data can be fully recovered by 'reverse()'
        self.failUnless((N.round(self.pm.reverse(p)) == self.ndlin.samples).all())


    def testAutoOptimizePCA(self):
        # train PCA
        self.pm.train(self.largefeat)

        # mixing matrix cannot be square
#        self.failUnlessEqual(self.pm.mix.shape, (10, 40))

        # only first eigenvalue significant
        self.failUnless(self.pm.sv[:1] > 10)
        self.failUnless((self.pm.sv[1:] < 10).all())

        # now project data into PCA space
        p = self.pm.forward(self.largefeat.samples)

        # only variance of first component significant
        var = p.var(axis=0)
        # test that only one component has variance
        self.failUnless(var[:1] > 1.0)
        self.failUnless((var[1:] < 0.0001).all())

        # check that the mapped data can be fully recovered by 'reverse()'
        rp = self.pm.reverse(p)
        self.failUnlessEqual(rp.shape, self.largefeat.samples.shape)
        self.failUnless((N.round(rp) == self.largefeat.samples).all())

        self.failUnlessEqual(self.pm.getInSize(), 40)
#        self.failUnlessEqual(self.pm.getOutSize(), 10)
        self.failUnlessEqual(self.pm.getOutSize(), 40)

        # copy mapper
        pm2 = deepcopy(self.pm)

        # now remove all but the first 2 components from the mapper
        pm2.selectOut([0,1])

        # sanity check
        self.failUnlessEqual(pm2.getInSize(), 40)
        self.failUnlessEqual(pm2.getOutSize(), 2)

        # but orginal mapper must be left intact
        self.failUnlessEqual(self.pm.getInSize(), 40)
#        self.failUnlessEqual(self.pm.getOutSize(), 10)
        self.failUnlessEqual(self.pm.getOutSize(), 40)

        # data should still be fully recoverable by 'reverse()'
        rp2 = pm2.reverse(p[:,[0,1]])
        self.failUnlessEqual(rp2.shape, self.largefeat.samples.shape)
        self.failUnless((N.round(rp2) == self.largefeat.samples).all())


def suite():
    return unittest.makeSuite(PCAMapperTests)


if __name__ == '__main__':
    import runner

