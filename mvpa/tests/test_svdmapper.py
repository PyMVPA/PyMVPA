# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SVD mapper"""


import unittest
from mvpa.support.copy import deepcopy
import numpy as N
from mvpa.datasets import Dataset
from mvpa.mappers.svd import SVDMapper


class SVDMapperTests(unittest.TestCase):

    def setUp(self):
        # data: 40 sample feature line in 20d space (40x20; samples x features)
        self.ndlin = Dataset(samples=N.concatenate(
            [N.arange(40) for i in range(20)]).reshape(20,-1).T, labels=1, chunks=1)

        # data: 10 sample feature line in 40d space
        #       (10x40; samples x features)
        self.largefeat = Dataset(samples=N.concatenate(
            [N.arange(10) for i in range(40)]).reshape(40,-1).T, labels=1, chunks=1)


    def testSimpleSVD(self):
        pm = SVDMapper()
        # train SVD
        pm.train(self.ndlin)

        self.failUnlessEqual(pm.proj.shape, (20, 20))

        # now project data into PCA space
        p = pm.forward(self.ndlin.samples)

        # only first eigenvalue significant
        self.failUnless(pm.sv[:1] > 1.0)
        self.failUnless((pm.sv[1:] < 0.0001).all())

        # only variance of first component significant
        var = p.var(axis=0)

       # test that only one component has variance
        self.failUnless(var[:1] > 1.0)
        self.failUnless((var[1:] < 0.0001).all())

        # check that the mapped data can be fully recovered by 'reverse()'
        pr = pm.reverse(p)

        self.failUnlessEqual(pr.shape, (40,20))
        self.failUnless(N.abs(pm.reverse(p) - self.ndlin.samples).sum() < 0.0001)


    def testMoreSVD(self):
        pm = SVDMapper()
        # train SVD
        pm.train(self.largefeat)

        # mixing matrix cannot be square
        self.failUnlessEqual(pm.proj.shape, (40, 10))

        # only first singular value significant
        self.failUnless(pm.sv[:1] > 10)
        self.failUnless((pm.sv[1:] < 10).all())

        # now project data into SVD space
        p = pm.forward(self.largefeat.samples)

        # only variance of first component significant
        var = p.var(axis=0)

        # test that only one component has variance
        self.failUnless(var[:1] > 1.0)
        self.failUnless((var[1:] < 0.0001).all())

        # check that the mapped data can be fully recovered by 'reverse()'
        rp = pm.reverse(p)
        self.failUnlessEqual(rp.shape, self.largefeat.samples.shape)
        self.failUnless((N.round(rp) == self.largefeat.samples).all())

        self.failUnlessEqual(pm.getInSize(), 40)
        self.failUnlessEqual(pm.getOutSize(), 10)

        # copy mapper
        pm2 = deepcopy(pm)

        # now remove all but the first 2 components from the mapper
        pm2.selectOut([0,1])

        # sanity check
        self.failUnlessEqual(pm2.getInSize(), 40)
        self.failUnlessEqual(pm2.getOutSize(), 2)

        # but orginal mapper must be left intact
        self.failUnlessEqual(pm.getInSize(), 40)
        self.failUnlessEqual(pm.getOutSize(), 10)

        # data should still be fully recoverable by 'reverse()'
        rp2 = pm2.reverse(p[:,[0,1]])
        self.failUnlessEqual(rp2.shape, self.largefeat.samples.shape)
        self.failUnless(N.abs(rp2 - self.largefeat.samples).sum() < 0.0001)


        # now make new random data and do forward->reverse check
        data = N.random.normal(size=(98,40))
        data_f = pm.forward(data)

        self.failUnlessEqual(data_f.shape, (98,10))

        data_r = pm.reverse(data_f)
        self.failUnlessEqual(data_r.shape, (98,40))



def suite():
    return unittest.makeSuite(SVDMapperTests)


if __name__ == '__main__':
    import runner

