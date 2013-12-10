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
import numpy as np

from mvpa2.mappers.svd import SVDMapper
from mvpa2.testing import reseed_rng
from mvpa2.support.copy import deepcopy


class SVDMapperTests(unittest.TestCase):

    def setUp(self):
        # data: 40 sample feature line in 20d space (40x20; samples x features)
        self.ndlin = np.concatenate([np.arange(40)
                                        for i in range(20)]).reshape(20,-1).T

        # data: 10 sample feature line in 40d space
        #       (10x40; samples x features)
        self.largefeat = np.concatenate([np.arange(10)
                                        for i in range(40)]).reshape(40,-1).T


    def test_simple_svd(self):
        pm = SVDMapper()
        # train SVD
        pm.train(self.ndlin)

        self.assertEqual(pm.proj.shape, (20, 20))

        # now project data into PCA space
        p = pm.forward(self.ndlin)

        # only first eigenvalue significant
        self.assertTrue(pm.sv[:1] > 1.0)
        self.assertTrue((pm.sv[1:] < 0.0001).all())

        # only variance of first component significant
        var = p.var(axis=0)

       # test that only one component has variance
        self.assertTrue(var[:1] > 1.0)
        self.assertTrue((var[1:] < 0.0001).all())

        # check that the mapped data can be fully recovered by 'reverse()'
        pr = pm.reverse(p)

        self.assertEqual(pr.shape, (40,20))
        self.assertTrue(np.abs(pm.reverse(p) - self.ndlin).sum() < 0.0001)


    @reseed_rng()
    def test_more_svd(self):
        pm = SVDMapper()
        # train SVD
        pm.train(self.largefeat)

        # mixing matrix cannot be square
        self.assertEqual(pm.proj.shape, (40, 10))

        # only first singular value significant
        self.assertTrue(pm.sv[:1] > 10)
        self.assertTrue((pm.sv[1:] < 10).all())

        # now project data into SVD space
        p = pm.forward(self.largefeat)

        # only variance of first component significant
        var = p.var(axis=0)

        # test that only one component has variance
        self.assertTrue(var[:1] > 1.0)
        self.assertTrue((var[1:] < 0.0001).all())

        # check that the mapped data can be fully recovered by 'reverse()'
        rp = pm.reverse(p)
        self.assertEqual(rp.shape, self.largefeat.shape)
        self.assertTrue((np.round(rp) == self.largefeat).all())

        # copy mapper
        pm2 = deepcopy(pm)

        # now make new random data and do forward->reverse check
        data = np.random.normal(size=(98,40))
        data_f = pm.forward(data)

        self.assertEqual(data_f.shape, (98,10))

        data_r = pm.reverse(data_f)
        self.assertEqual(data_r.shape, (98,40))



def suite():  # pragma: no cover
    return unittest.makeSuite(SVDMapperTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

