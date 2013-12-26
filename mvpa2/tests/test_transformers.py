# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA transformers."""

import unittest
import numpy as np

from mvpa2.base import externals

from mvpa2.misc.transformers import Absolute, one_minus, rank_order, \
     reverse_rank_order, l1_normed, l2_normed, OverAxis, \
     DistPValue, first_axis_sum_not_zero

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets

from mvpa2.base import cfg

class TransformerTests(unittest.TestCase):

    def setUp(self):
        self.d1 = np.array([ 1,  0, -1, -2, -3])
        self.d2 = np.array([ 2.3,  0, -1, 2, -30, 1])

    @reseed_rng()
    def test_absolute(self):
        # generate 100 values (gaussian noise mean -1000 -> all negative)
        out = Absolute(np.random.normal(-1000, size=100))

        self.assertTrue(out.min() >= 0)
        self.assertTrue(len(out) == 100)

    def test_absolute2(self):
        target = self.d1
        out = one_minus(np.arange(5))
        self.assertTrue((out == target).all())

    def test_first_axis_sum_not_zero(self):
        src = [[ 1, -22.9, 6.8, 0],
               [ -.8, 7, 0, 0.0],
               [88, 0, 0.0, 0],
               [0, 0, 0, 0.0]]
        target = np.array([ 3, 2, 1, 0])
        out = first_axis_sum_not_zero(src)
        self.assertTrue((out == target).all())
        
    def test_rank_order(self):
        nelements = len(self.d2)
        out = rank_order(self.d2)
        outr = reverse_rank_order(self.d2)
        uout = np.unique(out)
        uoutr = np.unique(outr)
        self.assertTrue((uout == np.arange(nelements)).all(),
                        msg="We should get all indexes. Got just %s" % uout)
        self.assertTrue((uoutr == np.arange(nelements)).all(),
                        msg="We should get all indexes. Got just %s" % uoutr)
        self.assertTrue((out+outr+1 == nelements).all())
        self.assertTrue((out == [ 0,  3,  4,  1,  5,  2]).all())

    def test_l2_norm(self):
        out = l2_normed(self.d2)
        self.assertTrue(np.abs(np.sum(out*out)-1.0) < 1e-10)

    def test_l1_norm(self):
        out = l1_normed(self.d2)
        self.assertTrue(np.abs(np.sum(np.abs(out))-1.0) < 1e-10)


    def test_over_axis(self):
        data = datasets['uni4large'].samples[:120,0].reshape((2,3,4,5))
        # Simple transformer/combiner which collapses across given
        # dimension, e.g. sum
        for axis in [None, 0, 1, 2]:
            oversum = OverAxis(np.sum, axis=axis)(data)
            sum_ = np.sum(data, axis=axis)
            assert_array_almost_equal(sum_, oversum)

        # Transformer which doesn't modify dimensionality of the data
        data = data.reshape((6, -1))
        overnorm = OverAxis(l2_normed, axis=1)(data)
        self.assertTrue(np.linalg.norm(overnorm)!=1.0)
        for d in overnorm:
            self.assertTrue(np.abs(np.linalg.norm(d) - 1.0)<0.00001)

        overnorm = OverAxis(l2_normed, axis=0)(data)
        self.assertTrue(np.linalg.norm(overnorm)!=1.0)
        for d in overnorm.T:
            self.assertTrue(np.abs(np.linalg.norm(d) - 1.0)<0.00001)

    @reseed_rng()
    def test_dist_p_value(self):
        """Basic testing of DistPValue"""
        if not externals.exists('scipy'):
            return
        ndb = 200
        ndu = 20
        nperd = 2
        pthr = 0.05
        Nbins = 400

        # Lets generate already normed data (on sphere) and add some nonbogus features
        datau = (np.random.normal(size=(nperd, ndb)))
        dist = np.sqrt((datau * datau).sum(axis=1))

        datas = (datau.T / dist.T).T
        tn = datax = datas[0, :]
        dataxmax = np.max(np.abs(datax))

        # now lets add true positive features
        tp = [-dataxmax * 1.1] * (ndu//2) + [dataxmax * 1.1] * (ndu//2)
        x = np.hstack((datax, tp))

        # lets add just pure normal to it
        x = np.vstack((x, np.random.normal(size=x.shape))).T
        for distPValue in (DistPValue(), DistPValue(fpp=0.05)):
            result = distPValue(x)
            self.assertTrue((result>=0).all)
            self.assertTrue((result<=1).all)

        if cfg.getboolean('tests', 'labile', default='yes'):
            self.assertTrue(distPValue.ca.positives_recovered[0] > 10)
            self.assertTrue((np.array(distPValue.ca.positives_recovered) +
                             np.array(distPValue.ca.nulldist_number) == ndb + ndu).all())
            self.assertEqual(distPValue.ca.positives_recovered[1], 0)


def suite():  # pragma: no cover
    return unittest.makeSuite(TransformerTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

