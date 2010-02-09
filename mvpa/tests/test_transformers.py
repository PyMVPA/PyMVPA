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
import numpy as N

from mvpa.base import externals

from mvpa.misc.transformers import Absolute, one_minus, rank_order, \
     reverse_rank_order, l1_normed, l2_normed, OverAxis, \
     DistPValue, first_axis_sum_not_zero

from tests_warehouse import sweepargs, datasets

from mvpa.base import cfg

class TransformerTests(unittest.TestCase):

    def setUp(self):
        self.d1 = N.array([ 1,  0, -1, -2, -3])
        self.d2 = N.array([ 2.3,  0, -1, 2, -30, 1])

    def test_absolute(self):
        # generate 100 values (gaussian noise mean -1000 -> all negative)
        out = Absolute(N.random.normal(-1000, size=100))

        self.failUnless(out.min() >= 0)
        self.failUnless(len(out) == 100)

    def test_absolute2(self):
        target = self.d1
        out = one_minus(N.arange(5))
        self.failUnless((out == target).all())

    def test_first_axis_sum_not_zero(self):
        src = [[ 1, -22.9, 6.8, 0],
               [ -.8, 7, 0, 0.0],
               [88, 0, 0.0, 0],
               [0, 0, 0, 0.0]]
        target = N.array([ 3, 2, 1, 0])
        out = first_axis_sum_not_zero(src)
        self.failUnless((out == target).all())
        
    def test_rank_order(self):
        nelements = len(self.d2)
        out = rank_order(self.d2)
        outr = reverse_rank_order(self.d2)
        uout = N.unique(out)
        uoutr = N.unique(outr)
        self.failUnless((uout == N.arange(nelements)).all(),
                        msg="We should get all indexes. Got just %s" % uout)
        self.failUnless((uoutr == N.arange(nelements)).all(),
                        msg="We should get all indexes. Got just %s" % uoutr)
        self.failUnless((out+outr+1 == nelements).all())
        self.failUnless((out == [ 0,  3,  4,  1,  5,  2]).all())

    def test_l2_norm(self):
        out = l2_normed(self.d2)
        self.failUnless(N.abs(N.sum(out*out)-1.0) < 1e-10)

    def test_l1_norm(self):
        out = l1_normed(self.d2)
        self.failUnless(N.abs(N.sum(N.abs(out))-1.0) < 1e-10)


    def test_over_axis(self):
        data = datasets['uni4large'].samples[:120,0].reshape((2,3,4,5))
        # Simple transformer/combiner which collapses across given
        # dimension, e.g. sum
        for axis in [None, 0, 1, 2]:
            oversum = OverAxis(N.sum, axis=axis)(data)
            sum_ = N.sum(data, axis=axis)
            self.failUnless(N.all(sum_ == oversum))

        # Transformer which doesn't modify dimensionality of the data
        data = data.reshape((6, -1))
        overnorm = OverAxis(l2_normed, axis=1)(data)
        self.failUnless(N.linalg.norm(overnorm)!=1.0)
        for d in overnorm:
            self.failUnless(N.abs(N.linalg.norm(d) - 1.0)<0.00001)

        overnorm = OverAxis(l2_normed, axis=0)(data)
        self.failUnless(N.linalg.norm(overnorm)!=1.0)
        for d in overnorm.T:
            self.failUnless(N.abs(N.linalg.norm(d) - 1.0)<0.00001)


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
        datau = (N.random.normal(size=(nperd, ndb)))
        dist = N.sqrt((datau * datau).sum(axis=1))

        datas = (datau.T / dist.T).T
        tn = datax = datas[0, :]
        dataxmax = N.max(N.abs(datax))

        # now lets add true positive features
        tp = [-dataxmax * 1.1] * (ndu/2) + [dataxmax * 1.1] * (ndu/2)
        x = N.hstack((datax, tp))

        # lets add just pure normal to it
        x = N.vstack((x, N.random.normal(size=x.shape))).T
        for distPValue in (DistPValue(), DistPValue(fpp=0.05)):
            result = distPValue(x)
            self.failUnless((result>=0).all)
            self.failUnless((result<=1).all)

        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless(distPValue.ca.positives_recovered[0] > 10)
            self.failUnless((N.array(distPValue.ca.positives_recovered) +
                             N.array(distPValue.ca.nulldist_number) == ndb + ndu).all())
            self.failUnless(distPValue.ca.positives_recovered[1] == 0)


def suite():
    return unittest.makeSuite(TransformerTests)


if __name__ == '__main__':
    import runner

