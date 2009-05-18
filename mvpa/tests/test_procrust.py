# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Procrustean mapper"""


import unittest
import numpy as N
from numpy.linalg import norm
from mvpa.datasets import Dataset
from tests_warehouse import datasets
from mvpa.mappers.procrustean import ProcrusteanMapper


class ProcrusteanMapperTests(unittest.TestCase):

    def testSimple(self):
        d_orig = datasets['uni2large'].samples
        d_orig2 = datasets['uni4large'].samples
        for sdim, nf_s, nf_t, full_test \
                in (('Same 2D',  2,  2,  True),
                    ('Same 10D', 10, 10, True),
                    ('2D -> 3D', 2,  3,  True),
                    ('3D -> 2D', 3,  2,  False)):

            # lets do evil -- use the mapper itself on some data to
            # figure out some "random" rotation matrix for us to use ;)
            pm_orig = ProcrusteanMapper(reflection=False)
            d = max(nf_s, nf_t)
            pm_orig.train(d_orig[:50, :d], d_orig2[10:60, :d])
            R = pm_orig._T[:nf_s, :nf_t].copy()

            if nf_s == nf_t:
                # Test if it is indeed a rotation matrix ;)
                self.failUnless(N.abs(1.0 - N.linalg.det(R)) < 1e-10)
                self.failUnless(norm(N.dot(R, R.T)
                                              - N.eye(R.shape[0])) < 1e-10)

            for s, scaling in ((0.3, True), (1.0, False)):
                pm = ProcrusteanMapper(scaling=scaling)
                pm2 = ProcrusteanMapper(scaling=scaling)

                t1, t2 = d_orig[23, 1], d_orig[22, 1]

                # Create source/target data
                d = d_orig[:, :nf_s]
                d_s = d + t1
                d_t = N.dot(s * d, R) + t2

                # train bloody mapper(s)
                pm.train(d_s, d_t)
                ds2 = Dataset(samples=d_s, labels=d_t)
                pm2.train(ds2)

                # verify that both created the same transformation
                self.failUnless(norm(pm._T - pm2._T) <= 1e-12)
                self.failUnless(norm(pm._trans - pm2._trans) <= 1e-12)

                # do forward transformation on the same source data
                d_s_f = pm.forward(d_s)

                self.failUnlessEqual(d_s_f.shape, d_t.shape,
                    msg="Mapped shape should be identical to the d_t")

                dsf = d_s_f - d_t
                ndsf = norm(dsf)/norm(d_t)
                if full_test:
                    dR = norm(R - pm._T)
                    self.failUnless(dR <= 1e-12,
                        msg="We should have got reconstructed rotation "
                            "perfectly. Now got dR=%g" % dR)

                    self.failUnless(N.abs(s - pm._scale) < 1e-12,
                        msg="We should have got reconstructed scale "
                            "perfectly. Now got %g for %g" % (pm._scale, s))

                    self.failUnless(ndsf <= 1e-12,
                      msg="%s: Failed to get to the target space correctly."
                        " normed error=%g" % (sdim, ndsf))

                # Test if we get back
                d_s_f_r = pm.reverse(d_s_f)

                dsfr = d_s_f_r - d_s
                ndsfr = norm(dsfr)/norm(d_s)
                if full_test:
                    self.failUnless(ndsfr <= 1e-12,
                      msg="%s: Failed to reconstruct into source space correctly."
                        " normed error=%g" % (sdim, ndsfr))



def suite():
    return unittest.makeSuite(ProcrusteanMapperTests)

if __name__ == '__main__':
    import runner

