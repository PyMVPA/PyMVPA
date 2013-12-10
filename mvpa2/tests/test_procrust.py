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
import numpy as np
from numpy.linalg import norm
from mvpa2.base import externals
from mvpa2.datasets.base import dataset_wizard
from mvpa2.testing import *
from mvpa2.testing.datasets import *
from mvpa2.mappers.procrustean import ProcrusteanMapper

svds = ['numpy']
if externals.exists('liblapack.so'):
    svds += ['dgesvd']
if externals.exists('scipy'):
    svds += ['scipy']

class ProcrusteanMapperTests(unittest.TestCase):

    @sweepargs(oblique=(False,True))
    @sweepargs(svd=svds)
    @reseed_rng()
    def test_simple(self, svd, oblique):
        d_orig = datasets['uni2large'].samples
        d_orig2 = datasets['uni4large'].samples
        for sdim, nf_s, nf_t, full_test \
                in (('Same 2D',  2,  2,  True),
                    ('Same 10D', 10, 10, True),
                    ('2D -> 3D', 2,  3,  True),
                    ('3D -> 2D', 3,  2,  False)):
            # figure out some "random" rotation
            d = max(nf_s, nf_t)
            R = get_random_rotation(nf_s, nf_t, d_orig)
            if nf_s == nf_t:
                adR = np.abs(1.0 - np.linalg.det(R))
                self.assertTrue(adR < 1e-10,
                                "Determinant of rotation matrix should "
                                "be 1. Got it 1+%g" % adR)
                self.assertTrue(norm(np.dot(R, R.T)
                                     - np.eye(R.shape[0])) < 1e-10)

            for s, scaling in ((0.3, True), (1.0, False)):
                pm = ProcrusteanMapper(scaling=scaling, oblique=oblique, svd=svd)
                # pm2 = ProcrusteanMapper(scaling=scaling, oblique=oblique)

                t1, t2 = d_orig[23, 1], d_orig[22, 1]

                # Create source/target data
                d = d_orig[:, :nf_s]
                d_s = d + t1
                d_t = np.dot(s * d, R) + t2

                # train bloody mapper(s)
                ds = dataset_wizard(samples=d_s, targets=d_t)
                pm.train(ds)
                ## not possible with new interface
                #pm2.train(d_s, d_t)

                ## verify that both created the same transformation
                #npm2proj = norm(pm.proj - pm2.proj)
                #self.assertTrue(npm2proj <= 1e-10,
                #                msg="Got transformation different by norm %g."
                #                " Had to be less than 1e-10" % npm2proj)
                #self.assertTrue(norm(pm._offset_in - pm2._offset_in) <= 1e-10)
                #self.assertTrue(norm(pm._offset_out - pm2._offset_out) <= 1e-10)

                # do forward transformation on the same source data
                d_s_f = pm.forward(d_s)

                self.assertEqual(d_s_f.shape, d_t.shape,
                    msg="Mapped shape should be identical to the d_t")

                dsf = d_s_f - d_t
                ndsf = norm(dsf)/norm(d_t)
                if full_test:
                    dsR = norm(s*R - pm.proj)

                    if not oblique:
                        self.assertTrue(dsR <= 1e-12,
                            msg="We should have got reconstructed rotation+scaling "
                                "perfectly. Now got d scale*R=%g" % dsR)

                        self.assertTrue(np.abs(s - pm._scale) < 1e-12,
                            msg="We should have got reconstructed scale "
                                "perfectly. Now got %g for %g" % (pm._scale, s))

                    self.assertTrue(ndsf <= 1e-12,
                      msg="%s: Failed to get to the target space correctly."
                        " normed error=%g" % (sdim, ndsf))

                # Test if we get back
                d_s_f_r = pm.reverse(d_s_f)

                dsfr = d_s_f_r - d_s
                ndsfr = norm(dsfr)/norm(d_s)
                if full_test:
                    self.assertTrue(ndsfr <= 1e-12,
                      msg="%s: Failed to reconstruct into source space correctly."
                        " normed error=%g" % (sdim, ndsfr))



def suite():  # pragma: no cover
    return unittest.makeSuite(ProcrusteanMapperTests)

if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

