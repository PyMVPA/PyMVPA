# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA kernels"""

import unittest
import numpy as N
from mvpa.datasets import Dataset
from mvpa.clfs.distance import squared_euclidean_distance, \
     pnorm_w, pnorm_w_python

import mvpa.kernels.base as K
try:
    import mvpa.kernels.sg as SGK
    _has_sg = True
except RuntimeError:
    _has_sg = False

# from mvpa.clfs.kernel import Kernel

from tests_warehouse import datasets

class KernelTests(unittest.TestCase):
    """Test bloody kernels
    """

    # mvpa.kernel stuff
    def testLinearKernel(self):
        """Simplistic testing of linear kernel"""
        d1 = Dataset(N.asarray([range(5)]*10, dtype=float))
        lk = K.LinearKernel()
        lk.compute(d1)
        self.failUnless(lk._k.shape == (10, 10),
                        "Failure computing LinearKernel (Size mismatch)")
        self.failUnless((lk._k == 30).all(),
                        "Failure computing LinearKernel")

    def testPrecomputedKernel(self):
        """Statistic Kernels"""
        d = N.random.randn(50, 50)
        nk = K.PrecomputedKernel(matrix=d)
        nk.compute()
        self.failUnless((d == nk._k).all(),
                        'Failure setting and retrieving PrecomputedKernel data')

    if _has_sg:
        # Unit tests which require shogun kernels
        def testSgConversions(self):
            nk = K.PrecomputedKernel(matrix=N.random.randn(50, 50))
            nk.compute()
            sk = nk.as_sg()
            sk.compute()
            # There is some loss of accuracy here - why???
            self.failUnless((N.abs(nk._k - sk.as_np()._k) < 1e-6).all(),
                            'Failure converting arrays between NP as SG')
            
        def testLinearSG(self):
            d1 = N.random.randn(105, 32)
            d2 = N.random.randn(41, 32)
            
            nk = K.LinearKernel()
            sk = SGK.LinearSGKernel()
            nk.compute(d1, d2)
            sk.compute(d1,d2)
            
            self.failUnless(N.all(N.abs(nk._k - sk.as_np()._k)<1e-10),
                            'Numpy and SG linear kernels are inconsistent')
            
        def testRbfSG(self):
            d1 = N.random.randn(105, 32)
            d2 = N.random.randn(41, 32)
            sk = SGK.RbfSGKernel()
            gammavals = N.logspace(-2, 5, num=10)
            for g in gammavals:
                sk.params.gamma=g
                sk.compute(d1, d2)

    # Older kernel stuff (ie not mvpa.kernel) - perhaps refactor?
    def testEuclidDist(self):
        """Euclidean distance kernel testing"""

        # select some block of data from already generated
        data = datasets['uni4large'].samples[:5, :8]

        ed = squared_euclidean_distance(data)

        # XXX not sure if that is right: 'weight' seems to be given by
        # feature (i.e. column), but distance is between samples (i.e. rows)
        # current behavior is:
        true_size = (5, 5)
        self.failUnless(ed.shape == true_size)

        # slow version to compute distance matrix
        ed_manual = N.zeros(true_size, 'd')
        for i in range(true_size[0]):
            for j in range(true_size[1]):
                #ed_manual[i,j] = N.sqrt(((data[i,:] - data[j,:] )** 2).sum())
                ed_manual[i,j] = ((data[i,:] - data[j,:] )** 2).sum()
        ed_manual[ed_manual < 0] = 0

        self.failUnless(N.diag(ed_manual).sum() < 0.0000000001)
        self.failUnless(N.diag(ed).sum() < 0.0000000001)

        # let see whether Kernel does the same
        self.failUnless((ed - ed_manual).sum() < 0.0000001)


    def testPNorm_w(self):
        data0 = datasets['uni4large'].samples.T
        weight = N.abs(data0[11, :60])

        self.failUnlessRaises(ValueError, pnorm_w_python,
                              data0[:10,:2], p=1.2, heuristic='buga')
        self.failUnlessRaises(ValueError, pnorm_w_python,
                              data0[:10,:2], weight=weight)

        self.failUnlessRaises(ValueError, pnorm_w_python,
                              data0[:10,:2], data0[:10, :3],
                              weight=weight)
        self.failUnlessRaises(ValueError, pnorm_w,
                              data0[:10,:2], data0[:10, :3],
                              weight=weight)

        self.failUnlessRaises(ValueError, pnorm_w,
                              data0[:10,:2], weight=weight)

        # some sanity checks
        for did, (data1, data2, w) in enumerate(
            [ (data0[:2, :60], None, None),
              (data0[:2, :60], data0[3:4, 1:61], None),
              (data0[:2, :60], None, weight),
              (data0[:2, :60], data0[3:4, 1:61], weight),
              ]):
            # test different norms
            for p in [1, 2, 1.2]:
                kwargs = {'data1': data1,
                          'data2': data2,
                          'weight' : w,
                          'p' : p}
                d = pnorm_w(**kwargs)    # default one
                # to assess how far we are
                kwargs0 = kwargs.copy()
                kwargs0['data2'] = N.zeros(data1.shape)
                d0 = pnorm_w(**kwargs0)
                d0norm = N.linalg.norm(d - d0, 'fro')
                # test different implementations
                for iid, d2 in enumerate(
                    [pnorm_w_python(**kwargs),
                     pnorm_w_python(use_sq_euclidean=True, **kwargs),
                     pnorm_w_python(heuristic='auto', **kwargs),
                     pnorm_w_python(use_sq_euclidean=False, **kwargs),
                     pnorm_w_python(heuristic='auto', use_sq_euclidean=False, **kwargs),
                     pnorm_w_python(heuristic='samples', use_sq_euclidean=False, **kwargs),
                     pnorm_w_python(heuristic='features', use_sq_euclidean=False, **kwargs),
                     ]):
                    dnorm = N.linalg.norm(d2 - d, 'fro')
                    self.failUnless(dnorm/d0norm < 1e-7,
                        msg="Failed comparison of different implementations on "
                            "data #%d, implementation #%d, p=%s. "
                            "Norm of the difference is %g"
                            % (did, iid, p, dnorm))


def suite():
    return unittest.makeSuite(KernelTests)


if __name__ == '__main__':
    import runner

