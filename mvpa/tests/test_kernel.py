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

from mvpa.clfs.distance import squared_euclidean_distance, \
     pnorm_w, pnorm_w_python
# from mvpa.clfs.kernel import Kernel

from tests_warehouse import datasets

class KernelTests(unittest.TestCase):

    def testEuclidDist(self):

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

