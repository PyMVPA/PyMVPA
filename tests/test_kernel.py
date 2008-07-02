#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA kernels"""


from mvpa.clfs.kernel import Kernel
import unittest
import numpy as N

class KernelTests(unittest.TestCase):

    def testEuclidDist(self):
        k = Kernel()

        data = N.random.normal(size=(20,30))
        ed = k.euclidean_distance(data)

        # XXX not sure if that is right: 'weight' seems to be given by
        # feature (i.e. column), but distance is between samples (i.e. rows)
        # current behavior is:
        true_size = (20, 20)
        self.failUnless(ed.shape == true_size)

        # slow version to compute distance matrix
        ed_manual = N.zeros(true_size, 'd')
        for i in range(true_size[0]):
            for j in range(true_size[1]):
                ed_manual[i,j] = ((data[i,:] - data[j,:] )** 2).sum()
        ed_manual[ed_manual < 0] = 0

        self.failUnless(N.diag(ed_manual).sum() < 0.0000000001)
        self.failUnless(N.diag(ed).sum() < 0.0000000001)

        # let see whether Kernel does the same
        self.failUnless((ed_manual == ed).all())




def suite():
    return unittest.makeSuite(KernelTests)


if __name__ == '__main__':
    import runner

