#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA State parent class"""

import unittest

import numpy as N

from scipy import linalg
from mvpa.datasets.dataset import Dataset
from mvpa.misc.signal import detrend

class SignalTests(unittest.TestCase):

    def testDetrend(self):
        thr = 1e-10;                    # threshold for comparison
        samples = N.array( [[1.0, 2, 3, 3, 2, 1],
                            [-2.0, -4, -6, -6, -4, -2]], ndmin=2 ).T

        chunks = [0, 0, 0, 1, 1, 1]

        target_all = N.array( [[-1.0, 0, 1, 1, 0, -1],
                               [2, 0, -2, -2, 0, 2]], ndmin=2 ).T

        ds = Dataset(samples=samples, labels=chunks, chunks=chunks, copy_samples=True)
        detrend(ds, perchunk=False)

        self.failUnless(linalg.norm(ds.samples - target_all) < thr,
                        msg="Detrend should have detrended all the samples at once")

        ds = Dataset(samples=samples, labels=chunks, chunks=chunks, copy_samples=True)
        detrend(ds, perchunk=True)
        self.failUnless(linalg.norm(ds.samples) < thr,
                        msg="Detrend should have detrended each chunk separately")

        self.failUnless(ds.samples.shape == samples.shape,
                        msg="Detrend must preserve the size of dataset")


        # small additional test for break points
        ds = Dataset(samples=N.array([[1.0, 2, 3, 1, 2, 3]], ndmin=2).T,
                     labels=chunks, chunks=chunks, copy_samples=True)
        detrend(ds, perchunk=True)
        self.failUnless(linalg.norm(ds.samples) < thr,
                        msg="Detrend should have removed all the signal")

def suite():
    return unittest.makeSuite(SignalTests)


if __name__ == '__main__':
    import test_runner

