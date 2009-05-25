# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA miscelaneouse functions operating on datasets and requiring SciPy"""

import unittest

import numpy as N

from mvpa.base import externals

from mvpa.datasets import Dataset
from mvpa.datasets.miscfx import removeInvariantFeatures

if externals.exists('scipy', raiseException=True):
    from scipy import linalg
from mvpa.datasets.miscfx import detrend

class MiscDatasetFxSpTests(unittest.TestCase):

    def testDetrend(self):
        thr = 1e-10;                    # threshold for comparison
        samples = N.array( [[1.0, 2, 3, 3, 2, 1],
                            [-2.0, -4, -6, -6, -4, -2]], ndmin=2 ).T

        chunks = [0, 0, 0, 1, 1, 1]
        chunks_bad = [ 0, 0, 1, 1, 1, 0]
        target_all = N.array( [[-1.0, 0, 1, 1, 0, -1],
                               [2, 0, -2, -2, 0, 2]], ndmin=2 ).T


        ds = Dataset(samples=samples, labels=chunks, chunks=chunks,
                     copy_samples=True)
        detrend(ds, perchunk=False)

        self.failUnless(linalg.norm(ds.samples - target_all) < thr,
                msg="Detrend should have detrended all the samples at once")


        ds_bad = Dataset(samples=samples, labels=chunks, chunks=chunks_bad,
                         copy_samples=True)
        self.failUnlessRaises(ValueError, detrend, ds_bad, perchunk=True)


        ds = Dataset(samples=samples, labels=chunks, chunks=chunks,
                     copy_samples=True)
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

        # tests of the regress version of detrend
        ds = Dataset(samples=samples, labels=chunks, chunks=chunks,
                     copy_samples=True)
        detrend(ds, perchunk=False, model='regress', polyord=1)
        self.failUnless(linalg.norm(ds.samples - target_all) < thr,
                msg="Detrend should have detrended all the samples at once")

        ds = Dataset(samples=samples, labels=chunks, chunks=chunks,
                     copy_samples=True)
        (res, reg) = detrend(ds, perchunk=True, model='regress', polyord=2)
        psamps = ds.samples.copy()
        self.failUnless(linalg.norm(ds.samples) < thr,
                msg="Detrend should have detrended each chunk separately")

        self.failUnless(ds.samples.shape == samples.shape,
                        msg="Detrend must preserve the size of dataset")

        ods = Dataset(samples=samples, labels=chunks, chunks=chunks,
                      copy_samples=True)
        opt_reg = reg.copy()
        (ores, oreg) = detrend(ods, perchunk=True, model='regress',
                               opt_reg=opt_reg)
        dsamples = (ods.samples - psamps).sum()
        self.failUnless(abs(dsamples) <= 1e-10,
            msg="Detrend for polyord reg should be same as opt_reg " + \
                "when popt_reg is the same as the polyord reg. But got %g" \
                % dsamples)

        self.failUnless(linalg.norm(ds.samples) < thr,
                msg="Detrend should have detrended each chunk separately")


        # test of different polyord on each chunk
        target_mixed = N.array( [[-1.0, 0, 1, 0, 0, 0],
                                 [2.0, 0, -2, 0, 0, 0]], ndmin=2 ).T

        ds = Dataset(samples=samples, labels=chunks, chunks=chunks,
                     copy_samples=True)
        (res, reg) = detrend(ds, perchunk=True, model='regress', polyord=[0,1])
        self.failUnless(linalg.norm(ds.samples - target_mixed) < thr,
            msg="Detrend should have baseline corrected the first chunk, " + \
                "but baseline and linear detrended the second.")

        # test applying detrend in sequence
        ds = Dataset(samples=samples, labels=chunks, chunks=chunks,
                     copy_samples=True)
        (res, reg) = detrend(ds, perchunk=True, model='regress', polyord=1)
        opt_reg = reg[N.ix_(range(reg.shape[0]),[1,3])]
        final_samps = ds.samples.copy()
        ds = Dataset(samples=samples, labels=chunks, chunks=chunks,
                     copy_samples=True)
        (res, reg) = detrend(ds, perchunk=True, model='regress', polyord=0)
        (res, reg) = detrend(ds, perchunk=True, model='regress',
                             opt_reg=opt_reg)
        self.failUnless(linalg.norm(ds.samples - final_samps) < thr,
                msg="Detrend of polyord 1 should be same as detrend with " + \
                    "0 followed by opt_reg the same as a 1st order.")



def suite():
    return unittest.makeSuite(MiscDatasetFxSpTests)


if __name__ == '__main__':
    import runner

