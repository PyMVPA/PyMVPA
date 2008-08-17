#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA miscelaneouse functions operating on datasets"""

import unittest

import numpy as N

from mvpa.base import externals
from mvpa.datasets import Dataset
from mvpa.datasets.miscfx import removeInvariantFeatures, coarsenChunks

class MiscDatasetFxTests(unittest.TestCase):

    def testInvarFeaturesRemoval(self):
        r = N.random.normal(size=(3,1))
        ds = Dataset(samples=N.hstack((N.zeros((3,2)), r)),
                     labels=1)

        self.failUnless(ds.nfeatures == 3)

        dsc = removeInvariantFeatures(ds)

        self.failUnless(dsc.nfeatures == 1)
        self.failUnless((dsc.samples == r).all())


    def testCoarsenChunks(self):
        """Just basic testing for now"""
        chunks = [1,1,2,2,3,3,4,4]
        ds = Dataset(samples=N.arange(len(chunks)).reshape(
            (len(chunks),1)), labels=[1]*8, chunks=chunks)
        coarsenChunks(ds, nchunks=2)
        chunks1 = coarsenChunks(chunks, nchunks=2)
        self.failUnless((chunks1 == ds.chunks).all())
        self.failUnless((chunks1 == N.asarray([0,0,0,0,1,1,1,1])).all())

        ds2 = Dataset(samples=N.arange(len(chunks)).reshape(
            (len(chunks),1)), labels=[1]*8)
        coarsenChunks(ds2, nchunks=2)
        self.failUnless((chunks1 == ds.chunks).all())


def suite():
    return unittest.makeSuite(MiscDatasetFxTests)


if __name__ == '__main__':
    import runner

