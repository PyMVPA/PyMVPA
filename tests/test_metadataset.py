#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA meta dataset handling"""

import unittest
import numpy as N
import os.path
from mvpa.datasets import Dataset
from mvpa.datasets.meta import MetaDataset
from mvpa.datasets.niftidataset import NiftiDataset
from mvpa.datasets.eep import EEPDataset

class MetaDatasetTests(unittest.TestCase):

    def testSimple(self):
        # bunch of datasets

        datasets = [
                     Dataset(samples=N.arange(12).reshape((4,3)), labels=1),
                     Dataset(samples=N.zeros((4,4)), labels=1),
                     Dataset(samples=N.ones((4,2), dtype='float'), labels=1),
                   ]

        mds = MetaDataset(datasets)

        # all together
        self.failUnless(mds.samples.shape == (4, 9))
        # should do upcasting
        self.failUnless(mds.samples.dtype == 'float')
        # simple samples attrs
        self.failUnless((mds.labels == [1] * 4).all())
        self.failUnless((mds.chunks == range(4)).all())

        # do sample selection across all datasets
        mds1 = mds.selectSamples([0,3])
        self.failUnless(mds1.samples.shape == (2, 9))
        self.failUnless(\
            (mds1.samples[0] == [0, 1, 2, 0, 0, 0, 0, 1, 1]).all())
        self.failUnless(\
            (mds1.samples[1] == [9, 10, 11, 0, 0, 0, 0, 1, 1]).all())

        # more tricky feature selection on all datasets
        mds2 = mds.selectFeatures([1,4,8])

        self.failUnless(\
            (mds2.samples == [[ 1, 0, 1],
                              [ 4, 0, 1],
                              [ 7, 0, 1],
                              [10, 0, 1]] ).all())


    def testMapping(self):
        eeds = EEPDataset(os.path.join('..', 'data', 'eep.bin'), labels=[1,2])
        nids = NiftiDataset(os.path.join('..', 'data', 'example4d.nii.gz'),
                            labels=[1,2])
        plainds = Dataset(samples=N.arange(8).reshape((2,4)), labels=[1,2])

        datasets = (eeds, plainds, nids)

        mds = MetaDataset(datasets)

        self.failUnless(mds.nfeatures == N.sum([d.nfeatures for d in datasets]))
        self.failUnless(mds.nsamples == 2)

        # try reverse mapping
        mr = mds.mapReverse(N.arange(mds.nfeatures))

        self.failUnless(len(mr) == 3)
        self.failUnless(mr[0].shape == eeds.mapper.getInShape())
        self.failUnless(mr[1].shape == (plainds.nfeatures,))
        self.failUnless(mr[2].shape == nids.mapper.getInShape())



def suite():
    return unittest.makeSuite(MetaDatasetTests)


if __name__ == '__main__':
    import runner

