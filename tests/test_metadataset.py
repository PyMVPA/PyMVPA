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
from mvpa.misc.copy import deepcopy
from mvpa.base import externals
from mvpa.datasets import Dataset
from mvpa.datasets.meta import MetaDataset
from mvpa.datasets.eep import EEPDataset
from mvpa.mappers.base import CombinedMapper, ChainMapper
from mvpa.mappers.array import DenseArrayMapper
from mvpa.mappers.mask import MaskMapper
from mvpa.mappers.boxcar import BoxcarMapper

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
        if not externals.exists('nifti'):
            return
        from mvpa.datasets.nifti import NiftiDataset
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


    def testCombinedMapper(self):
        # simple case: two array of different shape combined
        m = CombinedMapper([DenseArrayMapper(mask=N.ones((2,3,4))),
                            MaskMapper(mask=N.array((1,1)))])

        self.failUnless(m.getInSize() == 26)
        self.failUnless(m.getOutSize() == 26)
        self.failUnless(m.getInShape() == ((2,3,4), (2,)))
        self.failUnless(m.getOutShape() == (26,))

        d1 = N.ones((5,2,3,4))
        d2_broken = N.ones((6,2)) + 1
        d2 = N.ones((5,2)) + 1

        # should not work for sample mismatch
        self.failUnlessRaises(ValueError, m.forward, (d1, d2_broken))

        # check forward mapping (size and identity)
        mf = m.forward((d1, d2))
        self.failUnless(mf.shape == (5, 26))
        self.failUnless((mf[:,:24] == 1).all())
        self.failUnless((mf[:,-2:] == 2).all())

        # check reverse mapping
        self.failUnlessRaises(ValueError, m.reverse, N.arange(12))
        mr = m.reverse(N.arange(26) + 1)
        self.failUnless(len(mr) == 2)
        self.failUnless((mr[0] == N.arange(24).reshape((2,3,4)) + 1).all())
        self.failUnless((mr[1] == N.array((25,26))).all())

        # check reverse mapping of multiple samples
        mr = m.reverse(N.array([N.arange(26) + 1 for i in range(4)]))
        self.failUnless(len(mr) == 2)
        self.failUnless(
            (mr[0] == N.array([N.arange(24).reshape((2,3,4)) + 1
                                    for i in range(4)])).all())
        self.failUnless(
            (mr[1] == N.array([N.array((25,26)) for i in range(4)])).all())


        # check dummy train
        m.train(Dataset(samples=N.random.rand(10,26), labels=range(10)))
        self.failUnlessRaises(ValueError, m.train,
            Dataset(samples=N.random.rand(10,25), labels=range(10)))

        # check neighbor information
        # fail if invalid id
        self.failUnlessRaises(ValueError, m.getNeighbor, 26)
        # neighbors for last feature of first mapper, ie.
        # close in out space but infinite/undefined distance in in-space
        self.failUnless([n for n in m.getNeighbor(23, radius=2)]
                        == [6, 7, 10, 11, 15, 18, 19, 21, 22, 23])

        # check feature selection
        m.selectOut((23,25))
        self.failUnless(m.getInSize() == 26)
        self.failUnless(m.getOutSize() == 2)
        self.failUnless(m.getInShape() == ((2,3,4), (2,)))
        self.failUnless(m.getOutShape() == (2,))

        # check reverse mapping of truncated mapper
        mr = m.reverse(N.array((99,88)))
        target1 = N.zeros((2,3,4))
        target1[1,2,3] = 99
        target2 = N.array((0, 88))
        self.failUnless(len(mr) == 2)
        self.failUnless((mr[0] == target1).all())
        self.failUnless((mr[1] == target2).all())

        # check forward mapping
        self.failUnless((m.forward((d1, d2))[0] == (1, 2)).all())

        # check copying
        mc = deepcopy(m)
        mc.selectOut([1])
        self.failUnless(m.getOutSize() == 2)
        self.failUnless(mc.getOutSize() == 1)


    def testChainMapper(self):
        data = N.array([N.arange(24).reshape(3,4,2) + (i * 100)
                            for i in range(10)])

        startpoints = [ 2, 4, 3, 5 ]
        m = ChainMapper([BoxcarMapper(startpoints, 2),
                         DenseArrayMapper(mask=N.ones((2, 3, 4, 2)))])
        mp = m.forward(data)
        # 4 startpoint, with each two samples of shape (3,4,2)
        self.failUnless(mp.shape == (4, 48))

        self.failUnless(m.reverse(N.arange(48)).shape == (2, 3, 4, 2))

        # should behave a DenseArrayMapper alone
        self.failUnless((N.array([n for n in m.getNeighbor(24, radius=1.1)])
                         == N.array((0, 24, 25, 26, 32))).all())


def suite():
    return unittest.makeSuite(MetaDatasetTests)


if __name__ == '__main__':
    import runner

