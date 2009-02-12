# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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
from mvpa import pymvpa_dataroot
from mvpa.support.copy import deepcopy
from mvpa.base import externals
from mvpa.datasets import Dataset
from mvpa.datasets.meta import MetaDataset
from mvpa.datasets.eep import EEPDataset
from mvpa.mappers.base import CombinedMapper, ChainMapper
from mvpa.mappers.array import DenseArrayMapper
from mvpa.mappers.mask import MaskMapper
from mvpa.mappers.boxcar import BoxcarMapper
from mvpa.datasets.event import EventDataset
from mvpa.misc.support import Event
from mvpa.misc.exceptions import DatasetError


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
        eeds = EEPDataset(os.path.join(pymvpa_dataroot, 'eep.bin'), labels=[1,2])
        nids = NiftiDataset(os.path.join(pymvpa_dataroot, 'example4d.nii.gz'),
                            labels=[1,2])
        plainds = Dataset(samples=N.arange(8).reshape((2,4)), labels=[1,2])

        datasets = (eeds, plainds, nids)

        mds = MetaDataset(datasets)

        self.failUnless(mds.nfeatures == N.sum([d.nfeatures for d in datasets]))
        self.failUnless(mds.nsamples == 2)

        # try reverse mapping
        mr = mds.mapReverse(N.arange(mds.nfeatures))

        self.failUnless(len(mr) == 3)
        self.failUnless(mr[1].shape == (plainds.nfeatures,))


    def testCombinedMapper(self):
        # simple case: two array of different shape combined
        m = CombinedMapper([DenseArrayMapper(mask=N.ones((2,3,4))),
                            MaskMapper(mask=N.array((1,1)))])

        self.failUnless(m.getInSize() == 26)
        self.failUnless(m.getOutSize() == 26)

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


    def testEventDataset(self):
        # baisc checks
        self.failUnlessRaises(DatasetError, EventDataset)

        # simple data
        samples = N.arange(240).reshape(10, 2, 3, 4)

        # copy constructor does not work on non-2D data
        self.failUnlessRaises(DatasetError, EventDataset, samples=samples)

        # try case without extra features
        evs = [Event(onset=2, duration=2, label=1, chunk=2),
               Event(onset=5, duration=1, label=2, chunk=2),
               Event(onset=7, duration=2, label=3, chunk=4)]

        ds = EventDataset(samples=samples, events=evs)
        self.failUnless(ds.nfeatures == 48)
        self.failUnless(ds.nsamples == 3)
        self.failUnless((ds.labels == [1,2,3]).all())
        self.failUnless((ds.chunks == [2,2,4]).all())
        mr = ds.mapReverse(N.arange(48))
        self.failUnless((mr == N.arange(48).reshape(2,2,3,4)).all())

        # try case with extra features
        evs = [Event(onset=2, duration=2, label=1, features=[2,3]),
               Event(onset=5, duration=2, label=1, features=[4,5]),
               Event(onset=7, duration=2, label=1, features=[6,7]),]
        ds = EventDataset(samples=samples, events=evs)
        # we have 2 additional features
        self.failUnless(ds.nfeatures == 50)
        self.failUnless(ds.nsamples == 3)
        self.failUnless((ds.labels == [1,1,1]).all())
        self.failUnless((ds.chunks == [0,1,2]).all())
        # now for the long awaited -- map back into two distinct
        # feature spaces
        mr = ds.mapReverse(N.arange(50))
        # we get two sets of feature spaces (samples and extra features)
        self.failUnless(len(mr) == 2)
        msamples, mxfeat = mr
        # the sample side should be identical to the case without extra features
        self.failUnless((msamples == N.arange(48).reshape(2,2,3,4)).all())
        # the extra features should be flat
        self.failUnless((mxfeat == (48,49)).all())

        # now take a look at 
        orig = ds.O
        self.failUnless(len(mr) == 2)
        osamples, oextra = orig
        self.failUnless((oextra == [[2,3],[4,5],[6,7]]).all())
        self.failUnless(osamples.shape == samples.shape)
        # check that all samples not covered by an event are zero
        filt = N.array([True,True,False,False,True,False,False,False,False,True])
        self.failUnless(N.sum(osamples[filt]) == 0)
        self.failUnless((osamples[N.negative(filt)] > 0).all())

    def testEventDatasetExtended(self):
        if not externals.exists('nifti'):
            return
        from mvpa.datasets.nifti import ERNiftiDataset
        try:
            ds = ERNiftiDataset(
                samples=os.path.join(pymvpa_dataroot, 'bold.nii.gz'),
                mask=os.path.join(pymvpa_dataroot, 'mask.nii.gz'),
                events=[Event(onset=1,duration=5,label=1,chunk=1)],
                evconv=True, tr=2.0)
        except ValueError, e:
            self.fail("Failed to create a simple ERNiftiDataset from a volume"
                      " with only 1 slice. Exception was:\n %s" % e)

def suite():
    return unittest.makeSuite(MetaDatasetTests)


if __name__ == '__main__':
    import runner

