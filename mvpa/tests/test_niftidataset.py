#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA nifti dataset"""

import unittest
import os.path
import numpy as N

from mvpa import pymvpa_dataroot
from mvpa.datasets.nifti import *
from mvpa.misc.exceptions import *
from mvpa.misc.fsl.base import FslEV3

class NiftiDatasetTests(unittest.TestCase):

    def testNiftiDataset(self):
        data = NiftiDataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
                            labels=[1,2])
        self.failUnless(data.nfeatures == 294912)
        self.failUnless(data.nsamples == 2)

        self.failUnless((data.mapper.metric.elementsize \
                         == data.niftihdr['pixdim'][3:0:-1]).all())

        #check that mapper honours elementsize
        nb22=N.array([i for i in data.mapper.getNeighborIn((1,1,1), 2.2)])
        nb20=N.array([i for i in data.mapper.getNeighborIn((1,1,1), 2.0)])
        self.failUnless(nb22.shape[0] == 7)
        self.failUnless(nb20.shape[0] == 5)

        merged = data + data

        self.failUnless(merged.nfeatures == 294912)
        self.failUnless(merged.nsamples == 4)

        # check that the header survives
        #self.failUnless(merged.niftihdr == data.niftihdr)
        for k in merged.niftihdr.keys():
            self.failUnless(N.mean(merged.niftihdr[k] == data.niftihdr[k]) == 1)

        # throw away old dataset and see if new one survives
        del data
        self.failUnless(merged.samples[3, 120000] == merged.samples[1, 120000])


    def testNiftiMapper(self):
        data = NiftiDataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
                            labels=[1,2])

        # test mapping of ndarray
        vol = data.map2Nifti(N.ones((294912,), dtype='int16'))
        self.failUnless(vol.data.shape == (24,96,128))
        self.failUnless((vol.data == 1).all())

        # test mapping of the dataset
        vol = data.map2Nifti(data)
        self.failUnless(vol.data.shape == (2, 24, 96, 128))


    def testNiftiSelfMapper(self):
        example_path = os.path.join(pymvpa_dataroot, 'example4d')
        example = NiftiImage(example_path)
        data = NiftiDataset(samples=example_path,
                            labels=[1,2])

        # Map read data to itself
        vol = data.map2Nifti()

        self.failUnless(vol.data.shape == example.data.shape)
        self.failUnless((vol.data == example.data).all())

        data.samples[:] = 1
        vol = data.map2Nifti()
        self.failUnless((vol.data == 1).all())


    def testMultipleCalls(self):
        # test if doing exactly the same operation twice yields the same
        # result
        data = NiftiDataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
                            labels=1)
        data2 = NiftiDataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
                             labels=1)

        # Currently this test fails and I don't know why!
        # The problem occurs, because in the second call to
        # NiftiDataset.__init__() there is already a dsattr that has a 'mapper'
        # key, although dsattr is set to be an empty dict. Therefore the
        # constructor does not set the proper elementsize, because it thinks
        # there is already a mapper present. Actually this test is just looking
        # for a symptom of a buggy dsattr handling.
        # The tricky part is: I have no clue, what is going on... :(
        self.failUnless((data.mapper.metric.elementsize \
                         == data2.mapper.metric.elementsize).all())


    def testERNiftiDataset(self):
        self.failUnlessRaises(DatasetError, ERNiftiDataset)

        # setup data sources
        tssrc = os.path.join(pymvpa_dataroot, 'bold')
        evsrc = os.path.join(pymvpa_dataroot, 'fslev3.txt')
        masrc = os.path.join(pymvpa_dataroot, 'mask')
        evs = FslEV3(evsrc).toEvents()

        # more failure ;-)
        # no label!
        self.failUnlessRaises(ValueError, ERNiftiDataset,
                              samples=tssrc, events=evs)

        # set some label for each ev
        for ev in evs:
            ev['label'] = 1

        # for real!
        # using TR from nifti header
        ds = ERNiftiDataset(samples=tssrc, events=evs)

        # 40x20 volume, 9 volumes per sample + 1 intensity score = 7201 features
        self.failUnless(ds.nfeatures == 7201)
        self.failUnless(ds.nsamples == len(evs))

        # check samples
        origsamples = getNiftiFromAnySource(tssrc).data
        for i, ev in enumerate(evs):
            self.failUnless((ds.samples[i][:-1] \
                == origsamples[ev['onset']:ev['onset'] + ev['duration']].ravel()
                            ).all())

        # do again -- with conversion
        ds = ERNiftiDataset(samples=tssrc, events=evs, evconv=True,
                            storeoffset=True)
        self.failUnless(ds.nsamples == len(evs))
        # TR=2.5, 40x20 volume, 9 second per sample (4volumes), 1 intensity
        # score + 1 offset = 3202 features 
        self.failUnless(ds.nfeatures == 3202)

        # map back into voxel space, should ignore addtional features
        nim = ds.map2Nifti()
        self.failUnless(nim.data.shape == origsamples.shape)
        # check shape of a single sample
        nim = ds.map2Nifti(ds.samples[0])
        self.failUnless(nim.data.shape == (4, 1, 20, 40))



def suite():
    return unittest.makeSuite(NiftiDatasetTests)


if __name__ == '__main__':
    import runner

