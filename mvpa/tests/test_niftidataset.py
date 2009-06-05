# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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
from mvpa.misc.fsl import FslEV3
from mvpa.misc.support import Event

class NiftiDatasetTests(unittest.TestCase):
    """Tests of various Nifti-based datasets
    """

    def testNiftiDataset(self):
        """Basic testing of NiftiDataset
        """
        data = NiftiDataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
                            labels=[1,2])
        self.failUnless(data.nfeatures == 294912)
        self.failUnless(data.nsamples == 2)

        self.failUnless((data.mapper.metric.elementsize \
                         == data.niftihdr['pixdim'][3:0:-1]).all())

        #check that mapper honours elementsize
        nb22 = N.array([i for i in data.mapper.getNeighborIn((1, 1, 1), 2.2)])
        nb20 = N.array([i for i in data.mapper.getNeighborIn((1, 1, 1), 2.0)])
        self.failUnless(nb22.shape[0] == 7)
        self.failUnless(nb20.shape[0] == 5)

        # Can't rely on released pynifties, so doing really vague testing
        # XXX
        self.failUnless(data.dt in [2.0, 2000.0])
        self.failUnless(data.samplingrate in [5e-4, 5e-1])
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

        # check whether we can use a plain ndarray as mask
        mask = N.zeros((24, 96, 128), dtype='bool')
        mask[12, 20, 40] = True
        nddata = NiftiDataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
                              labels=[1,2],
                              mask=mask)
        self.failUnless(nddata.nfeatures == 1)
        rmap = nddata.mapReverse([44])
        self.failUnless(rmap.shape == (24, 96, 128))
        self.failUnless(N.sum(rmap) == 44)
        self.failUnless(rmap[12, 20, 40] == 44)


    def testNiftiMapper(self):
        """Basic testing of map2Nifti
        """
        data = NiftiDataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
                            labels=[1,2])

        # test mapping of ndarray
        vol = data.map2Nifti(N.ones((294912,), dtype='int16'))
        self.failUnless(vol.data.shape == (24, 96, 128))
        self.failUnless((vol.data == 1).all())

        # test mapping of the dataset
        vol = data.map2Nifti(data)
        self.failUnless(vol.data.shape == (2, 24, 96, 128))


    def testNiftiSelfMapper(self):
        """Test map2Nifti facility ran without arguments
        """
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
        """Test if doing exactly the same operation twice yields the same result
        """
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
        """Basic testing of ERNiftiDataset
        """
        self.failUnlessRaises(DatasetError, ERNiftiDataset)

        # setup data sources
        tssrc = os.path.join(pymvpa_dataroot, 'bold')
        evsrc = os.path.join(pymvpa_dataroot, 'fslev3.txt')
        # masrc = os.path.join(pymvpa_dataroot, 'mask')
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


    def testERNiftiDatasetMapping(self):
        """Some mapping testing -- more tests is better
        """
        sample_size = (4, 3, 2)
        samples = N.arange(120).reshape((5,) + sample_size)
        dsmask = N.arange(24).reshape(sample_size)%2
        ds = ERNiftiDataset(samples=NiftiImage(samples),
                            events=[Event(onset=0, duration=2, label=1,
                                          chunk=1, features=[1000, 1001]),
                                    Event(onset=1, duration=2, label=2,
                                          chunk=1, features=[2000, 2001])],
                            mask=dsmask)
        nfeatures = ds.mapper._mappers[1].getInSize()
        mask = N.zeros(sample_size)
        mask[0, 0, 0] = mask[1, 0, 1] = mask[0, 0, 1] = 1 # select only 3
        # but since 0th is masked out in the dataset, we should end up
        # selecting only 2 from the dataset
        #sel_orig_features = [1, 7]

        # select using mask in volume and all features in the other part
        ds_sel = ds.selectFeatures(
            ds.mapper.forward([mask, [1]*nfeatures]).nonzero()[0])

        # now tests
        self.failUnless((mask.reshape(24).nonzero()[0] == [0, 1, 7]).all())
        self.failUnless(ds_sel.samples.shape == (2, 6),
                        msg="We should have selected all samples, and 6 "
                        "features (2 voxels at 2 timepoints + 2 features). "
                        "Got %s" % (ds_sel.samples.shape,))
        self.failUnless((ds_sel.samples[:, -2:] ==
                         [[1000, 1001], [2000, 2001]]).all(),
                        msg="We should have selected additional features "
                        "correctly. Got %s" % ds_sel.samples[:, -2:])
        self.failUnless((ds_sel.samples[:, :-2] ==
                         [[   1,    7,   25,   31],
                          [  25,   31,   49,   55]]).all(),
                        msg="We should have selected original features "
                        "correctly. Got %s" % ds_sel.samples[:, :-2])


    def testNiftiDatasetFrom3D(self):
        """Test NiftiDataset based on 3D volume(s)
        """
        tssrc = os.path.join(pymvpa_dataroot, 'bold')
        masrc = os.path.join(pymvpa_dataroot, 'mask')

        # Test loading of 3D volumes

        # it should puke if we are not enforcing 4D:
        self.failUnlessRaises(Exception, NiftiDataset,
                              masrc, mask=masrc, labels=1, enforce4D=False)
        # by default we are enforcing it
        ds = NiftiDataset(masrc, mask=masrc, labels=1)

        plain_data = NiftiImage(masrc).data
        # Lets check if mapping back works as well
        self.failUnless(N.all(plain_data == \
                              ds.map2Nifti().data.reshape(plain_data.shape)))

        # test loading from a list of filenames

        # for now we should fail if trying to load a mix of 4D and 3D volumes
        self.failUnlessRaises(ValueError, NiftiDataset, (masrc, tssrc),
                              mask=masrc, labels=1)

        # Lets prepare some custom NiftiImage
        dsfull = NiftiDataset(tssrc, mask=masrc, labels=1)
        ds_selected = dsfull['samples', [3]]
        nifti_selected = ds_selected.map2Nifti()

        # Load dataset from a mix of 3D volumes
        # (given by filenames and NiftiImages)
        labels = [123, 2, 123]
        ds2 = NiftiDataset((masrc, masrc, nifti_selected),
                           mask=masrc, labels=labels)
        self.failUnless(ds2.nsamples == 3)
        self.failUnless((ds2.samples[0] == ds2.samples[1]).all())
        self.failUnless((ds2.samples[2] == dsfull.samples[3]).all())
        self.failUnless((ds2.labels == labels).all())


def suite():
    return unittest.makeSuite(NiftiDatasetTests)


if __name__ == '__main__':
    import runner

