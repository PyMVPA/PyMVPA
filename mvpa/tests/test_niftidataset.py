# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA nifti dataset"""

import os.path
import numpy as N
from nifti import NiftiImage

from numpy.testing import assert_array_equal
from nose.tools import ok_, assert_raises, assert_false, assert_equal, \
        assert_true

from mvpa import pymvpa_dataroot
from mvpa.datasets.mri import fmri_dataset
from mvpa.misc.fsl import FslEV3
from mvpa.misc.support import Event
from mvpa.misc.io.base import SampleAttributes


def testNiftiDataset():
    """Basic testing of NiftiDataset
    """
    ds = fmri_dataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
                       labels=[1,2], sprefix='voxel')
    assert_equal(ds.nfeatures, 294912)
    assert_equal(ds.nsamples, 2)

    assert_array_equal(ds.a.voxel_eldim, ds.a.imghdr['pixdim'][3:0:-1])
    assert_true(ds.a['voxel_dim'].value == (24,96,128))


    # XXX move elsewhere
    #check that mapper honours elementsize
    #nb22 = N.array([i for i in data.a.mapper.getNeighborIn((1, 1, 1), 2.2)])
    #nb20 = N.array([i for i in data.a.mapper.getNeighborIn((1, 1, 1), 2.0)])
    #self.failUnless(nb22.shape[0] == 7)
    #self.failUnless(nb20.shape[0] == 5)

    merged = ds + ds
    assert_equal(merged.nfeatures, 294912)
    assert_equal(merged.nsamples, 4)

    # check that the header survives
    for k in merged.a.imghdr.keys():
        assert_array_equal(merged.a.imghdr[k], ds.a.imghdr[k])

    # throw away old dataset and see if new one survives
    del ds
    assert_array_equal(merged.samples[3], merged.samples[1])

    # check whether we can use a plain ndarray as mask
    mask = N.zeros((24, 96, 128), dtype='bool')
    mask[12, 20, 40] = True
    nddata = fmri_dataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
                          labels=[1,2],
                          mask=mask)
    assert_equal(nddata.nfeatures, 1)
    rmap = nddata.a.mapper.reverse1(N.array([44]))
    assert_equal(rmap.shape, (24, 96, 128))
    assert_equal(N.sum(rmap), 44)
    assert_equal(rmap[12, 20, 40], 44)


def test_fmridataset():
    # full-blown fmri dataset testing
    attr = SampleAttributes(os.path.join(pymvpa_dataroot, 'attributes.txt'))
    ds = fmri_dataset(samples=os.path.join(pymvpa_dataroot,'bold'),
                      labels=attr.labels, chunks=attr.chunks,
                      mask=os.path.join(pymvpa_dataroot, 'mask.nii.gz'),
                      sprefix='subj1')
    # content
    assert_equal(len(ds), 1452)
    assert_true(ds.nfeatures, 530)
    assert_array_equal(sorted(ds.sa.keys()),
            ['chunks', 'labels', 'time_coords', 'time_indices'])
    assert_array_equal(sorted(ds.fa.keys()),
            ['subj1_indices'])
    assert_array_equal(sorted(ds.a.keys()),
            ['imghdr', 'mapper', 'subj1_dim', 'subj1_eldim'])
    # vol extent
    assert_equal(ds.a.subj1_dim, (1, 20, 40))
    # check time
    assert_equal(ds.sa.time_coords[-1], 3627.5)



def testNiftiMapper():
    """Basic testing of map2Nifti
    """
    data = fmri_dataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
                        labels=[1,2])

    # test mapping of ndarray
    vol = data.map2nifti(N.ones((294912,), dtype='int16'))
    assert_equal(vol.data.shape, (24, 96, 128))
    assert_true((vol.data == 1).all())

    # test mapping of the dataset
    vol = data.map2nifti(data)
    assert_equal(vol.data.shape, (2, 24, 96, 128))


def testNiftiSelfMapper():
    """Test map2Nifti facility ran without arguments
    """
    example_path = os.path.join(pymvpa_dataroot, 'example4d')
    example = NiftiImage(example_path)
    data = fmri_dataset(samples=example_path,
                         labels=[1,2])

    # Map read data to itself
    vol = data.map2nifti()

    assert_equal(vol.data.shape, example.data.shape)
    assert_array_equal(vol.data, example.data)

    data.samples[:] = 1
    vol = data.map2nifti()
    assert_true((vol.data == 1).all())


def testMultipleCalls():
    """Test if doing exactly the same operation twice yields the same result
    """
    data = fmri_dataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
                        labels=1, sprefix='abc')
    data2 = fmri_dataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
                         labels=1, sprefix='abc')
    assert_array_equal(data.a.abc_eldim, data2.a.abc_eldim)


#def testERNiftiDataset(self):
#    """Basic testing of ERNiftiDataset
#    """
#    self.failUnlessRaises(DatasetError, ERNiftiDataset)
#
#    # setup data sources
#    tssrc = os.path.join(pymvpa_dataroot, 'bold')
#    evsrc = os.path.join(pymvpa_dataroot, 'fslev3.txt')
#    # masrc = os.path.join(pymvpa_dataroot, 'mask')
#    evs = FslEV3(evsrc).toEvents()
#
#    # more failure ;-)
#    # no label!
#    self.failUnlessRaises(ValueError, ERNiftiDataset,
#                          samples=tssrc, events=evs)
#
#    # set some label for each ev
#    for ev in evs:
#        ev['label'] = 1
#
#    # for real!
#    # using TR from nifti header
#    ds = ERNiftiDataset(samples=tssrc, events=evs)
#
#    # 40x20 volume, 9 volumes per sample + 1 intensity score = 7201 features
#    self.failUnless(ds.nfeatures == 7201)
#    self.failUnless(ds.nsamples == len(evs))
#
#    # check samples
#    origsamples = getNiftiFromAnySource(tssrc).data
#    for i, ev in enumerate(evs):
#        self.failUnless((ds.samples[i][:-1] \
#            == origsamples[ev['onset']:ev['onset'] + ev['duration']].ravel()
#                        ).all())
#
#    # do again -- with conversion
#    ds = ERNiftiDataset(samples=tssrc, events=evs, evconv=True,
#                        storeoffset=True)
#    self.failUnless(ds.nsamples == len(evs))
#    # TR=2.5, 40x20 volume, 9 second per sample (4volumes), 1 intensity
#    # score + 1 offset = 3202 features
#    self.failUnless(ds.nfeatures == 3202)
#
#    # map back into voxel space, should ignore addtional features
#    nim = ds.map2Nifti()
#    self.failUnless(nim.data.shape == origsamples.shape)
#    # check shape of a single sample
#    nim = ds.map2Nifti(ds.samples[0])
#    self.failUnless(nim.data.shape == (4, 1, 20, 40))
#
#
#def testERNiftiDatasetMapping(self):
#    """Some mapping testing -- more tests is better
#    """
#    sample_size = (4, 3, 2)
#    samples = N.arange(120).reshape((5,) + sample_size)
#    dsmask = N.arange(24).reshape(sample_size)%2
#    ds = ERNiftiDataset(samples=NiftiImage(samples),
#                        events=[Event(onset=0, duration=2, label=1,
#                                      chunk=1, features=[1000, 1001]),
#                                Event(onset=1, duration=2, label=2,
#                                      chunk=1, features=[2000, 2001])],
#                        mask=dsmask)
#    nfeatures = ds.a.mapper._mappers[1].get_insize()
#    mask = N.zeros(sample_size)
#    mask[0, 0, 0] = mask[1, 0, 1] = mask[0, 0, 1] = 1 # select only 3
#    # but since 0th is masked out in the dataset, we should end up
#    # selecting only 2 from the dataset
#    #sel_orig_features = [1, 7]
#
#    # select using mask in volume and all features in the other part
#    ds_sel = ds[:, ds.a.mapper.forward([mask, [1]*nfeatures]).nonzero()[0]]
#
#    # now tests
#    self.failUnless((mask.reshape(24).nonzero()[0] == [0, 1, 7]).all())
#    self.failUnless(ds_sel.samples.shape == (2, 6),
#                    msg="We should have selected all samples, and 6 "
#                    "features (2 voxels at 2 timepoints + 2 features). "
#                    "Got %s" % (ds_sel.samples.shape,))
#    self.failUnless((ds_sel.samples[:, -2:] ==
#                     [[1000, 1001], [2000, 2001]]).all(),
#                    msg="We should have selected additional features "
#                    "correctly. Got %s" % ds_sel.samples[:, -2:])
#    self.failUnless((ds_sel.samples[:, :-2] ==
#                     [[   1,    7,   25,   31],
#                      [  25,   31,   49,   55]]).all(),
#                    msg="We should have selected original features "
#                    "correctly. Got %s" % ds_sel.samples[:, :-2])


#def testNiftiDatasetFrom3D():
#    """Test NiftiDataset based on 3D volume(s)
#    """
#    tssrc = os.path.join(pymvpa_dataroot, 'bold')
#    masrc = os.path.join(pymvpa_dataroot, 'mask')
#
#    # Test loading of 3D volumes
#
#    # it should puke if we are not enforcing 4D:
#    assert_raises(Exception, fmri_dataset,
#                  masrc, mask=masrc, labels=1, enforce_dim=None)
#    # by default we are enforcing it, however testing here with the demo 3d mask
#    ds = fmri_dataset(masrc, mask=masrc, labels=1, enforce_dim=4)
#
#    plain_data = NiftiImage(masrc).data
#    # Lets check if mapping back works as well
#    assert_array_equal(plain_data,
#                       ds.map2nifti().data.reshape(plain_data.shape))
#
#    # test loading from a list of filenames
#
#    # for now we should fail if trying to load a mix of 4D and 3D volumes
#    assert_raises(ValueError, fmri_dataset, (masrc, tssrc),
#                  mask=masrc, labels=1)
#
#    # Lets prepare some custom NiftiImage
#    dsfull = fmri_dataset(tssrc, mask=masrc, labels=1)
#    print dsfull.fa
#    ds_selected = dsfull[3]
#    nifti_selected = ds_selected.map2nifti()
#
#    # Load dataset from a mix of 3D volumes
#    # (given by filenames and NiftiImages)
#    labels = [123, 2, 123]
#    ds2 = fmri_dataset((masrc, masrc, nifti_selected),
#                       mask=masrc, labels=labels)
#    assert_equal(ds2.nsamples, 3)
#    assert_array_equal(ds2.samples[0], ds2.samples[1])
#    assert_array_equal(ds2.samples[2], dsfull.samples[3])
#    assert_array_equal(ds2.labels, labels)


#def testNiftiDatasetROIMaskNeighbors(self):
#    """Test if we could request neighbors within spherical ROI whenever
#       center is outside of the mask
#       """
#
#    # check whether we can use a plain ndarray as mask
#    mask_roi = N.zeros((24, 96, 128), dtype='bool')
#    mask_roi[12, 20, 38:42] = True
#    mask_roi[23, 20, 38:42] = True  # far away
#    ds_full = nifti_dataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
#                           labels=[1,2])
#    ds_roi = nifti_dataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
#                           labels=[1,2], mask=mask_roi)
#    # Should just work since we are in the mask
#    ids_roi = ds_roi.a.mapper.getNeighbors(
#                    ds_roi.a.mapper.getOutId((12, 20, 40)),
#                    radius=20)
#    self.failUnless(len(ids_roi) == 4)
#
#    # Trying to request feature outside of the mask
#    self.failUnlessRaises(ValueError,
#                          ds_roi.a.mapper.getOutId,
#                          (12, 20, 37))
#
#    # Lets work around using full (non-masked) volume
#    ids_out = []
#    for id_in in ds_full.a.mapper.getNeighborIn( (12, 20, 37), radius=20):
#        try:
#            ids_out.append(ds_roi.a.mapper.getOutId(id_in))
#        except ValueError:
#            pass
#    self.failUnless(ids_out == ids_roi)
