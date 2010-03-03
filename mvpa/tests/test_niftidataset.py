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
import numpy as np

from mvpa.testing import *

if not externals.exists('nifti') and not externals.exists('nibabel'):
    raise SkipTest

from mvpa import pymvpa_dataroot
from mvpa.datasets.mri import fmri_dataset, _load_anyimg, map2nifti
from mvpa.datasets.eventrelated import eventrelated_dataset
from mvpa.misc.fsl import FslEV3
from mvpa.misc.support import Event, value2idx
from mvpa.misc.io.base import SampleAttributes


def test_nifti_dataset():
    """Basic testing of NiftiDataset
    """
    ds = fmri_dataset(samples=os.path.join(pymvpa_dataroot,'example4d.nii.gz'),
                       targets=[1,2], sprefix='voxel')
    assert_equal(ds.nfeatures, 294912)
    assert_equal(ds.nsamples, 2)

    assert_array_equal(ds.a.voxel_eldim, ds.a.imghdr['pixdim'][1:4])
    assert_true(ds.a['voxel_dim'].value == (128,96,24))


    # XXX move elsewhere
    #check that mapper honours elementsize
    #nb22 = np.array([i for i in data.a.mapper.getNeighborIn((1, 1, 1), 2.2)])
    #nb20 = np.array([i for i in data.a.mapper.getNeighborIn((1, 1, 1), 2.0)])
    #self.failUnless(nb22.shape[0] == 7)
    #self.failUnless(nb20.shape[0] == 5)

    merged = ds.copy()
    merged.append(ds)
    assert_equal(merged.nfeatures, 294912)
    assert_equal(merged.nsamples, 4)

    # check that the header survives
    for k in merged.a.imghdr.keys():
        assert_array_equal(merged.a.imghdr[k], ds.a.imghdr[k])

    # throw away old dataset and see if new one survives
    del ds
    assert_array_equal(merged.samples[3], merged.samples[1])

    # check whether we can use a plain ndarray as mask
    mask = np.zeros((128, 96, 24), dtype='bool')
    mask[40, 20, 12] = True
    nddata = fmri_dataset(samples=os.path.join(pymvpa_dataroot,'example4d.nii.gz'),
                          targets=[1,2],
                          mask=mask)
    assert_equal(nddata.nfeatures, 1)
    rmap = nddata.a.mapper.reverse1(np.array([44]))
    assert_equal(rmap.shape, (128, 96, 24))
    assert_equal(np.sum(rmap), 44)
    assert_equal(rmap[40, 20, 12], 44)


def test_fmridataset():
    # full-blown fmri dataset testing
    if externals.exists('nibabel'):
        import nibabel
        maskimg = nibabel.load(os.path.join(pymvpa_dataroot, 'mask.nii.gz'))
        data = maskimg.get_data().copy()
        data[data>0] = np.arange(1, np.sum(data) + 1)
        maskimg = nibabel.Nifti1Image(data, None, maskimg.get_header())
    else:
        import nifti
        maskimg = nifti.NiftiImage(os.path.join(pymvpa_dataroot, 'mask.nii.gz'))
        # assign some values we can check later on
        maskimg.data.T[maskimg.data.T>0] = np.arange(1, np.sum(maskimg.data) + 1)
    attr = SampleAttributes(os.path.join(pymvpa_dataroot, 'attributes.txt'))
    ds = fmri_dataset(samples=os.path.join(pymvpa_dataroot,'bold.nii.gz'),
                      targets=attr.targets, chunks=attr.chunks,
                      mask=maskimg,
                      sprefix='subj1',
                      add_fa={'myintmask': maskimg})
    # content
    assert_equal(len(ds), 1452)
    assert_true(ds.nfeatures, 530)
    assert_array_equal(sorted(ds.sa.keys()),
            ['chunks', 'targets', 'time_coords', 'time_indices'])
    assert_array_equal(sorted(ds.fa.keys()),
            ['myintmask', 'subj1_indices'])
    assert_array_equal(sorted(ds.a.keys()),
            ['imghdr', 'mapper', 'subj1_dim', 'subj1_eldim'])
    # vol extent
    assert_equal(ds.a.subj1_dim, (40, 20, 1))
    # check time
    assert_equal(ds.sa.time_coords[-1], 3627.5)
    # non-zero mask values
    assert_array_equal(ds.fa.myintmask, np.arange(1, ds.nfeatures + 1))



def test_nifti_mapper():
    """Basic testing of map2Nifti
    """
    data = fmri_dataset(samples=os.path.join(pymvpa_dataroot,'example4d.nii.gz'),
                        targets=[1,2])

    # test mapping of ndarray
    vol = map2nifti(data, np.ones((294912,), dtype='int16'))
    if externals.exists('nibabel'):
        assert_equal(vol.get_shape(), (128, 96, 24))
        assert_true((vol.get_data() == 1).all())
        # test mapping of the dataset
        vol = map2nifti(data)
        assert_equal(vol.get_shape(), (128, 96, 24, 2))
    else:
        assert_equal(vol.data.shape, (24, 96, 128))
        assert_true((vol.data == 1).all())
        # test mapping of the dataset
        vol = map2nifti(data)
        assert_equal(vol.data.shape, (2, 24, 96, 128))


def test_multiple_calls():
    """Test if doing exactly the same operation twice yields the same result
    """
    data = fmri_dataset(samples=os.path.join(pymvpa_dataroot,'example4d.nii.gz'),
                        targets=1, sprefix='abc')
    data2 = fmri_dataset(samples=os.path.join(pymvpa_dataroot,'example4d.nii.gz'),
                         targets=1, sprefix='abc')
    assert_array_equal(data.a.abc_eldim, data2.a.abc_eldim)


def test_er_nifti_dataset():
    # setup data sources
    tssrc = os.path.join(pymvpa_dataroot, 'bold.nii.gz')
    evsrc = os.path.join(pymvpa_dataroot, 'fslev3.txt')
    masrc = os.path.join(pymvpa_dataroot, 'mask.nii.gz')
    evs = FslEV3(evsrc).to_events()
    # load timeseries
    ds_orig = fmri_dataset(tssrc)
    # segment into events
    ds = eventrelated_dataset(ds_orig, evs, time_attr='time_coords')

    # we ask for boxcars of 9s length, and the tr in the file header says 2.5s
    # hence we should get round(9.0/2.4) * np.prod((1,20,40) == 3200 features
    assert_equal(ds.nfeatures, 3200)
    assert_equal(len(ds), len(evs))
    # the voxel indices are reflattened after boxcaring , but still 3D
    assert_equal(ds.fa.voxel_indices.shape, (ds.nfeatures, 3))
    # and they have been broadcasted through all boxcars
    assert_array_equal(ds.fa.voxel_indices[:800], ds.fa.voxel_indices[800:1600])
    # each feature got an event offset value
    assert_array_equal(ds.fa.event_offsetidx, np.repeat([0,1,2,3], 800))
    # check for all event attributes
    assert_true('onset' in ds.sa)
    assert_true('duration' in ds.sa)
    assert_true('features' in ds.sa)
    # check samples
    origsamples = _load_anyimg(tssrc)[0]
    for i, onset in \
        enumerate([value2idx(e['onset'], ds_orig.sa.time_coords, 'floor')
                        for e in evs]):
        assert_array_equal(ds.samples[i], origsamples[onset:onset+4].ravel())
        assert_array_equal(ds.sa.time_indices[i], np.arange(onset, onset + 4))
        assert_array_equal(ds.sa.time_coords[i],
                           np.arange(onset, onset + 4) * 2.5)
        for evattr in [a for a in ds.sa
                        if a.count("event_attrs")
                           and not a.count('event_attrs_event')]:
            assert_array_equal(evs[i]['_'.join(evattr.split('_')[2:])],
                               ds.sa[evattr].value[i])
    # check offset: only the last one exactly matches the tr
    assert_array_equal(ds.sa.orig_offset, [1, 1, 0])

    # map back into voxel space, should ignore addtional features
    nim = map2nifti(ds)
    if externals.exists('nibabel'):
        # origsamples has t,x,y,z
        assert_equal(nim.get_shape(), origsamples.shape[1:] + (len(ds) * 4,))
        # check shape of a single sample
        nim = map2nifti(ds, ds.samples[0])
        # pynifti image has [t,]z,y,x
        assert_equal(nim.get_shape(), (40, 20, 1, 4))
    else:
        # origsamples has t,x,y,z but pynifti image has [t,]z,y,x
        assert_equal(nim.data.T.shape, origsamples.shape[1:] + (len(ds) * 4,))
        # check shape of a single sample
        nim = map2nifti(ds, ds.samples[0])
        # pynifti image has [t,]z,y,x
        assert_equal(nim.data.T.shape, (40, 20, 1, 4))

    # and now with masking
    ds = fmri_dataset(tssrc, mask=masrc)
    ds = eventrelated_dataset(ds, evs, time_attr='time_coords')
    nnonzero = len(_load_anyimg(masrc)[0].nonzero()[0])
    assert_equal(nnonzero, 530)
    # we ask for boxcars of 9s length, and the tr in the file header says 2.5s
    # hence we should get round(9.0/2.4) * np.prod((1,20,40) == 3200 features
    assert_equal(ds.nfeatures, 4 * 530)
    assert_equal(len(ds), len(evs))
    # and they have been broadcasted through all boxcars
    assert_array_equal(ds.fa.voxel_indices[:nnonzero],
                       ds.fa.voxel_indices[nnonzero:2*nnonzero])



def test_er_nifti_dataset_mapping():
    """Some mapping testing -- more tests is better
    """
    # z,y,x
    sample_size = (4, 3, 2)
    # t,z,y,x
    samples = np.arange(120).reshape((5,) + sample_size)
    dsmask = np.arange(24).reshape(sample_size) % 2
    if externals.exists('nibabel'):
        import nibabel
        tds = fmri_dataset(nibabel.Nifti1Image(samples.T, None),
                           mask=nibabel.Nifti1Image(dsmask.T, None))
    else:
        import nifti
        tds = fmri_dataset(nifti.NiftiImage(samples),
                           mask=nifti.NiftiImage(dsmask))
    ds = eventrelated_dataset(
            tds,
            events=[Event(onset=0, duration=2, label=1,
                          chunk=1, features=[1000, 1001]),
                    Event(onset=1, duration=2, label=2,
                          chunk=1, features=[2000, 2001])])
    nfeatures = tds.nfeatures
    mask = np.zeros(dsmask.shape, dtype='bool')
    mask[0, 0, 0] = mask[1, 0, 1] = mask[0, 0, 1] = 1
    fmask = ds.a.mapper.forward1(mask.T)
    # select using mask in volume and all features in the other part
    ds_sel = ds[:, fmask]

    # now tests
    assert_array_equal(mask.reshape(24).nonzero()[0], [0, 1, 7])
    # two events, 2 orig features at 2 timepoints
    assert_equal(ds_sel.samples.shape, (2, 4))
    assert_array_equal(ds_sel.sa.features,
                       [[1000, 1001], [2000, 2001]])
    assert_array_equal(ds_sel.samples,
                       [[   1,    7,   25,   31],
                        [  25,   31,   49,   55]])
    # reproducability
    assert_array_equal(ds_sel.samples,
                       ds_sel.a.mapper.forward(np.rollaxis(samples.T, -1)))

    # reverse-mapping
    rmapped = ds_sel.a.mapper.reverse1(np.arange(10, 14))
    assert_equal(np.rollaxis(rmapped, 0, 4).T.shape, (2,) + sample_size)
    expected = np.zeros((2,)+sample_size, dtype='int')
    expected[0,0,0,1] = 10
    expected[0,1,0,1] = 11
    expected[1,0,0,1] = 12
    expected[1,1,0,1] = 13
    assert_array_equal(np.rollaxis(rmapped, 0, 4).T, expected)


def test_nifti_dataset_from3_d():
    """Test NiftiDataset based on 3D volume(s)
    """
    tssrc = os.path.join(pymvpa_dataroot, 'bold.nii.gz')
    masrc = os.path.join(pymvpa_dataroot, 'mask.nii.gz')

    # Test loading of 3D volumes
    # by default we are enforcing 4D, testing here with the demo 3d mask
    ds = fmri_dataset(masrc, mask=masrc, targets=1)
    assert_equal(len(ds), 1)

    if externals.exists('nibabel'):
        import nibabel
        plain_data = nibabel.load(masrc).get_data()
        # Lets check if mapping back works as well
        assert_array_equal(plain_data,
                           map2nifti(ds).get_data().reshape(plain_data.shape))
    else:
        import nifti
        plain_data = nifti.NiftiImage(masrc).data
        # Lets check if mapping back works as well
        assert_array_equal(plain_data,
                           map2nifti(ds).data.reshape(plain_data.shape))

    # test loading from a list of filenames

    # for now we should fail if trying to load a mix of 4D and 3D volumes
    assert_raises(ValueError, fmri_dataset, (masrc, tssrc),
                  mask=masrc, targets=1)

    # Lets prepare some custom NiftiImage
    dsfull = fmri_dataset(tssrc, mask=masrc, targets=1)
    ds_selected = dsfull[3]
    nifti_selected = map2nifti(ds_selected)

    # Load dataset from a mix of 3D volumes
    # (given by filenames and NiftiImages)
    labels = [123, 2, 123]
    ds2 = fmri_dataset((masrc, masrc, nifti_selected),
                       mask=masrc, targets=labels)
    assert_equal(ds2.nsamples, 3)
    assert_array_equal(ds2.samples[0], ds2.samples[1])
    assert_array_equal(ds2.samples[2], dsfull.samples[3])
    assert_array_equal(ds2.targets, labels)


#def test_nifti_dataset_roi_mask_neighbors(self):
#    """Test if we could request neighbors within spherical ROI whenever
#       center is outside of the mask
#       """
#
#    # check whether we can use a plain ndarray as mask
#    mask_roi = np.zeros((24, 96, 128), dtype='bool')
#    mask_roi[12, 20, 38:42] = True
#    mask_roi[23, 20, 38:42] = True  # far away
#    ds_full = nifti_dataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
#                           targets=[1,2])
#    ds_roi = nifti_dataset(samples=os.path.join(pymvpa_dataroot,'example4d'),
#                           targets=[1,2], mask=mask_roi)
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
