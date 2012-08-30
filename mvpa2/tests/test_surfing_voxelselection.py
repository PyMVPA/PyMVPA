# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA surface searchlight voxel selection"""

import numpy as np
from numpy.testing.utils import assert_array_almost_equal

import nibabel as nb

import os
import tempfile

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets

from mvpa2 import cfg
from mvpa2.base import externals
from mvpa2.datasets import Dataset
from mvpa2.measures.base import Measure
from mvpa2.datasets.mri import fmri_dataset

import mvpa2.misc.surfing.surf as surf
import mvpa2.misc.surfing.surf_fs_asc as surf_fs_asc
import mvpa2.misc.surfing.volgeom as volgeom
import mvpa2.misc.surfing.volsurf as volsurf
import mvpa2.misc.surfing.sparse_attributes as sparse_attributes
import mvpa2.misc.surfing.surf_voxel_selection as surf_voxel_selection
import mvpa2.misc.surfing.queryengine as queryengine

from mvpa2.measures.searchlight import Searchlight
from mvpa2.misc.surfing.queryengine import SurfaceVerticesQueryEngine

from mvpa2.measures.base import Measure, \
        TransferMeasure, RepeatedMeasure, CrossValidation
from mvpa2.clfs.smlr import SMLR
from mvpa2.generators.partition import OddEvenPartitioner
from mvpa2.mappers.fx import mean_sample
from mvpa2.misc.io.base import SampleAttributes
from mvpa2.mappers.detrend import poly_detrend
from mvpa2.mappers.zscore import zscore
from mvpa2.misc.neighborhood import Sphere, IndexQueryEngine


class SurfVoxelSelectionTests(unittest.TestCase):
    # runs voxel selection and searchlight (surface-based) on haxby 2001 
    # single plane data using a synthetic planar surface 

    # TODO make this an actual test to see if voxel and surface based
    # voxel selection matches
    def test_voxel_selection(self):
        '''Define searchlight radius (in mm)
        
        Note that the current value is a float; if it were int, it would specify
        the number of voxels in each searchlight'''
        radius = 10.


        '''Define input filenames'''
        epi_fn = os.path.join(pymvpa_dataroot, 'bold.nii.gz')
        maskfn = os.path.join(pymvpa_dataroot, 'mask.nii.gz')

        '''
        Use the EPI datafile to define a surface.
        The surface has as many nodes as there are voxels
        and is parallel to the volume 'slice'
        '''
        vg = volgeom.from_nifti_file(epi_fn)
        aff = vg.affine
        nx, ny, nz = vg.shape[:3]

        '''Plane goes in x and y direction, so we take these vectors
        from the affine transformation matrix of the volume'''
        plane = surf.generate_plane(aff[:3, 3], aff[:3, 0], aff[:3, 1], nx, ny)

        '''
        Simulate pial and white matter as just above and below the central plane
        '''
        outer = plane + aff[2, 2]
        inner = plane + -aff[2, 2]

        '''
        Combine volume and surface information
        '''
        vs = volsurf.VolSurf(vg, outer, inner)

        '''
        Run voxel selection with specified radius (in mm)
        '''
        voxsel = surf_voxel_selection.voxel_selection(vs, radius)

        '''
        Load an apply a volume-metric mask, and get a new instance
        of voxel selection results.
        In this new instance, only voxels that survive the epi mask
        are kept
        '''
        maskfn = os.path.join(pymvpa_dataroot, 'mask.nii.gz')
        epi_mask = fmri_dataset(maskfn).samples[0]
        voxsel_masked = voxsel.get_masked_instance(epi_mask)

        '''Define the query engine, cross validation, and searchligyht'''

        qe = SurfaceVerticesQueryEngine(voxsel_masked)
        cv = CrossValidation(SMLR(), OddEvenPartitioner(),
                             errorfx=lambda p, t: np.mean(p == t))

        sl = Searchlight(cv, queryengine=qe, postproc=mean_sample())

        '''The following steps are similar to start_easy.py'''
        attr = SampleAttributes(os.path.join(pymvpa_dataroot,
                                'attributes_literal.txt'))

        dataset = fmri_dataset(samples=os.path.join(pymvpa_dataroot, 'bold.nii.gz'),
                               targets=attr.targets, chunks=attr.chunks,
                               mask=voxsel_masked.get_mask())

        # do chunkswise linear detrending on dataset
        poly_detrend(dataset, polyord=1, chunks_attr='chunks')

        # zscore dataset relative to baseline ('rest') mean
        zscore(dataset, chunks_attr='chunks', param_est=('targets', ['rest']))

        # select class face and house for this demo analysis
        # would work with full datasets (just a little slower)
        dataset = dataset[np.array([l in ['face', 'house'] for l in dataset.sa.targets],
                                  dtype='bool')]

        '''Apply searchlight to dataset'''
        dset_out = sl(dataset)



        # now define a similar searchlight for the volume
        a = np.abs(vg.affine) # to get voxel sizes
        sph = Sphere(radius, element_sizes=(a[0, 0], a[1, 1], a[2, 2])) # sphere
        kwa = {'voxel_indices': sph}

        qevol = IndexQueryEngine(**kwa)
        slvol = Searchlight(cv, queryengine=qevol, postproc=mean_sample())
        dsvol = slvol(dataset)

        '''Make ready for storing in AFNI NIML dataset'''
        niml_dset = dict(data=dset_out.samples.transpose(), node_indices=qe.ids)

        '''
        
        Additional commands, not executed here:
        
        1) to save surface geometry:
        # surf.write('surf.asc',plane)
        
        2) to save surface data in AFNI's NIML format:
        # afni_niml_dset.write('surfdata.niml.dset', niml_dset)
        
        3) To load in AFNI's SUMA (on the terminal):
        # suma -i surf.asc
        and load Dset and select surfdata.niml.dset')
        
        '''

def suite():
    """Create the suite"""
    return unittest.makeSuite(SurfVoxelSelectionTests)


if __name__ == '__main__':
    import runner
