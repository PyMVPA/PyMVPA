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


#from mvpa2.suite import *
#from mvpa2.datasets.mri import fmri_dataset


class SurfVoxelSelectionTests(unittest.TestCase):
    # runs voxel selection and searchlight (surface-based) on haxby 2001 
    # single plane data using a synthetic planar surface 

    # checks to see if results are 'pretty much' similar for surface
    # and volume base searchlights (currently disagreemnt in classification
    # accuracies up to 4 precent)

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
        vg = volgeom.from_nifti_file(maskfn, 0)
        #vg = volgeom.from_nifti_file(maskfn)

        aff = vg.affine
        nx, ny, nz = vg.shape[:3]

        '''Plane goes in x and y direction, so we take these vectors
        from the affine transformation matrix of the volume'''
        plane = surf.generate_plane(aff[:3, 3], aff[:3, 0], aff[:3, 1], nx, ny)



        '''
        Simulate pial and white matter as just above and below the central plane
        '''
        normal_vec = aff[:3, 2]
        outer = plane + normal_vec
        inner = plane + -normal_vec

        '''
        Combine volume and surface information
        '''
        vs = volsurf.VolSurf(vg, outer, inner)

        '''
        Run voxel selection with specified radius (in mm)
        '''
        surf_voxsel = surf_voxel_selection.voxel_selection(vs, radius, distancemetric='e')

        '''
        Load an apply a volume - metric mask, and get a new instance
        of voxel selection results.
        In this new instance, only voxels that survive the epi mask
        are kept
        '''
        #epi_mask = fmri_dataset(maskfn).samples[0]
        #voxsel_masked = voxsel.get_masked_instance(epi_mask)


        '''Define cross validation'''
        cv = CrossValidation(SMLR(), OddEvenPartitioner(),
                                  errorfx=lambda p, t: np.mean(p == t))

        '''
        Surface analysis: define the query engine, cross validation, and searchlight
        '''
        surf_qe = SurfaceVerticesQueryEngine(surf_voxsel)
        surf_sl = Searchlight(cv, queryengine=surf_qe, postproc=mean_sample())

        '''
        Same for the volume analysis
        '''
        element_sizes = tuple(map(abs, (aff[0, 0], aff[1, 1], aff[2, 2])))
        sph = Sphere(radius, element_sizes=element_sizes)
        kwa = {'voxel_indices': sph}

        vol_qe = IndexQueryEngine(**kwa)
        vol_sl = Searchlight(cv, queryengine=vol_qe, postproc=mean_sample())


        '''The following steps are similar to start_easy.py'''
        attr = SampleAttributes(os.path.join(pymvpa_dataroot,
                                'attributes_literal.txt'))

        mask = surf_voxsel.get_mask()

        dataset = fmri_dataset(samples=os.path.join(pymvpa_dataroot, 'bold.nii.gz'),
                               targets=attr.targets, chunks=attr.chunks,
                               mask=mask)

        # do chunkswise linear detrending on dataset
        poly_detrend(dataset, polyord=1, chunks_attr='chunks')

        # zscore dataset relative to baseline ('rest') mean
        zscore(dataset, chunks_attr='chunks', param_est=('targets', ['rest']))

        # select class face and house for this demo analysis
        # would work with full datasets (just a little slower)
        dataset = dataset[np.array([l in ['face', 'house']
                                    for l in dataset.sa.targets], dtype='bool')]

        '''Apply searchlight to datasets'''
        surf_dset = surf_sl(dataset)
        vol_dset = vol_sl(dataset)

        surf_data = surf_dset.samples
        vol_data = vol_dset.samples

        # TODO see why agreement is not perfect - hopefully get precision
        # to more decimals
        assert_array_almost_equal(surf_data, vol_data, 1)

def suite():
    """Create the suite"""
    return unittest.makeSuite(SurfVoxelSelectionTests)


if __name__ == '__main__':
    import runner
