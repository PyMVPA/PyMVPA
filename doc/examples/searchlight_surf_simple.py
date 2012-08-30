#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Simple surface-based saerchlight on plane of Haxby 2001 data

Surfaces here are 'planes' parallel to a slice of fMRI data.
It's a simple and quick example similar to start_easy.py

TODO: maybe move to unit tests?
"""

from mvpa2.suite import *


if __debug__:
    from mvpa2.base import debug
    debug.active += ["SVS", "SLC"]


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
volsurf = volsurf.VolSurf(vg, outer, inner)

'''
Run voxel selection with specified radius (in mm)
'''
voxsel = surf_voxel_selection.voxel_selection(volsurf, radius)

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
                       mask=voxsel.get_mask())


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
