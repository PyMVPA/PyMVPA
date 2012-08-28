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
Searchlight on fMRI data
========================

.. index:: Searchlight

This example is adapted from doc/searchlight.py.

As always, we first have to import PyMVPA.
"""

from mvpa2.suite import *


"""As searchlight analyses are usually quite expensive in term of computational
resources, we are going to enable some progress output to entertain us while
we are waiting."""

# enable debug output for searchlight call
if __debug__:
    from mvpa2.base import debug
    debug.active += ["SVS", "SLC"]

"""The next few calls load an fMRI dataset, while assigning associated class
targets and chunks (experiment runs) to each volume in the 4D timeseries.  One
aspect is worth mentioning. When loading the fMRI data with
:func:`~mvpa2.datasets.mri.fmri_dataset()` additional feature attributes can be
added, by providing a dictionary with names and source pairs to the `add_fa`
arguments. In this case we are loading a thresholded zstat-map of a category
selectivity contrast for voxels ventral temporal cortex."""

# data path
datapath = os.path.join(pymvpa_datadbroot,
                        'tutorial_data', 'tutorial_data', 'data', 'surfing')

"""First set up surface stuff"""
epi_ref_fn = os.path.join(datapath, '..', 'mask_brain.nii.gz')

"""
We're concerned with the left hemisphere only.
"""
hemi = 'l'

"""
Surfaces that enclose the grey matter. These are used for voxel selection.
These surfaces were resampled using AFNI's MapIcosahedron; ld refers to
the number of linear divisions of the 'large' triangles of the original
icosahedron (ld=x means there are 10*x**2+2 nodes and 20*x**2 triangles).
"""
highres_ld = 128 # 64 or 128 is reasonable

pial_surf_fn = os.path.join(datapath, "ico%d_%sh.pial_al.asc"
                                     % (highres_ld, hemi))
white_surf_fn = os.path.join(datapath, "ico%d_%sh.smoothwm_al.asc"
                                      % (highres_ld, hemi))

"""
The surface on which the nodes are centers of the searchlight. We use a
coarser surface (fewer nodes). A limitation of the current surface-based
searchlight implementation in PyMVPA is that the number of voxels cannot
exceed the number of nodes (i.e. 10*lowres_ld**2+2 should not exceed the
number of nodes.

It is crucial here that highres_ld is a multiple of lowres_ld, so that
all nodes in the low-res surface have a corresponding node (i.e., with the same,
or almost the same, spatial coordinate) on the high-res surface.

Choice of lowres_ld and highres_ld is somewhat arbitrary and always application
specific. For highres_ld a value of at least 64 may be advisable as this
ensures enough anatomical detail is available to select voxels in the grey
matter accurately. For lowres_ld, a low number may be advisable for functional
or information-based connectivity analyses; e.g. lowres_ld=8 means there
are 2*(10*8^2+2)=1284 nodes across the two hemispheres, and thus 823686 unique
pairs of nodes. A higher number for lowres_ld may be  suited for single-center
searchlight analyses.
"""
lowres_ld = 32 # 16, 32 or 64 is reasonable

intermediate_surf_fn = os.path.join(datapath, "ico%d_%sh.intermediate_al.asc"
                                             % (lowres_ld, hemi))

"""
Radius is specified as either an int (referring to a fixed number of voxels
across searchlights, with a variable radius in millimeters (or whatever unit
is used in the files that define the surfaces), or a float (referring to the
radius in millimeters, with a variable number of voxels).

Note that "a fixed number of voxels" in this context actually means an
approximation, in that on average that number of voxels is selected but the
actual number will vary slightly
"""
radius = 100


"""
Set the prefixes for output
"""
fn_infix = 'ico%d_%sh_%dvx' % (lowres_ld, hemi, radius)
voxel_selection_fn_prefix = os.path.join(datapath, fn_infix)
searchlight_fn_prefix = os.path.join(datapath, fn_infix)

"""
Load the surfaces
"""
white_surf = surf_fs_asc.read(white_surf_fn)
pial_surf = surf_fs_asc.read(pial_surf_fn)
intermediate_surf = surf_fs_asc.read(intermediate_surf_fn)

"""
Load the volume geometry information
"""
vg = volgeom.from_nifti_file(epi_ref_fn)

"""
Make a volsurf instance, which is useful for mapping between surface
and volume locations
"""
vs = volsurf.VolSurf(vg, white_surf, pial_surf)

"""
Use all centers and run voxel selection...
"""
nv = intermediate_surf.nvertices
src_ids = range(nv)
# big TODO:  voxsel must be QueryEngine
voxsel = surf_voxel_selection.voxel_selection(vs, radius, src_ids, intermediate_surf)


"""
From now on we simply follow the example in searchlight.py.
First we load and preprocess the data. Note that we use the
mask that came from the voxel selection.
"""
attr = SampleAttributes(os.path.join(datapath, '..', 'attributes.txt'))


dataset = fmri_dataset(
                samples=os.path.join(datapath, '..', 'bold.nii.gz'),
                targets=attr.targets,
                chunks=attr.chunks,
#                mask=epi_ref_fn # but it can come from voxsel.get_masked_voxels
    )


poly_detrend(dataset, polyord=1, chunks_attr='chunks')

dataset = dataset[np.array([l in ['rest', 'house', 'scrambledpix']
                           for l in dataset.targets], dtype='bool')]

zscore(dataset, chunks_attr='chunks', param_est=('targets', ['rest']), dtype='float32')

dataset = dataset[dataset.sa.targets != 'rest']

"""
Define classifier and cross-validation
"""

clf = LinearCSVMC()

cv = CrossValidation(clf, NFoldPartitioner(),
                     errorfx=lambda p, t: np.mean(p == t),
                     enable_ca=['stats'])



"""
The interesting part: define and run the searchlight
"""
qe = SurfaceVerticesQueryEngine(voxsel,
                                # you can optionally add additional
                                # information about each near-disk-voxels
                                add_fa=['center_distances',
                                        'grey_matter_position'])
sl = Searchlight(cv, queryengine=qe, postproc=mean_sample(),
                 # if you care to specify custom subset of vertices
                 # roi_ids=voxsel.keys()
                 )

sl_dset = sl(dataset)


"""
For visualization of results, make a NIML dset that can be viewed
by AFNI. Results are transposed because in NIML, rows correspond
to nodes (features) and columns to datapoints (samples)
"""

surf_sl_dset = dict(data=np.asarray(sl_dset).transpose(),
                  node_indices=center_ids)

dset_fn = searchlight_fn_prefix + '.niml.dset'

afni_niml_dset.write(dset_fn, surf_sl_dset)

print ("To view results, cd to '%s' and run ./%sh_ico%d_seesuma.sh,"
       "click on 'dset', and select %s" %
       (datapath, hemi, lowres_ld, dset_fn))

