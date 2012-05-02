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

"""
Also load additional modules"
"""
from mvpa2.misc.surfing import utils, volsurf, afni_niml_dset, afni_niml, \
     sparse_attributes, surf_fs_asc, volgeom, surf_voxel_selection

"""To store voxel selection results (for later re-use), we use pickle"""
import cPickle as pickle

"""Use nibabel and numpy"""
import nibabel as ni
import numpy as np

"""As searchlight analyses are usually quite expensive in term of computational
resources, we are going to enable some progress output to entertain us while
we are waiting."""

# enable debug output for searchlight call
if __debug__:
    from mvpa2.base import debug
    if not "SVS" in debug.registered:
        debug.register('SVS',
                       "Surface-based voxel selection (a.k.a. 'surfing')")
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
                        'tutorial_data', 'tutorial_data', 'subj1', 'surfing')

"""First set up surface stuff"""
epi_ref_fn = os.path.join(datapath, 'bold_mean.nii')

"""
We're concerned with the left hemisphere only
"""
hemi = 'l'

"""
Surfaces that enclose the grey matter. These are used for voxel selection.
These surfaces were resampled using AFNI's MapIcosahedron; ld refers to
the number of linear divisions of the 'large' triangles of the original
icosahedron (ld=x means there are 10*x**2+2 nodes and 20*x**2 triangles).
"""
highres_ld=32 # was 64

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
lowres_ld=32 # was 32

intermediate_surf_fn = os.path.join(datapath, "ico%d_%sh.intermediate_al.asc"
                                             % (lowres_ld, hemi))

"""
Radius is specified as either an int (referring to a fixed number of voxels
across searchlights, with a variable radius in milimeters (or whatever unit
is used in the files that define the surfaces), or a float (referring to the
radius in millimeters, with a variable number of voxels).

Note that "a fixed number of voxels" in this context actually means an
approximation, in that on average that number of voxels is selected but the
actual number will vary slightly
"""
radius=100


"""
Set the prefixes for output
"""
fn_infix='ico%d_%sh_%dvx' % (lowres_ld, hemi, radius)
voxel_selection_fn_prefix = os.path.join(datapath, fn_infix)
searchlight_fn_prefix = os.path.join(datapath, fn_infix)

"""
Load the surfaces
"""
white_surf = surf_fs_asc.read(white_surf_fn)
pial_surf = surf_fs_asc.read(pial_surf_fn)
intermediate_surf = surf_fs_asc.read(intermediate_surf_fn)

"""
Load the volume geamotry information
"""
vg = volgeom.from_nifti_file(epi_ref_fn)

"""
Make a volsurf instance, which is useful for mapping between surface
and volume locations
"""
vs=volsurf.VolSurf(vg,white_surf,pial_surf)
print vs
"""
Run voxel selection...
"""
nv=intermediate_surf.nv()
src_ids=range(nv)
voxsel=surf_voxel_selection.voxel_selection(vs, intermediate_surf, radius, src_ids)

print "Voxel selection results:"
print voxsel

voxel_ids_label='lin_vox_idxs'
#small_voxsel, small_keys, mask_img=voxsel.minimal_mask_mapping(voxel_ids_label)
#nbrhood = voxsel.get_neighborhood()
#mm=voxsel.mask_key_mapping()

center_ids = voxsel.keys()
mask=voxsel.get_niftiimage_mask()
nbrhood = voxsel.get_neighborhood(mask)


# source of class targets and chunks definitions
attr = ColumnData(os.path.join(datapath, '..', 'labels.txt'))

dataset = fmri_dataset(
                samples=os.path.join(datapath, '..', 'bold.nii.gz'),
                targets=attr.labels,
                chunks=attr.chunks,
                mask=mask)

print "dataset loaded"                

"""The dataset is now loaded and contains all brain voxels as features, and all
volumes as samples. To precondition this data for the intended analysis we have
to perform a few preprocessing steps (please note that the data was already
motion-corrected). The first step is a chunk-wise (run-wise) removal of linear
trends, typically caused by the acquisition equipment."""

poly_detrend(dataset, polyord=1, chunks_attr='chunks')

print "detrended"

"""Now that the detrending is done, we can remove parts of the timeseries we
are not interested in. For this example we are only considering volumes acquired
during a stimulation block with images of houses and scrambled pictures, as well
as rest periods (for now). It is important to perform the detrending before
this selection, as otherwise the equal spacing of fMRI volumes is no longer
guaranteed."""

dataset = dataset[np.array([l in ['rest', 'house', 'scrambledpix']
                           for l in dataset.targets], dtype='bool')]

"""The final preprocessing step is data-normalization. This is a required step
for many classification algorithms. It scales all features (voxels)
into approximately the same range and removes the mean. In this example, we
perform a chunk-wise normalization and compute standard deviation and mean for
z-scoring based on the volumes corresponding to rest periods in the experiment.
The resulting features could be interpreted as being voxel salience relative
to 'rest'."""

zscore(dataset, chunks_attr='chunks', param_est=('targets', ['rest']), dtype='float32')

print "zscores dataset"

"""After normalization is completed, we no longer need the 'rest'-samples and
remove them."""

dataset = dataset[dataset.sa.targets != 'rest']

print "Masked dataset"


"""But now for the interesting part: Next we define the measure that shall be
computed for each sphere. Theoretically, this can be anything, but here we
choose to compute a full leave-one-out cross-validation using a linear Nu-SVM
classifier."""

# choose classifier
clf = LinearNuSVMC()

# setup measure to be computed by Searchlight
# cross-validated mean transfer using an N-fold dataset splitter
#cv = CrossValidation(clf, NFoldPartitioner())

cv = CrossValidation(clf, NFoldPartitioner(),
                     errorfx=lambda p, t: np.mean(p == t),
                     enable_ca=['stats'])



print "Going to run searchlight"
searchlight = sparse_attributes.searchlight(cv, nbrhood,
                                           postproc=mean_sample(),
                                           center_ids=center_ids)

sl_dset = searchlight(dataset)

"""
For visualization of results, make a NIML dset that can be viewed
by AFNI. Results are transposed because in NIML, rows correspond
to nodes (features) and columns to datapoints (samples)
"""

surf_sl_dset = dict(data=np.asarray(sl_dset).transpose(),
                  node_indices=center_ids)

afni_niml_dset.write(searchlight_fn_prefix + '.niml.dset', surf_sl_dset)

print ("To view results, cd to '%s' and run ./%sh_ico%d_seesuma.sh" %
       (datapath, hemi, lowres_ld))

