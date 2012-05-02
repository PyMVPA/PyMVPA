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
Surface-based voxel selection and searchlight example
=====================================================

.. index:: searchlight, cross-validation

This scripts shows how to use a searchlight (:ref:`Kriegeskorte et al. (2006)
<KGB06>`) on the cortical surface. For more details about the procedure,
explanation of the algorithm, and a Matlab implementation,
see http://surfing.sourceforge.net.

Anatomical preprocessing can be done in various ways; the method describes here,
uses Freesurfer and AFNI/SUMA (for the pipeline, see the documentation at
http://surfing.sourceforge.net for details). Specifically, the command used to
produce the surfaces used here is::

  python mvpa2/misc/surfing/anatpreproc.py -e bold_mean.nii -A -r surfing \
  -d subj1/subj1
         
where bold_mean.nii is an averaged bold image from bold.nii.gz, subj1/subj1
the output directory from Freesurfer's recon-all, and surfing the directory
in which aligned volumes and surfaces are being stored.

The example here consists of two steps:
(1) voxel selection using cortical surfaces (which associates, with each node
on the surface, a set of neighboring voxels);
(2) running a searchlight based on the voxel selection

In this example we use the Haxby 2001 tutorial dataset subj1-2010.01.14.tar.gz  

References
----------
NN Oosterhof, T Wiestler, PE Downing (2011). A comparison of volume-based
and surface-based multi-voxel pattern analysis. Neuroimage, 56(2), pp. 593-600

'Surfing' toolbox: http://surfing.sourceforge.net
(and the associated documentation)
"""

"""First import all the ingredients"""

#from mvpa2.tutorial_suite import *
from mvpa2.suite import *

# not needed due to above import * from the suite
#from mvpa2.measures.base import CrossValidation
#from mvpa2.generators.partition import NFoldPartitioner
#from mvpa2.mappers.fx import mean_sample
#from mvpa2.datasets.mri import fmri_dataset

from mvpa2.misc.surfing import utils, volsurf, afni_niml_dset, afni_niml, \
     sparse_attributes, surf_fs_asc, volgeom, surf_voxel_selection

"""To store voxel selection results (for later re-use), we use pickle"""
import cPickle as pickle
import nibabel as ni
import numpy as np

if __debug__:
    from mvpa2.base import debug
    if not "SVS" in debug.registered:
        debug.register('SVS',
                       "Surface-based voxel selection (a.k.a. 'surfing')")
    debug.active += ["SVS", "SLC"]

"""
We start with defining all files we use as input.
For now, we use the fingerdata dataset available from
http://surfing.sourceforge.net. In the experiment, the participant pressed with the
index finger during some trials and middle finger during other trials.
After preprocessing and a GLM, this dataset contains 32 volumes with odd volumes
corresponding to the index finger and even volumes to the middle finger.
"""

datadir = os.path.join(utils._get_fingerdata_dir(), 'pyref')
"""
Use an arbitrary volume from the dataset. We only use this to get the
volume geometry (voxel size and voxel-to-world mapping)
"""
epi_ref_fn = os.path.join(datadir, '..', 'glm', 'rall_vol00.nii')

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
highres_ld = 64

pial_surf_fn = os.path.join(datadir, "ico%d_%sh.pial_al.asc"
                                     % (highres_ld, hemi))
white_surf_fn = os.path.join(datadir, "ico%d_%sh.smoothwm_al.asc"
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
lowres_ld = 32

intermediate_surf_fn = os.path.join(datadir, "ico%d_%sh.intermediate_al.asc"
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
radius = 100

"""
Set the prefix for output
"""
voxel_selection_fn_prefix = os.path.join(datadir, "voxsel_ico%d_%sh"
                                                  % (lowres_ld, hemi))
searchlight_fn_prefix = os.path.join(datadir, "searchlight_ico%d_%sh"
                                              % (lowres_ld, hemi))

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
Define which nodes to use as a searchlight center. In this example, we
use all nodes.
"""
node_count = intermediate_surf.nv()

src_ids = range(node_count)
"""
Make a volsurf instance, which is useful for mapping between surface
and volume locations
"""
vs = volsurf.VolSurf(vg, white_surf, pial_surf)

"""
Run voxel selection...
"""
voxsel = surf_voxel_selection.voxel_selection(vs, intermediate_surf, radius, src_ids)\

print "Voxel selection results:"
print voxsel
"""
>>>SparseAttributes with 2562 entries, 3 labels (['lin_vox_idxs', 'grey_matter_position', 'center_distances'])
>>>General attributes: ['volgeom']
...and save voxel selection results.
"""
sparse_attributes.to_file(voxel_selection_fn_prefix + ".pickle", voxsel)
"""
The following just generates some files as a sanity check; they are not
necessary for a typical voxel selection pipeline.

(1) Count how often each voxel was selected in a searchlight.
We use linear indexing (akin to FlattenMapper), then reshape
our data to make it 3D
"""
vg = voxsel.a['volgeom']
voxel_count = vg.nv()

vol_data_lin = np.zeros((voxel_count, 1)) # use linear voxel indexing

vol_map = voxsel.get_attr_mapping('lin_vox_idxs')
for node_idx, voxel_idxs in vol_map.iteritems():
    if voxel_idxs is not None:
        vol_data_lin[voxel_idxs] += 1

vol_data_sub = np.reshape(vol_data_lin, vg.shape()) # 3D shape

img = ni.Nifti1Image(vol_data_sub, vg.affine())
img.to_filename(voxel_selection_fn_prefix + ".nii")

"""
Generate another mapping from node indices to the distances of
each voxel to the center. Here we are concerned with the maximal
distance (i.e. radius in millimeters) of each searchlight).
"""
surf_data = np.zeros((node_count, 2))

radius_map = voxsel.get_attr_mapping('center_distances')
for node_idx, distances in radius_map.iteritems():
    if distances is not None:
        surf_data[node_idx, 0] = max(distances)
        surf_data[node_idx, 1] = sum(distances) / len(distances)


surf_dset = dict(data=surf_data, labels=["max_d", "mean_d"])
afni_niml_dset.write(voxel_selection_fn_prefix + "_full.niml.dset", surf_dset)
"""
Do exactly the same thing, but now we write a sparse dataset that only contains
data for nodes that have voxels associated with them (some nodes are outside
the functional volume and have no voxels associated with them).
"""
nonempty_nodes = voxsel.keys()
nonempty_node_count = len(nonempty_nodes)

surf_data_sparse = np.zeros((nonempty_node_count, 2))

radius_map = voxsel.get_attr_mapping('center_distances')
for i, node_idx in enumerate(nonempty_nodes):
    distances = radius_map[node_idx]

    surf_data_sparse[i, 0] = max(distances)
    surf_data_sparse[i, 1] = sum(distances) / len(distances)


surf_dset_sparse = dict(data=surf_data_sparse, labels=["max_d", "mean_d"], node_indices=nonempty_nodes)
afni_niml_dset.write(voxel_selection_fn_prefix + "_sparse.niml.dset", surf_dset_sparse)

"""
Delete the voxel selection results, then reload them from disk
"""
del voxsel

print "Loading voxel selection data from disk:"
voxsel = sparse_attributes.from_file(voxel_selection_fn_prefix + ".pickle")
print voxsel

"""
Load functional data for running the searchlight,
and make it into an fmri_dataset
"""
epi_data_fn = '%s/../glm/rall_4D_nibabel.nii' % datadir
nsamples = 32
targetnames = ['index', 'middle']
targets = [targetnames[i % 2] for i in xrange(nsamples)]
chunks = [i / 4 for i in xrange(nsamples)]
ds = fmri_dataset(samples=epi_data_fn, targets=targets,
                chunks=chunks)


"""
As in the volume-based searchlight example, set up
a classifier and cross-validation
"""
clf = LinearCSVMC()
cvte = CrossValidation(clf, NFoldPartitioner(),
                     errorfx=lambda p, t: np.mean(p == t),
                     enable_ca=['stats'])



"""
Set up the searchlight
"""
nbrhood = voxsel.get_neighborhood()
center_ids = voxsel.keys() # these are only nodes with voxels associated

searchlight = sparse_attributes.searchlight(cvte, nbrhood,
                                           postproc=mean_sample(),
                                           center_ids=center_ids)
"""
As in the volume-based searchlight example, make a copy
of the dataset with only necessary attributes
"""
sds = ds.copy(deep=False,
                   sa=['targets', 'chunks'],
                   fa=['voxel_indices'],
                   a=['mapper'])


"""
Run the searchlight
"""
sl_dset = searchlight(sds)

"""
For visualization of results, make a NIML dset that can be viewed
by AFNI. Results are transposed because in NIML, rows correspond
to nodes (features) and columns to datapoints (samples)
"""

surf_sl_dset = dict(data=np.asarray(sl_dset).transpose(),
                  node_indices=center_ids)

afni_niml_dset.write(searchlight_fn_prefix + "_%dvx.niml.dset" % radius,
                     surf_sl_dset)

print ("To view results, cd to '%s' and run ./%sh_ico%d_seesuma.sh" %
       (datadir, hemi, lowres_ld))
