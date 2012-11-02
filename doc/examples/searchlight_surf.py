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
Surface-based searchlight on fMRI data
========================

.. index:: SurfaceSearchlight

This example is adapted from doc/searchlight.py. It employs a surface-based
searchlight as described in :ref:`Oosterhof et al. (2011) <OWD+11>`, with a 
minor difference that distances are currently computed using a Dijkstra 
distance metric rather than a geodesic one.

More details can be found at the `Surfing <http://surfing.sourceforge.net>`_  
website. 

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

"""Define surface and volume data paths"""

datapath = os.path.join(pymvpa_datadbroot,
                        'tutorial_data', 'tutorial_data', 'data', 'surfing')

"""Define functional data volume filename"""

epi_fn = os.path.join(datapath, '..', 'bold.nii.gz')

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
Define the surface on which the nodes are centers of the searchlight. In this
example a coarser surface (fewer nodes) is employed. 

It is crucial here that highres_ld is a multiple of lowres_ld, so that
all nodes in the low-res surface have a corresponding node (i.e., with the same,
or almost the same, spatial coordinate) on the high-res surface.

Choice of lowres_ld and highres_ld is somewhat arbitrary and always application
specific. For highres_ld a value of at least 64 may be advisable as this
ensures enough anatomical detail is available to select voxels in the grey
matter accurately. 
"""

lowres_ld = 8 # 16, 32 or 64 is reasonable. 8 is really fast

intermediate_surf_fn = os.path.join(datapath, "ico%d_%sh.intermediate_al.asc"
                                             % (lowres_ld, hemi))

"""
Radius is specified as either an int (referring to a fixed number of voxels
across searchlights, with a variable radius in millimeters (or whatever unit
is used in the files that define the surfaces), or a float (referring to the
radius in millimeters, with a variable number of voxels).

Note that "a fixed number of voxels" in this context actually means an
approximation, in that on average that number of voxels is selected but the
actual number will vary slightly (typically in the range +/- 2 voxels)
"""

radius = 100

"""
Set the prefixes for output
"""

fn_infix = 'ico%d_%sh_%dvx' % (lowres_ld, hemi, radius)
searchlight_fn_prefix = os.path.join(datapath, fn_infix)


"""We're all set to go to create a query engine to determine for
each node which voxels are near it.

As a reminder, the only essential values we have set are the
filenames of three surfaces (high-res inner and outer,
and low-res intermediate), and the searchlight radius.

Note that setting the low-res intermediate surface can be omitted
(i.e. set it to None), in which case it is computed as the average from the
high-res outer and inner. The searchlight would then be based on
a high-res intermediate surface with a lot of nodes, which means that it takes
longer to run the searchlight.
"""

qe = disc_surface_queryengine(
    radius,
    epi_fn,
    white_surf_fn, pial_surf_fn, intermediate_surf_fn)

'''As in the example in searchlight.py, define cross-validation
using an (SVM) classifier
'''

clf = LinearCSVMC()

cv = CrossValidation(clf, NFoldPartitioner(),
                     errorfx=lambda p, t: np.mean(p == t),
                     enable_ca=['stats'])



"""
Combining the query-engine and the cross-validation defines the 
searchlight. The postproc-step averages the classification accuracies
in each cross-validation fold to a single overall classification accuracy.
"""

sl = Searchlight(cv, queryengine=qe, postproc=mean_sample())



'''
Next step is to load the functional data. But before that we can reduce
memory requirements significantly by considering which voxels to load.
Since we will only use voxels that were selected at least once by the 
voxel selection step, a mask is taken from the voxel selection results 
and used when loading the functional data 
'''

mask = qe.voxsel.get_mask()

"""
From now on we simply follow the example in searchlight.py.
First we load and preprocess the data. Note that we use the
mask that came from the voxel selection.
"""

attr = SampleAttributes(os.path.join(datapath, '..', 'attributes.txt'))

dataset = fmri_dataset(
                samples=epi_fn,
                targets=attr.targets,
                chunks=attr.chunks,
                mask=mask)


poly_detrend(dataset, polyord=1, chunks_attr='chunks')

dataset = dataset[np.array([l in ['rest', 'house', 'scrambledpix']
                           for l in dataset.targets], dtype='bool')]

zscore(dataset, chunks_attr='chunks', param_est=('targets', ['rest']),
        dtype='float32')

dataset = dataset[dataset.sa.targets != 'rest']

"""
Run the searchlight on the dataset. 
"""

sl_dset = sl(dataset)


"""
For visualization of results, make a NIML dset that can be viewed
by AFNI's SUMA. Results are transposed because in NIML, rows correspond
to nodes (features) and columns to datapoints (samples).

In certain cases (though not in this example) some nodes may have no
voxels associated with them (in case of partial brain coverage, for example).
Therefore center_ids is based on the keys used in the query engine. 
"""

center_ids = qe.ids

surf_sl_dset = dict(data=np.asarray(sl_dset).transpose(),
                    node_indices=center_ids,
                    labels=['HOUSvsSCRM'])

dset_fn = searchlight_fn_prefix + '.niml.dset'
from mvpa2.support.nibabel import afni_niml_dset

afni_niml_dset.write(dset_fn, surf_sl_dset)

print ("To view results, cd to '%s' and run ./%sh_ico%d_runsuma.sh,"
       "click on 'dset', and select %s" %
       (datapath, hemi, lowres_ld, dset_fn))

