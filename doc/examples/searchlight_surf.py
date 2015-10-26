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
======================================

.. index:: surface, searchlight, cross-validation

This example employs a surface-based searchlight as described in
:ref:`Oosterhof et al. (2011) <OWD+11>` (with a minor difference that distances
are currently computed using a Dijkstra distance metric rather than a geodesic
one). For more details, see the `Surfing <http://surfing.sourceforge.net>`_
website.

Surfaces used in this example are available in the tutorial dataset files;
either the tutorial_data_surf_minimal or tutorial_data_surf_complete version.
The surfaces were reconstructed using FreeSurfer and
subsequently preprocessed with AFNI and SUMA using the
pymvpa2-prep-afni-surf wrapper script in PyMVPA's 'bin' directory, which
resamples the surfaces to standard topologies (with different resolutions)
using MapIcosehedron, aligns surfaces to a reference functional volume, and
merges left and right hemispheres into single surface files. A more detailed
description of the steps that this script takes is provided in the
documentation on the `Surfing <http://surfing.sourceforge.net>`_
website.

If you use the surface-based searchlight code for a publication, please cite
both :ref:`PyMVPA (2009) <HHS+09a>` and :ref:`Oosterhof et al. (2011) <OWD+11>`.


As always, we first have to import PyMVPA.
"""

from mvpa2.suite import *
from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.base.hdf5 import h5save, h5load
"""As searchlight analyses are usually quite expensive in term of computational
resources, we are going to enable some progress output to entertain us while
we are waiting."""

if __debug__:
    from mvpa2.base import debug
    debug.active += ["SVS", "SLC"]

"""Define surface and volume data paths:"""

rootpath = os.path.join(pymvpa_datadbroot,
                        'tutorial_data', 'tutorial_data')

datapath = os.path.join(rootpath, 'haxby2001')
surfpath = os.path.join(rootpath, 'suma_surfaces')

"""Define functional data volume filename:"""

epi_fn = os.path.join(datapath, 'sub001', 'BOLD', 'task001_run001', 'bold.nii.gz')

"""
In this example we are concerned with the left hemisphere only.
(Other possible values are 'r' for the right hemisphere and 'm' for merged
hemispheres; the latter contains the nodes from the left and right
hemispheres in a single file. Both the 'r' and 'm' options require
the tutorial_data_surf_complete tutorial data.)
"""

hemi = 'l'

"""
Define the surfaces that enclose the grey matter, which are used to
delineate the grey matter. The pial surface is the 'outside' border of the
grey matter; the white surface is the 'inside' border.

The surfaces in this example were resampled using AFNI's MapIcosahedron
(for more details, see the top of this script). ld refers to the number of
linear divisions of the twenty 'large' triangles of the original
icosahedron; ld=x means there are 10*x**2+2 nodes (a.k.a. vertices)
and 20*x**2 triangles (a.k.a. faces).
"""

highres_ld = 128 # 64 or 128 is reasonable

pial_surf_fn = os.path.join(surfpath, "ico%d_%sh.pial_al.asc"
                                     % (highres_ld, hemi))
white_surf_fn = os.path.join(surfpath, "ico%d_%sh.smoothwm_al.asc"
                                      % (highres_ld, hemi))

"""
Define the surface on which the nodes are centers of the searchlight. This
surface should be an 'intermediate' surface, which is formed by the
node-wise average spatial coordinates of the inner (white) and outer (pial)
surfaces.

In this example a surface coarser (fewer nodes) than the grey matter-enclosing
surfaces is employed. This reduces the number of searchlights and therefore
the script's execution time. Of course one could also use a surface that has
the same number of nodes as the grey-matter enclosing surfaces; this is
actually the default and used when souce_surf_fn (assigned below) is set
to None.

It is required that highres_ld is an integer multiple of lowres_ld, so that
all nodes in the low-res surface have a corresponding node (i.e., with the
same, or almost the same, spatial coordinate) on the high-res surface.

Choice of lowres_ld and highres_ld is somewhat arbitrary and a trade-off
between spatial specificity and execution speed. For highres_ld a value of at
least 64 is be advisable as this ensures enough anatomical detail is available
to select voxels in the grey matter accurately. Typical values for lowres_ld
range from 8 to 64.

Note that the data in tutorial_data_surf_minimal only contains
all necessary surfaces for visualization for lowres_ld=16. For other values
of lowres_ld (4, 8, 32, 64 and 128) the surfaces in
tutorial_data_surf_complete are required.

"""

lowres_ld = 16 # 16, 32 or 64 is reasonable. 4 and 8 are really fast

source_surf_fn = os.path.join(surfpath, "ico%d_%sh.intermediate_al.asc"
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


"""We're all set to go to create a query engine that performs 'voxel
selection', that is determines, for each node, which voxels are near it
(that is, in the corresponding searchlight disc).

As a reminder, the only essential values we have set so far are the
filenames of three surfaces (high-res inner and outer,
and low-res source surface), the functional volume, and the searchlight
radius.

Note that if the functional data was preprocessed and subsequently masked,
voxel selection should take into account this mask. To do so, the
instantiation of the query engine below takes an optional argument
'volume_mask' (which can be a PyMVPA dataset, a numpy array, a Nifti
volume, or a string representing the file name of a Nifti volume). It is,
however, recommended to *not* mask the functional data prior to voxel
selection, because the voxel selection uses (implicitly) a mask based on the
grey-matter enclosing surfaces already, and this mask is assumed to be more
precise than typical volume-based masking implementations.

Also note that, as described above, the argument defining the low-res source
surface can be omitted, in which case it is computed as the node-wise
average of the white and pial surface.)
"""

qe = disc_surface_queryengine(radius, epi_fn,
                              white_surf_fn, pial_surf_fn,
                              source_surf_fn)

"""
Voxel selection is now completed; each node has been assigned a list of
linear voxel indices in the searchlight. These result are stored in
'qe.voxsel' and can be saved with h5save for later re-use.

(Linear voxel indices mean that each voxel is indexed by a value between
0 (inclusive) and N (exclusive), where N is the number of voxels in the
volume (N = NX * NY * NZ, where NX, NY and NZ are the number of voxels in
the three spatial dimensions). For certain analyses one may want to index
voxels by 'sub indices' (triples (i,j,k) with 0<=i<NX, 0<=j<=NY,
and 0<=k<NZ) or spatial coordinates; conversions amongst
linear and sub indices and spatial coordinates is provided by
functions in the  VolGeom (volume geometry) instance stored in
'qe.voxsel.volgeom'.)

From now on we follow the example as in doc/examples/searchlight.py.

First, cross-validation is defined using a (SVM) classifier.
"""

clf = LinearCSVMC()

cv = CrossValidation(clf, NFoldPartitioner(),
                     errorfx=lambda p, t: np.mean(p == t),
                     enable_ca=['stats'])

"""
Set the roi_ids, that is the node indices that serve as searchlight
center. In this example it is set to None, meaning that all nodes are used
as a searchlight center. It is also possible to restrict the nodes that serve
as a searchlight center: setting roi_ids=np.arange(400,800) means that only
nodes in the range from 400 (inclusive) to 800 (exclusive) are used as a
searchlight center, and the result would be a partial brain map.
"""

roi_ids = None

"""
Combining the query-engine and the cross-validation defines the
searchlight. The postproc-step averages the classification accuracies
in each cross-validation fold to a single overall classification accuracy.

Because roi_ids is None is this example it could be omitted - it is only
included for instructive purposes.
"""

sl = Searchlight(cv, queryengine=qe, postproc=mean_sample(), roi_ids=roi_ids)



'''
In the next step the functional data is loaded. We can reduce
memory requirements significantly by considering which voxels to load:
since the searchlight analysis will only use data from voxels that
were selected (at least once) by the voxel selection step, a mask is
derived from the voxel selection results and used when loading the
functional data.
'''

mask = qe.voxsel.get_mask()

"""
Load the functional data for subject 1 and the condition model 1 in the
dataset (object viewing with 8 object categories). Note that we use the
mask that came from the voxel selection.
"""

model = 1
subject = 1
of = OpenFMRIDataset(datapath)
dataset = of.get_model_bold_dataset(model, subject,
                                    noinfolabel='rest', mask=mask)

"""
Apply some typical preprocessing steps
"""

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
Searchlight results are now stored in sl_dset. As sl_dset is just like
any other PyMVPA dataset, it can be stored with h5save for future use.

The remainder of this example provides a data file that
can be visualized using AFNI's SUMA. This is achieved by storing the dataset
as an NIML (NeuroImaging Markup Language) dataset that can be viewed by
AFNI's SUMA. sl_dset contains a feature attribute 'center_ids' that is
automagically used to define the node indices of the searchlight centers in
this NIML dataset.

Note that this conversion will not preserve all information in sl_dset but
only the samples and (feature, sample, dataset) attributes that behave
like arrays or strings or scalars. For example, in this example sl_dset has a
dataset attribute 'mapper' which is not stored in the NIML dataset (and
a warning message is printed during the conversion, which can be ignored
savely). As mentioned above, using h5save will preserve this information
(but its output cannot be viewed in SUMA).

Before saving the dataset, first the labels are set for each sample (in
this case, only one) so that they show up in SUMA.
"""

sl_dset.sa['labels'] = ['HOUSvsSCRM']

"""
Set the filename for output.
Searchlight results are stored in the surface directory for easy
visualization. Finally print an informative message on how the
generated data can be visualized using SUMA.
"""

# save as NIML dataset
fn = 'ico%d-%d_%sh_%dvx.niml.dset' % (lowres_ld, highres_ld, hemi, radius)
path_fn = os.path.join(surfpath, fn)
niml.write(path_fn, sl_dset)

# save as GIFTI
if externals.exists('nibabel'):
    fn = 'ico%d-%d_%sh_%dvx.func.gii' % (lowres_ld, highres_ld, hemi, radius)
    path_fn = os.path.join(surfpath, fn)
    map2gifti(sl_dset, path_fn)


print ("To view results in SUMA, cd to '%s', run 'suma -spec "
      "%sh_ico%d_al.spec', press ctrl+s, "
       "click on 'Load Dset', and select %s" %
       (surfpath, hemi, lowres_ld, fn))
