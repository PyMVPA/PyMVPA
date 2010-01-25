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

.. index:: searchlight, NIfTI

The example shows how to run a searchlight analysis on the example fMRI dataset
that is shipped with PyMVPA.

As always, we first have to import PyMVPA.
"""

from mvpa.suite import *

"""As searchlight analyses are usually quite expensive in term of computational
ressources, we are going to enable some progress output, to entertain us while
we are waiting."""

# enable debug output for searchlight call
if __debug__:
    debug.active += ["SLC"]

"""The next section simply loads the example dataset and performs some standard
preprocessing steps on it."""

#
# load PyMVPA example dataset
#
# data path
datapath = os.path.join(pymvpa_dataroot, 'demo_blockfmri', 'demo_blockfmri')
attr = SampleAttributes(os.path.join(datapath, 'attributes.txt'))
# later on we want to perform the searchlight analysis in some ROI only
# we add the ROI mask to the dataset as a feature attribute
dataset = fmri_dataset(
                samples=os.path.join(datapath, 'bold.nii.gz'),
                labels=attr.labels,
                chunks=attr.chunks,
                mask=os.path.join(datapath, 'mask_brain.nii.gz'),
                add_fa={'vt_thr_glm': os.path.join(datapath, 'mask_vt.nii.gz')})

#
# preprocessing
#

# do chunkswise linear detrending on dataset
# it is important to do this before selecting subsets of the timeseries!
poly_detrend(dataset, polyord=1, chunks='chunks')

# only use 'rest', 'house' and 'scrambled' samples from dataset
dataset = dataset[N.array([l in ['rest', 'house', 'scrambledpix']
                           for l in dataset.labels], dtype='bool')]

# zscore dataset relative to baseline ('rest') mean
zscore(dataset, perchunk=True, baselinelabels=['rest'], targetdtype='float32')

# remove baseline samples from dataset for final analysis
dataset = dataset[dataset.sa.labels != 'rest']

"""But now for the interesting part: Next we define the measure that shall be
computed for each sphere. Theoretically, this can be anything, but here we
choose to compute a full leave-one-out cross-validation using a linear Nu-SVM
classifier."""

#
# Run Searchlight
#

# choose classifier
clf = LinearNuSVMC()

# setup measure to be computed by Searchlight
# cross-validated mean transfer using an N-fold dataset splitter
cv = CrossValidatedTransferError(TransferError(clf),
                                 NFoldSplitter())

"""Finally, we run the searchlight analysis for three different radii, each
time computing an error for each sphere. To achieve this, we simply use the
:class:`~mvpa.measures.searchlight.Searchlight` class, which takes any
:term:`processing object` and a radius as arguments. The :term:`processing
object` has to compute the intended measure, when called with a dataset. The
:class:`~mvpa.measures.searchlight.Searchlight` object will do nothing more
than generating small datasets for each sphere, feeding it to the processing
object and storing its result.

After the errors are computed for all spheres, the resulting vector is then
mapped back into the original fMRI dataspace and plotted."""

# setup plotting

plot_args = {
    'background' : os.path.join(datapath, 'anat.nii.gz'),
    'background_mask' : os.path.join(datapath, 'mask_brain.nii.gz'),
    'overlay_mask' : os.path.join(datapath, 'mask_vt.nii.gz'),
    'do_stretch_colors' : False,
    'cmap_bg' : 'gray',
    'cmap_overlay' : 'autumn', # YlOrRd_r # P.cm.autumn
    'interactive' : cfg.getboolean('examples', 'interactive', True),
    }

# get ids of features that have a nonzero value
center_ids = dataset.fa.vt_thr_glm.nonzero()[0]

for radius in [0, 1, 3]:
    # tell which one we are doing
    print "Running searchlight with radius: %i ..." % (radius)

    # setup Searchlight with a custom radius
    # on multi-core machines try increasing the `nproc` argument
    # to utilize more than one core
    sl = sphere_searchlight(cv, radius=radius, space='voxel_indices',
                            center_ids=center_ids,
                            nproc=2, mapper=mean_sample())

    # to increase efficiency, we strip all unnecessary attributes from the
    # dataset before we hand it over to the searchlight
    ds = dataset.copy(deep=False,
                      sa=['labels', 'chunks'], fa=['voxel_indices'], a=['mapper'])
    # run searchlight on example dataset and retrieve error map
    sl_map = sl(ds)
    # let's plot accuracy
    sl_map.samples *= -1
    sl_map.samples += 1

    # results back in fMRI space
    niftiresults = map2nifti(sl_map, imghdr=dataset.a.imghdr)

    fig = P.figure(figsize=(12, 4), facecolor='white')
    subfig = plot_lightbox(overlay=niftiresults,
                           vlim=(0.5, None), slices=range(23,31),
                           fig=fig, **plot_args)
    P.title('Accuracy distribution for radius %i' % radius)


if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    P.show()

"""
The following figures show the resulting accuracy maps for the slices covered
by the ventral temporal cortex mask. Note that each voxel value represents the
accuracy of a sphere centered around this voxel.

.. figure:: ../pics/ex_searchlight_vt_r0.*
   :align: center

   Searchlight (single element; univariate) accuracy maps for binary
   classification *house* vs. *scrambledpix*.

.. figure:: ../pics/ex_searchlight_vt_r1.*
   :align: center

   Searchlight (sphere of neighboring voxels; 9 elements) accuracy maps for
   binary classification *house* vs.  *scrambledpix*.

.. figure:: ../pics/ex_searchlight_vt_r3.*
   :align: center

   Searchlight (radius 3 elements; 123 voxels) accuracy maps for binary
   classification *house* vs.  *scrambledpix*.

With radius 0 (only the center voxel is part of the part the sphere) there is a
clear distinction between two distributions. The *chance distribution*,
relatively symetric and centered around the expected chance-performance at 50%.
The second distribution, presumambly of voxels with univariate signal, is nicely
segregated from that. Increasing the searchlight size significantly blurrs the
accuracy map, but also lead to an increase in classification accuracy.
"""
