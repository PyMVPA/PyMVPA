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
attr = SampleAttributes(os.path.join(pymvpa_dataroot,
                        'attributes_literal.txt'))
dataset = fmri_dataset(samples=os.path.join(pymvpa_dataroot, 'bold.nii.gz'),
                       labels=attr.labels,
                       chunks=attr.chunks,
                       mask=os.path.join(pymvpa_dataroot, 'mask.nii.gz'))

#
# preprocessing
#

# do chunkswise linear detrending on dataset
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
fig = 0
P.figure(figsize=(12, 4))


for radius in [0, 1, 3]:
    # tell which one we are doing
    print "Running searchlight with radius: %i ..." % (radius)

    # setup Searchlight with a custom radius
    # on multi-core machines try increasing the `nproc` argument
    # to utilize more than one core
    sl = sphere_searchlight(cv, radius=radius, space='voxel_indices',
                            nproc=1, mapper=mean_sample())

    # to increase efficiency, we strip all unnecessary attributes from the
    # dataset before we hand it over to the searchlight
    ds = dataset.copy(deep=False,
                      sa=['labels', 'chunks'], fa=['voxel_indices'], a=[])
    # run searchlight on example dataset and retrieve error map
    sl_map = sl(ds)
    # map sensitivity map into original dataspace
    orig_sl_map = dataset.mapper.reverse(sl_map)
    masked_orig_sl_map = N.ma.masked_array(orig_sl_map,
                                           mask=orig_sl_map == 0)

    # make a new subplot for each classifier
    fig += 1
    P.subplot(1,3,fig)

    P.title('Radius %i' % radius)
    # plot 1-results, since we get errors
    P.imshow(1 - masked_orig_sl_map.squeeze(),
             interpolation='nearest',
             aspect=1.25,
             cmap=P.cm.autumn)
    P.clim(0.5, 1.0)
    P.colorbar(shrink=0.6)


if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    P.show()
