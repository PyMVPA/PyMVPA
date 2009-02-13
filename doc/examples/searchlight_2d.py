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
attr = SampleAttributes(os.path.join(pymvpa_dataroot, 'attributes.txt'))
dataset = NiftiDataset(samples=os.path.join(pymvpa_dataroot, 'bold.nii.gz'),
                       labels=attr.labels,
                       chunks=attr.chunks,
                       mask=os.path.join(pymvpa_dataroot, 'mask.nii.gz'))

#
# preprocessing
#

# do chunkswise linear detrending on dataset
detrend(dataset, perchunk=True, model='linear')

# only use 'rest', 'house' and 'scrambled' samples from dataset
dataset = dataset.selectSamples(
                N.array([ l in [0,2,6] for l in dataset.labels],
                dtype='bool'))

# zscore dataset relative to baseline ('rest') mean
zscore(dataset, perchunk=True, baselinelabels=[0], targetdtype='float32')

# remove baseline samples from dataset for final analysis
dataset = dataset.selectSamples(N.array([l != 0 for l in dataset.labels],
                                        dtype='bool'))

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
# cross-validated mean transfer using an odd-even dataset splitter
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
P.figure(figsize=(12,4))


for radius in [1,5,10]:
    # tell which one we are doing
    print "Running searchlight with radius: %i ..." % (radius)

    # setup Searchlight with a custom radius
    # radius has to be in the same unit as the nifti file's pixdim
    # property.
    sl = Searchlight(cv, radius=radius)

    # run searchlight on example dataset and retrieve error map
    sl_map = sl(dataset)

    # map sensitivity map into original dataspace
    orig_sl_map = dataset.mapReverse(N.array(sl_map))
    masked_orig_sl_map = N.ma.masked_array(orig_sl_map,
                                           mask=orig_sl_map == 0)

    # make a new subplot for each classifier
    fig += 1
    P.subplot(1,3,fig)

    P.title('Radius %i' % radius)

    P.imshow(masked_orig_sl_map[0],
             interpolation='nearest',
             aspect=1.25,
             cmap=P.cm.autumn)
    P.clim(0.5, 0.65)
    P.colorbar(shrink=0.6)


if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    P.show()
