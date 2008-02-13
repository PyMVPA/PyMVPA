#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Example demonstrating a searchlight analysis on an fMRI dataset"""

import numpy as N
import pylab as P

# local imports
from mvpa.misc.iohelpers import SampleAttributes
from mvpa.datasets.niftidataset import NiftiDataset
from mvpa.datasets.misc import zscore
from mvpa.misc.signal import detrend
from mvpa.clfs.knn import kNN
from mvpa.clfs.svm import LinearNuSVMC
from mvpa.clfs.transerror import TransferError
from mvpa.datasets.splitter import NFoldSplitter, OddEvenSplitter
from mvpa.algorithms.clfcrossval import ClfCrossValidation
from mvpa.algorithms.searchlight import Searchlight
from mvpa.misc import debug

# enable debug output for searchlight call
debug.active += ["SLC"]


#
# load PyMVPA example dataset
#
attr = SampleAttributes('data/attributes.txt')
dataset = NiftiDataset(samples='data/bold.nii.gz',
                       labels=attr.labels,
                       chunks=attr.chunks,
                       mask='data/mask.nii.gz')

#
# preprocessing
#

# do chunkswise linear detrending on dataset
detrend(dataset, perchunk=True, type='linear')

# only use 'rest', 'house' and 'scrambled' samples from dataset
dataset = dataset.selectSamples(
                N.array([ l in [0,2,6] for l in dataset.labels], dtype='bool'))

# zscore dataset relative to baseline ('rest') mean
zscore(dataset, perchunk=True, baselinelabels=[0], targetdtype='float32')

# remove baseline samples from dataset for final analysis
dataset = dataset.selectSamples(N.array([l != 0 for l in dataset.labels],
                                        dtype='bool'))

#
# Run Searchlight
#

# choose classifier
clf = LinearNuSVMC()

# setup measure to be computed by Searchlight
# cross-validated mean transfer using an odd-even dataset splitter
cv = ClfCrossValidation(TransferError(clf),
                        NFoldSplitter())

# setup plotting
fig = 0
P.figure(figsize=(12,4))


for radius in [1,5,10]:
    # tell which one we are doing
    print "Running searchlight with radius: %i ..." % (radius)

    # setup Searchlight with a custom radius
    # radius has to be in the same unit as the nifti file's pixdim property.
    sl = Searchlight(cv, radius=radius)

    # run searchlight on example dataset and retrieve error map
    sl_map = sl(dataset)

    # map sensitivity map into original dataspace
    orig_sl_map = dataset.mapReverse(sl_map)
    masked_orig_sl_map = N.ma.masked_array(orig_sl_map, mask=orig_sl_map == 0)

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


# show all the cool figures
P.show()

