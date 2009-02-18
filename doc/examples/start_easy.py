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
Tiny Example of a Full Cross-Validation
=======================================

Very, very simple example showing a complete cross-validation procedure
with no fancy additions whatsoever.
"""

# get PyMVPA running
from mvpa.suite import *

# load PyMVPA example dataset
attr = SampleAttributes(os.path.join(pymvpa_dataroot, 'attributes.txt'))
dataset = NiftiDataset(samples=os.path.join(pymvpa_dataroot, 'bold.nii.gz'),
                       labels=attr.labels,
                       chunks=attr.chunks,
                       mask=os.path.join(pymvpa_dataroot, 'mask.nii.gz'))

# do chunkswise linear detrending on dataset
detrend(dataset, perchunk=True, model='linear')

# zscore dataset relative to baseline ('rest') mean
zscore(dataset, perchunk=True, baselinelabels=[0],
       targetdtype='float32')

# select class 1 and 2 for this demo analysis
# would work with full datasets (just a little slower)
dataset = dataset.selectSamples(
                N.array([l in [1, 2] for l in dataset.labels],
                        dtype='bool'))

# setup cross validation procedure, using SMLR classifier
cv = CrossValidatedTransferError(
            TransferError(SMLR()),
            OddEvenSplitter())
# and run it
error = cv(dataset)

print "Error for %i-fold cross-validation on %i-class problem: %f" \
      % (len(dataset.uniquechunks), len(dataset.uniquelabels), error)
