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
attr = SampleAttributes(os.path.join(pymvpa_dataroot,
                        'attributes_literal.txt'))
dataset = fmri_dataset(samples=os.path.join(pymvpa_dataroot, 'bold.nii.gz'),
                       targets=attr.targets, chunks=attr.chunks,
                       mask=os.path.join(pymvpa_dataroot, 'mask.nii.gz'))

# do chunkswise linear detrending on dataset
poly_detrend(dataset, polyord=1, chunks_attr='chunks')

# zscore dataset relative to baseline ('rest') mean
zscore(dataset, chunks_attr='chunks', param_est=('targets', ['rest']))

# select class face and house for this demo analysis
# would work with full datasets (just a little slower)
dataset = dataset[np.array([l in ['face', 'house'] for l in dataset.sa.targets],
                          dtype='bool')]

# setup cross validation procedure, using SMLR classifier
cv = CrossValidation(SMLR(), OddEvenPartitioner())

# and run it
error = np.mean(cv(dataset))

# UC: unique chunks, UT: unique targets
print "Error for %i-fold cross-validation on %i-class problem: %f" \
      % (len(dataset.UC), len(dataset.UT), error)
