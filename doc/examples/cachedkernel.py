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
Simple Example of Classification using Cached Kernel
=======================================

Very, very very simple example showing how to use cached kernel 
with a SVM classifier implemented in Shogun library
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

# setup a cached kernel
kernel = LinearSGKernel(normalizer_cls=False)
kernel = CachedKernel(kernel)

# setup a classifier and cross-validation procedure
clf = sg.SVM(svm_impl='libsvm', C=-1.0, kernel=kernel)
cv = CrossValidatedTransferError(
            TransferError(clf),
            splitter=NFoldSplitter())

# prepare dataset by cleaning up attributes used by cached kernel
dataset.init_origids(which='samples')

# compute kernel for the dataset
kernel.compute(dataset)

# and run it
error = np.mean(cv(dataset))

# UC: unique chunks, UT: unique targets
print "Error for %i-fold cross-validation on %i-class problem: %f" \
      % (len(dataset.UC), len(dataset.UT), error)
