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
Efficient cross-validation using a cached kernel
================================================

.. index:: Cross-validation

This is a simple example showing how to use cached kernel with a SVM
classifier from the Shogun library.  Pre-caching of the kernel for all
samples in dataset eliminates necessity of possibly lengthy
recomputation of the same kernel values on different splits of the
data.  Depending on the data it might provide considerable speed-ups.
"""

from mvpa2.suite import *
from time import time

"""The next few calls load an fMRI dataset and do basic preprocessing."""

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

"""Cached kernel is just a proxy around an existing kernel."""

# setup a cached kernel
kernel_plain = LinearSGKernel(normalizer_cls=False)
kernel = CachedKernel(kernel_plain)

"""Lets setup two cross-validation, where first would use cached
kernel, whenever the later one plain kernel to demonstrate the
speed-up and achievement of exactly the same results"""

# setup a classifier and cross-validation procedure
clf = sg.SVM(svm_impl='libsvm', C=-1.0, kernel=kernel)
cv = CrossValidation(clf, NFoldPartitioner())

# setup exactly the same using a plain kernel for demonstration of
# speedup and equivalence of the results
clf_plain = sg.SVM(svm_impl='libsvm', C=-1.0, kernel=kernel_plain)
cv_plain = CrossValidation(clf_plain, NFoldPartitioner())


"""Although it would be done internally by cached kernel during
initial computation, it is advisable to make initialization of origids
for samples explicit. It would prepare dataset by cleaning up
attributes used by cached kernel possibly on another version of the
same dataset prior to this analysis in real use cases."""

dataset.init_origids(which='samples')

"""Cached kernel needs to be computed given the full dataset which
would later on be used during cross-validation.
"""

# compute kernel for the dataset
t0 = time()
kernel.compute(dataset)
t_caching = time() - t0

"""Lets run both cross-validation procedures using plain and cached
kernels and report the results."""

# run cached cross-validation
t0 = time()
error = np.mean(cv(dataset))
t_cached = time() - t0

# run plain SVM cross-validation for validation and benchmarking
t0 = time()
error_plain = np.mean(cv_plain(dataset))
t_plain = time() - t0

# UC: unique chunks, UT: unique targets
print "Results for %i-fold cross-validation on %i-class problem:" \
      % (len(dataset.UC), len(dataset.UT))
print " plain kernel:  error=%.3f computed in %.2f sec" \
      % (error_plain, t_plain)
print " cached kernel: error=%.3f computed in %.2f sec (cached in %.2f sec)" \
      % (error, t_cached, t_caching)

"""The following is output from running this example::

 Results for 12-fold cross-validation on 9-class problem:
  plain kernel:  error=0.273 computed in 35.82 sec
  cached kernel: error=0.273 computed in 6.50 sec (cached in 3.68 sec)
"""
