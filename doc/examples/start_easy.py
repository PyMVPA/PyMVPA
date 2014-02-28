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
A simple start
==============

Here we show how to perform a simple cross-validated classification analysis
with PyMVPA. This script is the exact equivalent of the
:ref:`example_cmdline_start_easy` example, but using the Python API instead of
the command line interface.

First, we import the PyMVPA suite to enable all PyMVPA building blocks
"""

from mvpa2.suite import *

"""
Now we load an fMRI dataset with some attributes for each volume, only
considering voxels that are non-zero in a mask image.
"""

attr = SampleAttributes(os.path.join(pymvpa_dataroot,
                        'attributes_literal.txt'))
dataset = fmri_dataset(samples=os.path.join(pymvpa_dataroot, 'bold.nii.gz'),
                       targets=attr.targets, chunks=attr.chunks,
                       mask=os.path.join(pymvpa_dataroot, 'mask.nii.gz'))

"""
Next we remove linear trends by polynomial regression for each voxel and
each chunk (recording run) of the dataset individually.
"""

poly_detrend(dataset, polyord=1, chunks_attr='chunks')

"""
For this example we are only interested in data samples that correspond
to the ``face`` or to the ``house`` condition.
"""

dataset = dataset[np.array([l in ['face', 'house'] for l in dataset.sa.targets],
                          dtype='bool')]

"""
The setup for our cross-validation analysis include the selection of a
classifier, and a partitioning scheme, and an error function
to convert literal predictions into a quantitative performance metric.
"""

cv = CrossValidation(SMLR(), OddEvenPartitioner(), errorfx=mean_mismatch_error)
error = cv(dataset)

"""
The resulting dataset contains the computed accuracy.
"""

# UC: unique chunks, UT: unique targets
print "Error for %i-fold cross-validation on %i-class problem: %f" \
      % (len(dataset.UC), len(dataset.UT), np.mean(error))
