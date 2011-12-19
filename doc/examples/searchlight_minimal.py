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
Minimal Searchlight Example
===========================

.. index:: searchlight, cross-validation

The term :class:`~mvpa2.measures.searchlight.Searchlight` refers to an
algorithm that runs a scalar :class:`~mvpa2.measures.base.Measure` on
all possible spheres of a certain size within a dataset (that provides
information about distances between feature locations).  The measure
typically computed is a cross-validation of a classifier performance
(see :ref:`CrossValidation <sec_tutorial_crossvalidation>` section in
the tutorial). The idea to use a searchlight as a sensitivity analyzer
on fMRI datasets stems from :ref:`Kriegeskorte et al. (2006) <KGB06>`.

A searchlight analysis is can be easily performed. This examples shows a minimal
draft of a complete analysis.

First import a necessary pieces of PyMVPA -- this time each bit individually.
"""

import numpy as np

from mvpa2.generators.partition import OddEvenPartitioner
from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.measures.base import CrossValidation
from mvpa2.measures.searchlight import sphere_searchlight
from mvpa2.testing.datasets import datasets
from mvpa2.mappers.fx import mean_sample

"""For the sake of simplicity, let's use a small artificial dataset."""

# Lets just use our tiny 4D dataset from testing battery
dataset = datasets['3dlarge']

"""Now it only takes three lines for a searchlight analysis."""

# setup measure to be computed in each sphere (cross-validated
# generalization error on odd/even splits)
cv = CrossValidation(LinearCSVMC(), OddEvenPartitioner())

# setup searchlight with 2 voxels radius and measure configured above
sl = sphere_searchlight(cv, radius=2, space='myspace',
                        postproc=mean_sample())

# run searchlight on dataset
sl_map = sl(dataset)

print 'Best performing sphere error:', np.min(sl_map.samples)

"""
If this analysis is done on a fMRI dataset using `NiftiDataset` the resulting
searchlight map (`sl_map`) can be mapped back into the original dataspace
and viewed as a brain overlay. :ref:`Another example <example_searchlight>`
shows a typical application of this algorithm.

.. Mention the fact that it also is a special `SensitivityAnalyzer`
"""
