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

The term :class:`~mvpa.measures.searchlight.Searchlight` refers to an algorithm
that runs a scalar :class:`~mvpa.measures.base.DatasetMeasure` on all possible
spheres of a certain size within a dataset (that provides information about
distances between feature locations).  The measure typically computed is a
cross-validated transfer error (see :ref:`CrossValidatedTransferError
<cross-validation>`). The idea to use a searchlight as a sensitivity analyzer
on fMRI datasets stems from :ref:`Kriegeskorte et al. (2006) <KGB06>`.

A searchlight analysis is can be easily performed. This examples shows a minimal
draft of a complete analysis.

First import a necessary pieces of PyMVPA -- this time each bit individually.
"""
from mvpa.datasets.base import dataset_wizard
from mvpa.datasets.splitters import OddEvenSplitter
from mvpa.clfs.svm import LinearCSVMC
from mvpa.clfs.transerror import TransferError
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.measures.searchlight import Searchlight
from mvpa.misc.data_generators import normal_feature_dataset

"""For the sake of simplicity, let's use a small artificial dataset."""

# overcomplicated way to generate an example dataset
ds = normal_feature_dataset(perlabel=10, nlabels=2, nchunks=2,
                          nfeatures=10, nonbogus_features=[3, 7],
                          snr=5.0)
dataset = dataset_wizard(samples=ds.samples, targets=ds.targets,
                  chunks=ds.chunks)

"""Now it only takes three lines for a searchlight analysis."""

# setup measure to be computed in each sphere (cross-validated
# generalization error on odd/even splits)
cv = CrossValidatedTransferError(
         TransferError(LinearCSVMC()),
         OddEvenSplitter())

# setup searchlight with 5 mm radius and measure configured above
sl = Searchlight(cv, radius=5)

# run searchlight on dataset
sl_map = sl(dataset)

print 'Best performing sphere error:', min(sl_map)

"""
If this analysis is done on a fMRI dataset using `NiftiDataset` the resulting
searchlight map (`sl_map`) can be mapped back into the original dataspace
and viewed as a brain overlay. :ref:`Another example <example_searchlight>`
shows a typical application of this algorithm.

.. Mention the fact that it also is a special `SensitivityAnalyzer`
"""
