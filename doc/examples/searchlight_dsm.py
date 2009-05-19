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
A searchlight computing a dissimilarity matrix measure
======================================================

.. index:: searchlight, cross-validation, dissimilarity matrix

This example extends the minimal Searchlight example to use a dissimilarity
matrix-based DatasetMetric to compute Searchlight-center significance.  This
is based on representational similarity analysis (RSA) as presented in
:ref:`Kriegeskorte et al. (2008) <KMB08>`.

First import all necessary parts of PyMVPA.
"""

from mvpa.suite import *

"""Create a small artificial dataset."""

# overcomplicated way to generate an example dataset
ds = normalFeatureDataset(perlabel=10, nlabels=2, nchunks=2,
                          nfeatures=10, nonbogus_features=[3, 7],
                          snr=5.0)
dataset = MaskedDataset(samples=ds.samples, labels=ds.labels,
                        chunks=ds.chunks)

"""Create a dissimilarity matrix based on the labels of the data points
in our test dataset.  This will allow us to see if there is a correlation
between any given searchlight sphere and the experimental conditions."""

# create dissimilarity matrix using the 'confusion' distance
# metric
dsm = DSMatrix(dataset.labels, 'confusion')

"""Now it only takes three lines for a searchlight analysis."""

# setup measure to be computed in each sphere (correlation
# distance between dissimilarity matrix and the dissimilarities
# of a particular searchlight sphere across experimental
# conditions), N.B. in this example between-condition
# dissimilarity is also pearson's r (i.e., correlation distance)
dsmetric = DSMDatasetMeasure(dsm, 'pearson', 'pearson')
 
# setup searchlight with 5 mm radius and measure configured above
sl = Searchlight(dsmetric, radius=5)

# run searchlight on dataset
sl_map = sl(dataset)

print 'Best performing sphere error:', max(sl_map)

"""
If this analysis is done on a fMRI dataset using `NiftiDataset` the resulting
searchlight map (`sl_map`) can be mapped back into the original dataspace and
viewed as a brain overlay. :ref:`Another example <example_searchlight_2d>`
shows a typical application of this algorithm.

.. Mention the fact that it also is a special `SensitivityAnalyzer`
"""
