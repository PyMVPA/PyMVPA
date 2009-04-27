# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dissimilarity measure.

"""

__docformat__ = 'restructuredtext'

import numpy as N
from mvpa.measures.base import DatasetMeasure
from mvpa.misc.stats import DSMatrix

class DSMDatasetMeasure(DatasetMeasure):
    """DSMDatasetMeasure creates a DatasetMeasure object
       where metric can be one of 'euclidean', 'spearman', 'pearson'
       or 'confusion'"""

    def __init__(self, dsmatrix, dset_metric, output_metric='spearman'):
        DatasetMeasure.__init__(self)

        self.dsmatrix = dsmatrix
        self.dset_metric = dset_metric
        self.output_metric = output_metric
        self.dset_dsm = []


    def __call__(self, dataset):
        # create the dissimilarity matrix for the data in the input dataset
        self.dset_dsm = DSMatrix(dataset.samples, self.dset_metric)

        in_vec = self.dsmatrix.getVectorForm()
        dset_vec = self.dset_dsm.getVectorForm()

        # concatenate the two vectors, send to dissimlarity function
        test_mat = N.asarray([in_vec, dset_vec])

        test_dsmatrix = DSMatrix(test_mat, self.output_metric)

        # return correct dissimilarity value
        return test_dsmatrix.getFullMatrix()[0, 1]
