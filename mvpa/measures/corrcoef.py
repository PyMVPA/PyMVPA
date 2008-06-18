#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""FeaturewiseDatasetMeasure of correlation with the labels."""

__docformat__ = 'restructuredtext'

import numpy as N
from scipy.stats import pearsonr

from mvpa.measures.base import FeaturewiseDatasetMeasure

class CorrCoef(FeaturewiseDatasetMeasure):
    """`FeaturewiseDatasetMeasure` that performs correlation with labels

    F-scores are computed for each feature as the standard fraction of between
    and within group variances. Groups are defined by samples with unique
    labels.

    No statistical testing is performed, but raw F-scores are returned as a
    sensitivity map. As usual F-scores have a range of [0,inf] with greater
    values indicating higher sensitivity.
    """

    def __init__(self, pvalue=False, attr='labels', **kwargs):
        """Initialize

        :Parameters:
          pvalue : bool
            Either to report p-value of pearsons correlation coefficient
            instead of pure correlation coefficient
          attr : basestring
            What attribut to correlate with
        """
        # init base classes first
        FeaturewiseDatasetMeasure.__init__(self, **kwargs)

        self.__pvalue = int(pvalue)
        self.__attr = attr


    def _call(self, dataset):
        """Computes featurewise scores."""

        attrdata = eval('dataset.' + self.__attr)
        samples = dataset.samples
        pvalue_index = self.__pvalue
        result = N.empty((dataset.nfeatures,), dtype=float)

        for ifeature in xrange(dataset.nfeatures):
            corr = pearsonr(samples[:, ifeature], attrdata)
            result[ifeature] = corr[pvalue_index]

        return result
