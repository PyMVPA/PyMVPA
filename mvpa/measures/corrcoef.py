# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""FeaturewiseDatasetMeasure of correlation with the labels."""

__docformat__ = 'restructuredtext'

from mvpa.base import externals

import numpy as N

if externals.exists('scipy', raiseException=True):
    # TODO: implement corrcoef optionally without scipy, e.g. N.corrcoef
    from scipy.stats import pearsonr

from mvpa.measures.base import FeaturewiseDatasetMeasure

class CorrCoef(FeaturewiseDatasetMeasure):
    """`FeaturewiseDatasetMeasure` that performs correlation with labels

    XXX: Explain me!
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
            samples_ = samples[:, ifeature]
            corr = pearsonr(samples_, attrdata)
            corrv = corr[pvalue_index]
            # Should be safe to assume 0 corr_coef (or 1 pvalue) if value
            # is actually NaN, although it might not be the case (covar of
            # 2 constants would be NaN although should be 1)
            if N.isnan(corrv):
                if N.var(samples_) == 0.0 and N.var(attrdata) == 0.0 \
                   and len(samples_):
                    # constant terms
                    corrv = 1.0 - pvalue_index
                else:
                    corrv = pvalue_index
            result[ifeature] = corrv

        return result
