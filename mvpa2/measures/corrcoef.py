# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""FeaturewiseMeasure of correlation with the labels."""

__docformat__ = 'restructuredtext'

from mvpa2.base import externals

import numpy as np

if externals.exists('scipy', raise_=True):
    # TODO: implement corrcoef optionally without scipy, e.g. np.corrcoef
    from scipy.stats import pearsonr

from mvpa2.measures.base import FeaturewiseMeasure
from mvpa2.datasets.base import Dataset

class CorrCoef(FeaturewiseMeasure):
    """`FeaturewiseMeasure` that performs correlation with labels

    XXX: Explain me!
    """
    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, pvalue=False, attr='targets', **kwargs):
        """Initialize

        Parameters
        ----------
        pvalue : bool
          Either to report p-value of pearsons correlation coefficient
          instead of pure correlation coefficient
        attr : str
          What attribut to correlate with
        """
        # init base classes first
        FeaturewiseMeasure.__init__(self, **kwargs)

        self.__pvalue = int(pvalue)
        self.__attr = attr


    def _call(self, dataset):
        """Computes featurewise scores."""

        attrdata = dataset.sa[self.__attr].value
        if (np.issubdtype(attrdata.dtype, 'c') or
             np.issubdtype(attrdata.dtype, 'U')):
            raise ValueError("Correlation coefficent measure is not meaningful "
                             "for datasets with literal labels.")

        samples = dataset.samples
        pvalue_index = self.__pvalue
        result = np.empty((dataset.nfeatures,), dtype=float)

        for ifeature in xrange(dataset.nfeatures):
            samples_ = samples[:, ifeature]
            corr = pearsonr(samples_, attrdata)
            corrv = corr[pvalue_index]
            # Should be safe to assume 0 corr_coef (or 1 pvalue) if value
            # is actually NaN, although it might not be the case (covar of
            # 2 constants would be NaN although should be 1)
            if np.isnan(corrv):
                if np.var(samples_) == 0.0 and np.var(attrdata) == 0.0 \
                   and len(samples_):
                    # constant terms
                    corrv = 1.0 - pvalue_index
                else:
                    corrv = pvalue_index
            result[ifeature] = corrv

        return Dataset(result[np.newaxis])
