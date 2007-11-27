#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""SensitivityAnalyzer performing a univariate ANOVA."""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.algorithms.datameasure import SensitivityAnalyzer


class OneWayAnova(SensitivityAnalyzer):
    """`SensitivityAnalyzer` that performs a univariate ANOVA.

    F-scores are computed for each feature as the standard fraction of between
    and within group variances. Groups are defined by samples with unique
    labels.

    No statistical testing is performed, but raw F-scores are returned as a
    sensitivity map. As usual F-scores have a range of [0,inf] with greater
    values indicating higher sensitivity.
    """
    def __init__(self):
        """Nothing special to do here.
        """
        # init base classes first
        SensitivityAnalyzer.__init__(self)


    def __call__(self, dataset, callables=[]):
        """Computes featurewise f-scores."""
        # group means
        means = []
        # with group variance
        vars = []

        # split by groups -> [groups x [samples x features]]
        for ul in dataset.uniquelabels:
            ul_samples = dataset.samples[dataset.labels == ul]
            means.append(ul_samples.mean(axis=0))
            vars.append(ul_samples.var(axis=0))

        # compute featurewise f = var(group_means) / mean(within_group_vars)
        f = N.array(means).var(axis=0) / N.array(vars).mean(axis=0)

        return f
