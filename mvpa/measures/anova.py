# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""FeaturewiseDatasetMeasure performing a univariate ANOVA."""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.measures.base import FeaturewiseDatasetMeasure

# TODO: Extend with access to functionality from scipy.stats?
# For binary:
#  2-sample kolmogorov-smirnof might be interesting
#   (scipy.stats.ks_2samp) to judge if two conditions are derived
#   from different distributions (take it as 'activity' vs 'rest'),
#
# For binary+multiclass:
#  kruskal-wallis H-test (scipy.stats.kruskal)
#  GLM: scipy.stats.glm
#
# and may be some others

class OneWayAnova(FeaturewiseDatasetMeasure):
    """`FeaturewiseDatasetMeasure` that performs a univariate ANOVA.

    F-scores are computed for each feature as the standard fraction of between
    and within group variances. Groups are defined by samples with unique
    labels.

    No statistical testing is performed, but raw F-scores are returned as a
    sensitivity map. As usual F-scores have a range of [0,inf] with greater
    values indicating higher sensitivity.
    """
    def __init__(self, **kwargs):
        """Nothing special to do here.
        """
        # init base classes first
        FeaturewiseDatasetMeasure.__init__(self, **kwargs)


    def _call(self, dataset):
        """Computes featurewise f-scores."""
        # group means
        means = []
        # with group variance
        vars_ = []

        # split by groups -> [groups x [samples x features]]
        for ul in dataset.uniquelabels:
            ul_samples = dataset.samples[dataset.labels == ul]
            means.append(ul_samples.mean(axis=0))
            vars_.append(ul_samples.var(axis=0))

        # mean of within group variances
        mvw = N.array(vars_).mean(axis=0)
        # variance of group means
        vgm = N.array(means).var(axis=0)

        # compute f-scores (in-place to save some cycles)
        # XXX may cause problems when there are features with no variance in
        # some groups. One could deal with them here and possibly assign a
        # zero f-score to throw them out, but at least theoretically zero
        # variance is possible. Another possiblilty could be to apply
        # N.nan_to_num(), but this might hide the problem.
        # Michael therefore thinks that it is best to let the user deal with
        # it prior to any analysis.

        # for features where there is no variance between the groups,
        # we should simply leave 0 as is, and avoid that way NaNs for
        # invariance features
        vgm0 = vgm.nonzero()
        vgm[vgm0] /= mvw[vgm0]

        return vgm
