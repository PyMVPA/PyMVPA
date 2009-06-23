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

    def _call(self, dataset, labels=None):
        # This code is based on SciPy's stats.f_oneway()
        # Copyright (c) Gary Strangman.  All rights reserved
        # License: BSD
        #
        # However, it got tweaked and optimized to better fit into PyMVPA.

        # number of groups
        if labels is None:
            labels = dataset.labels

        ul = N.unique(labels)

        na = len(ul)
        bign = float(dataset.nsamples)
        alldata = dataset.samples

        # total squares of sums
        sostot = N.sum(alldata, axis=0)
        sostot *= sostot
        sostot /= bign

        # total sum of squares
        sstot = N.sum(alldata * alldata, axis=0) - sostot

        # between group sum of squares
        ssbn = 0
        for l in ul:
            # all samples for the respective label
            d = alldata[labels == l]
            sos = N.sum(d, axis=0)
            sos *= sos
            ssbn += sos / float(len(d))

        ssbn -= sostot
        # within
        sswn = sstot - ssbn

        # degrees of freedom
        dfbn = na-1
        dfwn = bign - na

        # mean sums of squares
        msb = ssbn / float(dfbn)
        msw = sswn / float(dfwn)
        f = msb / msw
        # assure no NaNs -- otherwise it leads instead of
        # sane unittest failure (check of NaNs) to crazy
        #   File "mtrand.pyx", line 1661, in mtrand.shuffle
        #  TypeError: object of type 'numpy.int64' has no len()
        # without any sane backtrace
        f[N.isnan(f)] = 0

        return f

        # XXX maybe also compute p-values?
        #prob = scipy.stats.fprob(dfbn, dfwn, f)
        #return prob


class CompoundOneWayAnova(OneWayAnova):
    """Compound comparisons via univariate ANOVA.

    Provides F-scores per each label if compared to the other labels.
    """

    def _call(self, dataset):
        """Computes featurewise f-scores using compound comparisons."""

        orig_labels = dataset.labels
        labels = orig_labels.copy()

        results = []
        for ul in dataset.uniquelabels:
            labels[orig_labels == ul] = 1
            labels[orig_labels != ul] = 2
            results.append(OneWayAnova._call(self, dataset, labels))

        # features x labels
        return N.array(results).T


