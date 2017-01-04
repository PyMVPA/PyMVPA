# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Univariate ANOVA"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import externals
from mvpa2.measures.base import FeaturewiseMeasure
from mvpa2.base.dataset import vstack
from mvpa2.datasets.base import Dataset

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


class OneWayAnova(FeaturewiseMeasure):
    """`FeaturewiseMeasure` that performs a univariate ANOVA.

    F-scores are computed for each feature as the standard fraction of between
    and within group variances. Groups are defined by samples with unique
    labels.

    No statistical testing is performed, but raw F-scores are returned as a
    sensitivity map. As usual F-scores have a range of [0,inf] with greater
    values indicating higher sensitivity.

    The sensitivity map is returned as a single-sample dataset. If SciPy is
    available the associated p-values will also be computed and are available
    from the 'fprob' feature attribute.
    """

    def __init__(self, space='targets', **kwargs):
        """
        Parameters
        ----------
        space : str
          What samples attribute to use as targets (labels).
        """
        # set auto-train flag since we have nothing special to be done
        # so by default auto train
        kwargs['auto_train'] = kwargs.get('auto_train', True)
        FeaturewiseMeasure.__init__(self, space=space, **kwargs)

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        if self.get_space() != 'targets':
            prefixes += ['targets_attr=%r' % (self.get_space())]
        return \
            super(FeaturewiseMeasure, self).__repr__(prefixes=prefixes)

    def _call(self, dataset):
        # This code is based on SciPy's stats.f_oneway()
        # Copyright (c) Gary Strangman.  All rights reserved
        # License: BSD
        #
        # However, it got tweaked and optimized to better fit into PyMVPA.

        # number of groups
        targets_sa = dataset.sa[self.get_space()]
        labels = targets_sa.value
        ul = targets_sa.unique

        na = len(ul)
        alldata = dataset.samples

        # total squares of sums
        # determine data-dtype for derived "sum of" measurements, which we need
        # to have as float for proper divisions
        so_dtype = np.float if np.issubdtype(alldata.dtype, np.integer) else alldata.dtype
        bign = dataset.nsamples

        sostot = np.sum(alldata, axis=0, dtype=so_dtype)
        sostot *= sostot
        sostot /= bign

        # total sum of squares
        sstot = np.sum(alldata * alldata, axis=0, dtype=so_dtype) - sostot

        # between group sum of squares
        ssbn = np.zeros(dataset.nfeatures, dtype=so_dtype)
        for l in ul:
            # all samples for the respective label
            d = alldata[labels == l]
            sos = np.sum(d, axis=0, dtype=so_dtype)
            sos *= sos
            sos /= len(d)  # inplace so we don't demand new temp storage
            ssbn += sos

        ssbn -= sostot
        # within
        sswn = sstot - ssbn

        # degrees of freedom
        dfbn = na - 1
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
        f[np.isnan(f)] = 0

        if externals.exists('scipy'):
            from scipy.stats.stats import special
            return Dataset(f[np.newaxis], fa={'fprob': special.fdtrc(dfbn, dfwn, f)})
        else:
            return Dataset(f[np.newaxis])


class CompoundOneWayAnova(OneWayAnova):
    """Compound comparisons via univariate ANOVA.

    This measure compute an ANOVA F-score per each feature, for each
    one-vs-rest comparison for all unique labels in a dataset. Each F-score
    vector for each comparison is included in the return datasets as a separate
    samples. Corresponding p-values are available in feature attributes named
    'fprob_X', where `X` is the name of the actual comparison label. Note that
    p-values are only available, if SciPy is installed. The comparison labels
    for each F-vector are also stored as 'targets' sample attribute in the
    returned dataset.
    """

    def _call(self, dataset):
        """Computes feature-wise f-scores using compound comparisons."""

        targets_sa = dataset.sa[self.get_space()]
        orig_labels = targets_sa.value
        labels = orig_labels.copy()

        # Lets create a very shallow copy of a dataset with just
        # samples and targets_attr
        dataset_mod = Dataset(dataset.samples,
                              sa={self.get_space(): labels})
        results = []
        for ul in targets_sa.unique:
            labels[orig_labels == ul] = 1
            labels[orig_labels != ul] = 2
            f_ds = OneWayAnova._call(self, dataset_mod)
            if 'fprob' in f_ds.fa:
                # rename the fprob attribute to something label specific
                # to survive final aggregation stage
                f_ds.fa['fprob_' + str(ul)] = f_ds.fa.fprob
                del f_ds.fa['fprob']
            results.append(f_ds)

        # use 'update' strategy for fa since we need to accumulate all those
        # fprob_X's
        results = vstack(results, fa='update')
        results.sa[self.get_space()] = targets_sa.unique
        return results
