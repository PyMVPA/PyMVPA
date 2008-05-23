#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Estimator for classifier error distributions."""

__docformat__ = 'restructuredtext'

import numpy as N


class MCNullDist(object):
    # XXX this should be the baseclass of a bunch of tests with more
    # sophisticated tests, perhaps making more assumptions about the data
    # TODO invent derived classes that make use of some reasonable assumptions
    # e.g. a gaussian distribution of transfer errors under the null hypothesis
    # this would have the advantage that a model could be fitted to a much
    # lower number of transfer errors and therefore dramatically reduce the
    # necessary CPU time. This is almost trivial to do with
    #   scipy.stats.norm.{fit,cdf}
    """Class to determine the distribution of a measure under the NULL
    distribution (no signal).

    No assumptions are made about the shape of the distribution under the null
    hypothesis. Instead this distribution is estimated by performing multiple
    measurements with permuted `label` vectors, hence no or random signal.

    The distribution is estimated by calling fit() with an appropriate
    `DatasetMeasure` or `TransferError` instance and a training and a
    validation dataset (in case of a `TransferError`). For a customizable
    amount of cycles the training data labels are permuted and the
    corresponding measure computed. In case of a `TransferError` this is the
    error when predicting the *correct* labels of the validation dataset.

    The distribution be queried using the `cdf()` method, which can be
    configured to report probabilities/frequencies from `left` or `right` tail,
    i.e. fraction of the distribution that is lower or larger than some
    critical value.

    This class also supports `FeaturewiseDatasetMeasure`. In that case `cdf()`
    returns an array of featurewise probabilities/frequencies.
    """
    def __init__(self, permutations=1000, tail='left'):
        """Cheap initialization.

        :Parameter:
            permutations: int
                This many classification attempts with permuted label vectors
                will be performed to determine the distribution under the null
                hypothesis.
            tail: str ['left', 'right']
                Which tail of the distribution to report.

        """
        self.__dist_samples = None
        self.__permutations = permutations
        """Number of permutations to compute the estimate the null
        distribution."""
        self.__tail = tail


    def fit(self, measure, wdata, vdata=None):
        """Fit the distribution by performing multiple cycles which repeatedly
        permuted labels in the training dataset.

        :Parameter:
            measure: (`Featurewise`)`DatasetMeasure` | `TransferError`
                TransferError instance used to compute all errors.
            wdata: `Dataset` which gets permuted and used to compute the
                measure/transfer error multiple times.
            vdata: `Dataset` used for validation. 
                If provided measure is assumed to be a `TransferError` and
                working and validation dataset are passed onto it.
        """
        dist_samples = []
        """Holds the transfer errors when randomized signal."""

        # estimate null-distribution
        for p in xrange(self.__permutations):
            # new permutation all the time
            # but only permute the training data and keep the testdata constant
            # TODO this really needs to be more clever! If data samples are
            # shuffled within a class it really makes no difference for the
            # classifier, hence the number of permutations to estimate the
            # null-distribution of transfer errors can be reduced dramatically
            # when the *right* permutations (the ones that matter) are done.
            wdata.permuteLabels(True, perchunk=False)

            # compute and store the measure of this permutation
            if not vdata is None:
                # assume it has `TransferError` interface
                dist_samples.append(measure(vdata, wdata))
            else:
                dist_samples.append(measure(wdata))

        # store errors
        self.__dist_samples = N.asarray(dist_samples)

        print self.__dist_samples
        # restore original labels
        wdata.permuteLabels(False, perchunk=False)


    def cdf(self, x):
        """Returns the frequency/probability of a value `x` given the estimated
        distribution. Returned values are determined left or right tailed
        depending on the constructor setting.

        In case a `FeaturewiseDatasetMeasure` was used to estimate the
        distribution the method returns an array. In that case `x` can be
        a scalar value or an array of a matching shape.
        """
        if self.__tail == 'left':
            return (self.__dist_samples <= x).mean(axis=0)
        elif self.__tail == 'right':
            return (self.__dist_samples >= x).mean(axis=0)
        else:
            raise ValueError, 'Unknown value "%s" to `tail` argument.' \
                              % self.__tail
