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
    """Class to determine the distribution of a transfer error under the NULL
    distribution (no signal).

    No assumptions are made about the shape of the distribution under the null
    hypothesis. Instead this distribution is estimated by performing multiple
    classification attempts with permuted `label` vectors, hence no or random
    signal.

    The distribution is estimated by calling fit() with an appropriate
    `TransferError` instance and a training and a validation dataset. For a
    customizable amount of cycles the training data labels are permuted and the
    corresponding error when predicting the *correct* labels of the validation
    dataset is determined.

    The distribution be queried using the `cdf()` method, which can be
    configured to report probabilities/frequencies from `left` or `right` tail,
    i.e. fraction of the distribution that is lower or larger than some
    critical value.
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


    def fit(self, transerr, wdata, vdata):
        """Fit the distribution by performing multiple cycles which repeatedly
        permuted labels in the training dataset.

        :Parameter:
            transerror: `TransferError`
                TransferError instance used to compute all errors.
            wdata: `Dataset` which gets permuted and used to train a
                classifier multiple times.
            vdata: `Dataset` used to validate each trained classifier.
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

            # compute and store the training error of this permutation
            dist_samples.append(transerr(vdata, wdata))

        # store errors
        self.__dist_samples = N.asarray(dist_samples)

        # restore original labels
        wdata.permuteLabels(False, perchunk=False)


    def cdf(self, x):
        """Returns the probability of a scalar value `x` or lower given the
        estimated distribution.
        """
        if self.__tail == 'left':
            return (self.__dist_samples <= x).mean()
        elif self.__tail == 'right':
            return (self.__dist_samples >= x).mean()
        else:
            raise ValueError, 'Unknown value "%s" to `tail` argument.' \
                              % self.__tail
