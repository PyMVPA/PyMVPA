#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Null-hypothesis testing of transfer errors."""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.misc.state import StateVariable, Stateful


class NullHypothesisTest(Stateful):
    # XXX this should be the baseclass of a bunch of tests with more
    # sophisticated tests, perhaps making more assumptions about the data
    # TODO find a nicer name for it
    # TODO invent derived classes that make use of some reasonable assumptions
    # e.g. a gaussian distribution of transfer errors under the null hypothesis
    # this would have the advantage that a model could be fitted to a much
    # lower number of transfer errors and therefore dramatically reduce the
    # necessary CPU time. This is almost trivial to do with
    #   scipy.stats.norm.{fit,cdf}
    """Simple class to determine statistical significance of classification
    transfer errors.

    No assumptions are made about the distribution of the errors under the null
    hypothesis. Instead this distribution is estimated by performing multiple
    classification attempts with permuted `label` vectors, hence no or random
    signal (abrv. RTE: random transfer error). After a customizable number of
    permutations another transfer error is computed using the original `label`
    vector (abrv. ETE: empirical transfer error).

    The `__call__()` method finally returns the fraction of RTEs that are of
    *lower or equal* value than the ETE. This value is an estimate of the
    probability to achieve a transfer error as low or lower as the ETE, when
    the data samples contain *no* signal.
    """

    # register state members
    null_errors = StateVariable()
    emp_error = StateVariable()

    def __init__(self,
                 transerror,
                 permutations=1000
                ):
        """Cheap initialization.

        :Parameter:
            `transerror`: `TransferError` instance.
            `permutations`: Number of permutations.
                This many classification attempts with permuted label vectors
                will be performed to determine the distribution under the null
                hypothesis.
        """
        Stateful.__init__(self)

        self.__trans_error = transerror
        """`TransferError` instance used to compute all errors."""

        self.__permutations = permutations
        """Number of permutations to compute the estimate the null
        distribution."""


    def __call__(self, data, testdata):
        """Returns an estimate of the probability of an empirical transfer
        error (or lower) computed using a given dataset and a classifier when
        the `data` contains no relevant information.

        :Parameter:
            `data`: `Dataset` which gets permuted and used to train a
                classifier multiple times.
            `testdata`: `Dataset` used to validate each trained classifier.

        Return a single scalar floating point value.
        """
        null_errors = []
        """Holds the transfer errors when randomized signal."""

        emp_error = self.__trans_error(testdata, data)
        """Transfer error with original signal."""

        # estimate null-distribution
        for p in xrange(self.__permutations):
            # new permutation all the time
            # but only permute the training data and keep the testdata constant
            # TODO this really needs to be more clever! If data samples are
            # shuffled within a class it really makes no difference for the
            # classifier, hence the number of permutations to estimate the
            # null-distribution of transfer errors can be reduced dramatically
            # when the *right* permutations (the ones that matter) are done.
            data.permuteLabels(True, perchunk=False)

            # compute and store the training error of this permutation
            null_errors.append(self.__trans_error(testdata, data))

        # calculate the probability estimate of 'emp_error' being likely
        # when no signal is in the data
        null_errors = N.array(null_errors)
        prob = (null_errors <= emp_error).mean()

        # restore original labels
        data.permuteLabels(False, perchunk=False)

        self.emp_error = emp_error
        self.null_errors = null_errors

        return prob
