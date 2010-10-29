# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Error functions helpers.

PyMVPA can use arbitrary function which takes 2 arguments: predictions
and targets and spits out a scalar value. Functions below are for the
convinience, and they confirm the agreement that 'smaller' is 'better'"""

__docformat__ = 'restructuredtext'


import numpy as np
from numpy import trapz

from mvpa.base import externals

# Various helper functions
##REF: Name was automagically refactored
def mean_power_fx(data):
    """Returns mean power

    Similar to var but without demeaning
    """
    return np.mean(np.asanyarray(data)**2)

##REF: Name was automagically refactored
def root_mean_power_fx(data):
    """Returns root mean power

    to be comparable against RMSE
    """
    return np.sqrt(mean_power_fx(data))


def rms_error(predicted, target):
    """Computes the root mean squared error of some target and some
    predicted values.

    Both 'predicted' and 'target' can be either scalars or sequences,
    but have to be of the same length.
    """
    return np.sqrt(np.mean(np.subtract(predicted, target)**2))


def mean_mismatch_error(predicted, target):
    """Computes the percentage of mismatches between some target and some
    predicted values.
    Both 'predicted' and 'target' can be either scalars or sequences,
    but have to be of the same length.
    """
    return np.mean( predicted != target )


def mismatch_error(predicted, target):
    """Computes number of mismatches between some target and some
    predicted values.
    Both 'predicted' and 'target' can be either scalars or sequences,
    but have to be of the same length.
    """
    return np.sum( predicted != target )


def match_accuracy(predicted, target):
    """Computes number of matches between some target and some
    predicted values.
    Both 'predicted' and 'target' can be either scalars or sequences,
    but have to be of the same length.
    """
    return np.sum( predicted == target )

def mean_match_accuracy(predicted, target):
    """Computes mean of number of matches between some target and some
    predicted values.
    Both 'predicted' and 'target' can be either scalars or sequences,
    but have to be of the same length.
    """
    return np.mean( predicted == target )


def auc_error(predicted, target):
    """Computes the area under the ROC for the given the
    target and predicted to make the prediction."""
    # sort the target in descending order based on the predicted and
    # set to boolean
    t = np.asanyarray(target)[np.argsort(predicted)[::-1]] > 0

    # calculate the true positives
    tp = np.concatenate(
        ([0], np.cumsum(t)/t.sum(dtype=np.float), [1]))

    # calculate the false positives
    fp = np.concatenate(
        ([0], np.cumsum(~t)/(~t).sum(dtype=np.float), [1]))

    return trapz(tp, fp)


if externals.exists('scipy'):
    from scipy.stats import pearsonr

    def corr_error(predicted, target):
        """Computes the correlation between the target and the
        predicted values. Resultant value is the 1 - correlation
        coefficient, so minimization leads to the best value (at 0).

        In case of NaN correlation (no variance in predictors or
        targets) result output error is 1.0.
        """
        r = pearsonr(predicted, target)[0]
        if np.isnan(r):
            r = 0.0
        return 1.0 - r

    def correlation(predicted, target):
        """Computes the correlation between the target and the
        predicted values.

        In case of NaN correlation (no variance in predictors or
        targets) result output error is 0.
        """
        r = pearsonr(predicted, target)[0]
        if np.isnan(r):
            r = 0.0
        return r


    def corr_error_prob(predicted, target):
        """Computes p-value of correlation between the target and the predicted
        values.
        """
        return pearsonr(predicted, target)[1]

else:
    # slower(?) and bogus(p-value) implementations for non-scipy users
    # TODO: implement them more or less correcly with numpy
    #       functionality
    def correlation(predicted, target):
        """Computes the correlation between the target and the predicted
        values. Return 1-CC

        In case of NaN correlation (no variance in predictors or
        targets) result output error is 1.0.
        """
        l = len(predicted)
        r = np.corrcoef(np.reshape(predicted, l),
                       np.reshape(target, l))[0,1]
        if np.isnan(r):
            r = 0.0
        return 1.0 - r


    def corr_error_prob(predicted, target):
        """Computes p-value of correlation between the target and the predicted
        values.
        """
        from mvpa.base import warning
        warning("p-value for correlation is implemented only when scipy is "
                "available. Bogus value -1.0 is returned otherwise")
        return -1.0


def relative_rms_error(predicted, target):
    """Ratio between RMSE and root mean power of target output.

    So it can be considered as a scaled RMSE -- perfect reconstruction
    has values near 0, while no reconstruction has values around 1.0.
    Word of caution -- it is not commutative, ie exchange of predicted
    and target might lead to completely different answers
    """
    return rms_error(predicted, target) / root_mean_power_fx(target)


def variance_1sv(predicted, target):
    """Ratio of variance described by the first singular value component.

    Of limited use -- left for the sake of not wasting it
    """
    data = np.vstack( (predicted, target) ).T
    # demean
    data_demeaned = data - np.mean(data, axis=0)
    u, s, vh = np.linalg.svd(data_demeaned, full_matrices=0)
    # assure sorting
    s.sort()
    s=s[::-1]
    cvar = s[0]**2 / np.sum(s**2)
    return cvar
