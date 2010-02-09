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


import numpy as N
from numpy import trapz

from mvpa.base import externals

# Various helper functions
##REF: Name was automagically refactored
def mean_power_fx(data):
    """Returns mean power

    Similar to var but without demeaning
    """
    return N.mean(N.asanyarray(data)**2)

##REF: Name was automagically refactored
def root_mean_power_fx(data):
    """Returns root mean power

    to be comparable against RMSE
    """
    return N.sqrt(mean_power_fx(data))


class _ErrorFx(object):
    """Common error function interface, computing the difference between
    some target and some predicted values.
    """

    """XXX there is no reason to keep this class around imho -- it is
    just the skeleton for all the _ErrorFxs -- interface they
    must conform... and there is no reason to have all those ErrorFx
    as classes... may be they should be just functions?"""

    def __str__(self):
        """Print class name when asked for string
        """
        return self.__class__.__name__

    def __repr__(self):
        """Proper repr for _ErrorFx
        """
        return self.__class__.__name__ + "()"

    def __call__(self, predicted, target):
        """Compute some error value from the given target and predicted
        values (both sequences).
        """
        raise NotImplemented


class RMSErrorFx(_ErrorFx):
    """Computes the root mean squared error of some target and some
    predicted values.
    """
    def __call__(self, predicted, target):
        """Both 'predicted' and 'target' can be either scalars or sequences,
        but have to be of the same length.
        """
        return N.sqrt(N.mean(N.subtract(predicted, target)**2))


class MeanMismatchErrorFx(_ErrorFx):
    """Computes the percentage of mismatches between some target and some
    predicted values.
    """
    def __call__(self, predicted, target):
        """Both 'predicted' and 'target' can be either scalars or sequences,
        but have to be of the same length.
        """
        return 1 - N.mean( predicted == target )


class AUCErrorFx(_ErrorFx):
    """Computes the area under the ROC for the given the
    target and predicted to make the prediction."""
    def __call__(self, predicted, target):
        """Requires all arguments."""
        # sort the target in descending order based on the predicted and
        # set to boolean
        self.t = t = N.asanyarray(target)[N.argsort(predicted)[::-1]] > 0

        # calculate the true positives
        self.tp = tp = N.concatenate(
            ([0], N.cumsum(t)/t.sum(dtype=N.float), [1]))

        # calculate the false positives
        self.fp = fp = N.concatenate(
            ([0], N.cumsum(~t)/(~t).sum(dtype=N.float), [1]))

        return trapz(tp, fp)


if externals.exists('scipy'):
    from scipy.stats import pearsonr

    class CorrErrorFx(_ErrorFx):
        """Computes the correlation between the target and the
        predicted values. Resultant value is the 1 - correlation
        coefficient, so minimization leads to the best value (at 0)

        """
        def __call__(self, predicted, target):
            """Requires all arguments."""
            return 1.0-pearsonr(predicted, target)[0]


    class CorrErrorPFx(_ErrorFx):
        """Computes p-value of correlation between the target and the predicted
        values.

        """
        def __call__(self, predicted, target):
            """Requires all arguments."""
            return pearsonr(predicted, target)[1]

else:
    # slower(?) and bogus(p-value) implementations for non-scipy users
    # TODO: implement them more or less correcly with numpy
    #       functionality
    class CorrErrorFx(_ErrorFx):
        """Computes the correlation between the target and the predicted
        values. Return 1-CC

        """
        def __call__(self, predicted, target):
            """Requires all arguments."""
            l = len(predicted)
            return 1.0 - N.corrcoef(N.reshape(predicted, l),
                                N.reshape(target, l))[0,1]


    class CorrErrorPFx(_ErrorFx):
        """Computes p-value of correlation between the target and the predicted
        values.

        """
        def __call__(self, predicted, target):
            """Requires all arguments."""
            from mvpa.base import warning
            warning("p-value for correlation is implemented only when scipy is "
                    "available. Bogus value -1.0 is returned otherwise")
            return -1.0


class RelativeRMSErrorFx(_ErrorFx):
    """Ratio between RMSE and root mean power of target output.

    So it can be considered as a scaled RMSE -- perfect reconstruction
    has values near 0, while no reconstruction has values around 1.0.
    Word of caution -- it is not commutative, ie exchange of predicted
    and target might lead to completely different answers
    """
    def __call__(self, predicted, target):
        return RMSErrorFx()(predicted, target) / root_mean_power_fx(target)


class Variance1SVFx(_ErrorFx):
    """Ratio of variance described by the first singular value component.

    Of limited use -- left for the sake of not wasting it
    """

    def __call__(self, predicted, target):
        data = N.vstack( (predicted, target) ).T
        # demean
        data_demeaned = data - N.mean(data, axis=0)
        u, s, vh = N.linalg.svd(data_demeaned, full_matrices=0)
        # assure sorting
        s.sort()
        s=s[::-1]
        cvar = s[0]**2 / N.sum(s**2)
        return cvar
