#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Error functions"""

__docformat__ = 'restructuredtext'


from cmath import sqrt
import numpy as N
from scipy import trapz
from scipy.stats import pearsonr

class ErrorFunctionBase(object):
    """
    Dummy error function base class
    """
    pass


class ErrorFunction(ErrorFunctionBase):
    """Common error function interface, computing the difference between
    some desired and some predicted values.
    """
    def __call__(self, predicted, desired):
        """Compute some error value from the given desired and predicted
        values (both sequences).
        """
        raise NotImplemented


class RMSErrorFx(ErrorFunction):
    """Computes the root mean squared error of some desired and some
    predicted values.
    """
    def __call__(self, predicted, desired):
        """Both 'predicted' and 'desired' can be either scalars or sequences,
        but have to be of the same length.
        """
        difference = N.subtract(predicted, desired)

        return sqrt(N.dot(difference, difference))


class MeanMismatchErrorFx(ErrorFunction):
    """Computes the percentage of mismatches between some desired and some
    predicted values.
    """
    def __call__(self, predicted, desired):
        """Both 'predicted' and 'desired' can be either scalars or sequences,
        but have to be of the same length.
        """
        return 1 - N.mean( predicted == desired )


class AUCErrorFx(ErrorFunction):
    """Computes the area under the ROC for the given the 
    desired and predicted to make the prediction."""
    def __call__(self, predicted, desired):
        """Requires all arguments."""
        # sort the desired in descending order based on the predicted and
        # set to boolean
        t = desired[N.argsort(predicted)[::-1]] > 0
        
        # calculate the true positives
        tp = N.concatenate(([0],
                            N.cumsum(t)/t.sum(dtype=N.float),
                            [1]))

        # calculate the false positives
        fp = N.concatenate(([0],
                            N.cumsum(~t)/(~t).sum(dtype=N.float),
                            [1]))

        return trapz(tp,fp)

        
class CorrErrorFx(ErrorFunction):
    """Computes the correlation between the desired and the predicted
    values."""
    def __call__(self, predicted, desired):
        """Requires all arguments."""
        return pearsonr(predicted, desired)[0]

