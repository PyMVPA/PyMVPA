#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Error functions"""


from cmath import sqrt
import numpy as N


class ErrorFunctionBase(object):
    """
    Dummy error function base class
    """
    pass



class ErrorFunction(ErrorFunctionBase):
    """ Common error function interface, computing the difference between
    some desired and some predicted values.
    """
    def __call__(self, predicted, desired):
        """ Compute some error value from the given desired and predicted
        values (both sequences).
        """
        raise NotImplemented



class RMSErrorFx(ErrorFunction):
    """ Computes the root mean squared error of some desired and some
    predicted values.
    """
    def __call__(self, predicted, desired):
        """ Both 'predicted' and 'desired' can be either scalars or sequences,
        but have to be of the same length.
        """
        difference = N.subtract(predicted, desired)

        return sqrt(N.dot(difference, difference))



class MeanMatchErrorFx(ErrorFunction):
    """ Computes the percentage of matches between some desired and some
    predicted values.
    """
    def __call__(self, predicted, desired):
        """ Both 'predicted' and 'desired' can be either scalars or sequences,
        but have to be of the same length.
        """
        return N.mean( predicted == desired )




