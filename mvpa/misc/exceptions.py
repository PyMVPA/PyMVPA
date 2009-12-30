# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Exception classes which might get thrown"""

__docformat__ = 'restructuredtext'

class UnknownStateError(Exception):
    """Thrown if the internal state of the class is not yet defined.

    Classifiers and Algorithms classes might have properties, which
    are not defined prior to training or invocation has happened.
    """
    pass

class ConvergenceError(Exception):
    """Thrown if some algorithm does not converge to a solution.
    """
    pass

class InvalidHyperparameterError(Exception):
    """Generic exception to be raised when setting improper values
    as hyperparameters.
    """
    pass
