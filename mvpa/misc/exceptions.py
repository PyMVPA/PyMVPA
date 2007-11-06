#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Exception classes which might get thrown"""

class UnknownStateError(Exception):
    """ Thrown if the internal state of the class is not yet defined.

    Classifiers and Algorithms classes might have properties, which
    are not defined prior to training or invocation has happened.
    """

    def __init__(self, msg=""):
        self.__msg = msg

    def __repr__(self):
        return "Exception: " + self.__msg

