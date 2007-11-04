#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Parameter representation"""


class Parameter(object):
    """This class shall serve as a representation of a parameter.

    It might be useful if a little more information than the pure parameter
    value is required (or even only useful).

    Each parameter must have a value. However additional property can be
    passed to the constructor and will be stored in the object.

    Here is a list of possible property names:

        min   - minimum value
        max   - maximum value
        step  - increment/decrement stepsize
        descr - a descriptive string
    """
    def __init__(self, value, **kwargs):
        """Specify a parameter by its value and optionally an arbitrary number
        of additional parameters.

        The value will be available via the 'val' class member. No addtional
        property with this name is allowed.
        """
        if __debug__:
            if kwargs.has_key('val'):
                raise ValueError, "'val' property name is illegal."

        self.val = value

        for k,v in kwargs.iteritems():
            self.__setattr__(k, v)


