### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Common error function interface, computing the difference between
#            some desired and some predicted values
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the MIT License.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the COPYING
#    file that comes with this package for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


class ErrorFunction(ErrorFunctionBase):
    """ Common error function interface, computing the difference between
    some desired and some predicted values.
    """
    def __call__(self, predicted, desired):
        """ Compute some error value from the given desired and predicted
        values (both sequences).
        """
        raise NotImplemented




