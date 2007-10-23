### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Error function object that computes the root mean squared error
#            of some desired and some predicted values
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

import numpy as N

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




