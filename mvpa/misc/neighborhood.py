# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Neighborhood objects """

import numpy as N
from numpy import array
import operator

class Sphere(object):
    """ 3 Dimensional sphere

    Use this if you want to obtain all the neighbors within a given diameter of
    a 3 dimensional coordiante.

    Example
    -------
    Create a Sphere of diamter 9 and obtain all coordinates within range for the
    coordinate (1,1,1).

    >>> s = Sphere(9)
    >>>coords = s((1,1,1))

    """
    def __init__(self, diameter):
        """ Initialise the Sphere

        Parameters
        ----------
        diameter : odd int

        """
        if diameter%2 != 1:
            raise ValueError("Sphere diameter must be odd, but is: %d"
                             % diameter)
        self.diameter = diameter
        self.coord_list = self._create_template()

    def _create_template(self):
        center = array((0,0,0))
        radius = self.diameter/2
        lr = range(-radius,radius+1) # linear range
        return array([array((i,j,k)) for i in lr
                              for j in lr
                              for k in lr
                              if _euclid(array((i,j,k)),center) <= radius])

    def __call__(self, coordinate):
        """  Get all coordinates within diameter

        Parameters
        ----------
        coordinate : sequence type of length 3 with integers

        """
        # type checking
        coordinate = N.asanyarray(coordinate)
        if __debug__:
            if len(coordinate) != 3 \
            or coordinate.dtype.char not in N.typecodes['AllInteger']:
                raise ValueError("Sphere must be called on a sequence of integers of "
                                 "length 3, you gave: "+ str(coordinate))
        # function call
        return  coordinate + self.coord_list

def _euclid(coord1, coord2):
    return N.sqrt(N.sum((coord1-coord2)**2))

