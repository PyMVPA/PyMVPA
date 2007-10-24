#emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Classes to provide search for the neighbors
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

def cartesianDistance(a, b):
    """ Return cartesian distance between a and b
    """
    return N.linalg.norm(a-b)


def absminDistance(a, b):
    """ Returns disntace max in |a-b|
    XXX There must be better name!

    Useful to select a whole cube of a given "radius"
    """
	return max(abs(a-b))

def manhattenDistance(a, b):
    """ Return Manhatten distance between a and b
    """
    return sum(abs(a-b))


class NeighborFinder:
    """ The class which knows about structure of the data and thus can provide
        information about the neighbors
    """

    def getNeighbors(self, coord):
        """ Return the list of coordinates for the neighbors.
        """
        raise NotImplementedError


class DescreteNeighborFinder(NeighborFinder):
    """ Return list of neighboring points in descretized space

    If input space is descretized and all points fill in
    N-dimensional cube, this finder returns list of neighboring
    points for a given distance.

    As input points it operates on discretized values, not absolute
    coordinates (which are e.g. in mm)
    """

    def __init__(self,
                 elementsize=1,
                 distance_function=cartesianDistance):
        """
        XXX
        """
        self.__filter_radius = None
        self.__filter_coord = None
        self.__distance_function = distance_function

        # XXX might not need assume compatible spacementric
        self.__elementsize = N.array(elementsize)
        self.__Ndims = len(self.__elementsize)


    def __computeFilter(self, radius):
        """ (Re)Computer filter_coord based on given radius
        """
        # compute radius in units of elementsize per axis
        elementradius_per_axis = float(radius) / self.__elementsize

        # build prototype search space
		filter_center = filter_radiuses = N.array(map(lambda x: int(N.ceil(abs(x))), \
													   elementradius_per_axis))
		filter_mask = N.ones( ( filter_radiuses * 2 ) + 1 )

        # now zero out all too distant elements
        f_coords = N.transpose( filter_mask.nonzero() )

        # check all filter element
        for coord in f_coords:
            dist = self.__distance_function(coord*self.__elementsize,
                                            filter_center*self.__elementsize)
            # compare with radius
            if radius < dist:
                # zero too distant
                filter_mask[N.array(coord, ndmin=2).T.tolist()] = 0

        self.__filter_coord = N.array( filter_mask.nonzero() ).T \
							            - filter_center
        self.__filter_radius = radius


    def __call__(self, origin, radius=0):
        """ Returns coordinates of the neighbors which are within
        distance from coord
        XXX radius might need to be not a scalar but a vector of scalars to specify search distance in different dimensions differently... but then may be it has to be a tensor to specify orientation etc? :-) so it might not be necessary for now
        """
		if len(origin) != self.__Ndims:
			raise ValueError("Obtained coordinates [%s] which have different" + \
							 " number of dimensions (%d) from known elementsize" % \
							 (`origin`, self.__Ndims))
        if radius != self.__filter_radius:
            self.__computeFilter(radius)

        return origin + self.__filter_coord


    def setFilter(self, filter_coord):
        """ Lets allow to specify some custom filter to use
        """
        self.__filter_coord = filter_coord

    filter = property( fget=lambda self: self.__filter_coord, fset=setFilter)


class MeshNeighborFinder(NeighborFinder):
    """ Return list of neighboring points on a mesh
    """
    pass
