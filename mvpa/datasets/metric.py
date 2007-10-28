#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Classes to provide search for the neighbors"""


import numpy as N

#
# Distance functions.
#
def cartesianDistance(a, b):
    """ Return Cartesian distance between a and b
    """
    return N.linalg.norm(a-b)

def absminDistance(a, b):
    """ Returns disntace max(|a-b|)
    XXX There must be better name!

    Useful to select a whole cube of a given "radius"
    """
    return max(abs(a-b))

def manhattenDistance(a, b):
    """ Return Manhatten distance between a and b
    """
    return sum(abs(a-b))


class Metric(object):
    """ Abstract class for any finder.

    Classes subclasses from this class show know about structure of
    the data and thus be able to provide information about the
    neighbors.
    At least one of the methods (getNeighbors, getNeighbor) has to be
    overriden in the derived class.
    NOTE: derived #2 from derived class #1 has to override all methods
    which were overrident in class #1
    """

    def getNeighbors(self, *args, **kwargs):
        """ Return the list of coordinates for the neighbors.

        By default it simply constracts the list based on
        the generator getNeighbor
        """
        return [ x for x in self.getNeighbor(*args, **kwargs) ]


    def getNeighbor(self, *args, **kwargs):
        """ Generator to return coordinate of the neighbor.

        Base class contains the simplest implementation, assuming that
        getNeighbors returns iterative structure to spit out neighbors
        1-by-1
        """
        for neighbor in self.getNeighbors(*args, **kwargs):
            yield neighbor

# XXX Disabled by Michael as it doesn't seem to be required and conflicts
# with Mapper interface when doing multiple inheritance.
#    def __call__(self, *args, **kwargs):
#        """ Sugar -- call implements the generator """
#        return self.getNeighbor(*args, **kwargs)



class DescreteMetric(Metric):
    """ Find neighboring points in descretized space

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
        Initialize the class provided @elementsize and @distance_function
        """
        Metric.__init__(self)
        self.__filter_radius = None
        self.__filter_coord = None
        self.__distance_function = distance_function

        # XXX might not need assume compatible spacementric
        self.__elementsize = N.array(elementsize)
        self.__Ndims = len(self.__elementsize)


    def _computeFilter(self, radius):
        """ (Re)Computer filter_coord based on given radius
        """
        # compute radius in units of elementsize per axis
        elementradius_per_axis = float(radius) / self.__elementsize

        # build prototype search space
        filter_radiuses = N.array([int(N.ceil(abs(x)))
                                   for x in elementradius_per_axis])
        filter_center = filter_radiuses
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


    def getNeighbors(self, origin, radius=0):
        """ Returns coordinates of the neighbors which are within
        distance from coord

        XXX radius might need to be not a scalar but a vector of
        scalars to specify search distance in different dimensions
        differently... but then may be it has to be a tensor to
        specify orientation etc? :-) so it might not be necessary for
        now
        """
        if len(origin) != self.__Ndims:
            raise ValueError("Obtained coordinates [%s] which have different"
                             + " number of dimensions (%d) from known "
                             + " elementsize" % (`origin`, self.__Ndims))
        if radius != self.__filter_radius:
            self._computeFilter(radius)

        # for the ease of future references, it is better to transform
        # coordinates into tuples
        return origin + self.__filter_coord


    def _setFilter(self, filter_coord):
        """ Lets allow to specify some custom filter to use
        """
        self.__filter_coord = filter_coord


    def _getFilter(self):
        """ Lets allow to specify some custom filter to use
        """
        return self.__filter_coord

    filter_coord = property(fget=_getFilter, fset=_setFilter)

# Template for future classes
#
# class MeshMetric(Metric):
#     """ Return list of neighboring points on a mesh
#     """
#     def getNeighbors(self, origin, distance=0):
#         """Return neighbors"""
#         raise NotImplementedError

