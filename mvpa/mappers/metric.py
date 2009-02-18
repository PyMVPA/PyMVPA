# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Classes and functions to provide sense of distances between sample points"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.clfs.distance import cartesianDistance

class Metric(object):
    """Abstract class for any metric.

    Subclasses abstract a metric of a dataspace with certain properties and can
    be queried for structural information. Currently, this is limited to
    neighborhood information, i.e. identifying the surround a some coordinate in
    the respective dataspace.

    At least one of the methods (getNeighbors, getNeighbor) has to be overriden
    in every derived class.  NOTE: derived #2 from derived class #1 has to
    override all methods which were overrident in class #1
    """

    def getNeighbors(self, *args, **kwargs):
        """Return the list of coordinates for the neighbors.

        By default it simply constracts the list based on
        the generator getNeighbor
        """
        return [ x for x in self.getNeighbor(*args, **kwargs) ]


    def getNeighbor(self, *args, **kwargs):
        """Generator to return coordinate of the neighbor.

        Base class contains the simplest implementation, assuming that
        getNeighbors returns iterative structure to spit out neighbors
        1-by-1
        """
        for neighbor in self.getNeighbors(*args, **kwargs):
            yield neighbor



class DescreteMetric(Metric):
    """Find neighboring points in descretized space

    If input space is descretized and all points fill in N-dimensional cube,
    this finder returns list of neighboring points for a given distance.

    For all `origin` coordinates this class exclusively operates on discretized
    values, not absolute coordinates (which are e.g. in mm).

    Additionally, this metric has the notion of compatible and incompatible
    dataspace metrics, i.e. the descrete space might contain dimensions for
    which computing an overall distance is not meaningful. This could, for
    example, be a combined spatio-temporal space (three spatial dimension,
    plus the temporal one). This metric allows to define a boolean mask
    (`compatmask`) which dimensions share the same dataspace metrics and for
    which the distance function should be evaluated. If a `compatmask` is
    provided, all cordinates are projected into the subspace of the non-zero
    dimensions and distances are computed within that space.

    However, by using a per dimension radius argument for the getNeighbor
    methods, it is nevertheless possible to define a neighborhood along all
    dimension. For all non-compatible axes the respective radius is treated
    as a one-dimensional distance along the respective axis.
    """

    def __init__(self, elementsize=1, distance_function=cartesianDistance,
                 compatmask=None):
        """
        :Parameters:
          elementsize: float | sequence
            The extent of a dataspace element along all dimensions.
          distance_function: functor
            The distance measure used to determine distances between
            dataspace elements.
          compatmask: 1D bool array | None
            A mask where all non-zero elements indicate dimensions
            with compatiable spacemetrics. If None (default) all dimensions
            are assumed to have compatible spacemetrics.
        """
        Metric.__init__(self)
        self.__filter_radius = None
        self.__filter_coord = None
        self.__distance_function = distance_function

        self.__elementsize = N.array(elementsize, ndmin=1)
        self.__Ndims = len(self.__elementsize)
        if compatmask is None:
            self.__compatmask = N.ones(self.__elementsize.shape, dtype='bool')
        else:
            self.__compatmask = N.array(compatmask, dtype='bool')
            if not self.__elementsize.shape == self.__compatmask.shape:
                raise ValueError, '`compatmask` is of incompatible shape ' \
                        '(need %s, got %s)' % (`self.__elementsize.shape`,
                                               `self.__compatmask.shape`)


    def _expandRadius(self, radius):
        # expand radius to be equal along all dimensions if just scalar
        # is provided
        if N.isscalar(radius):
            radius = N.array([radius] * len(self.__elementsize), dtype='float')
        else:
            radius = N.array(radius, dtype='float')

        return radius


    def _computeFilter(self, radius):
        """ (Re)compute filter_coord based on given radius
        """
        if not N.all(radius[self.__compatmask][0] == radius[self.__compatmask]):
            raise ValueError, \
                  "Currently only neighborhood spheres are supported, " \
                  "not ellipsoids."
        # store radius in compatible space
        compat_radius = radius[self.__compatmask][0]
        # compute radius in units of elementsize per axis
        elementradius_per_axis = radius / self.__elementsize

        # build prototype search space
        filter_radiuses = N.ceil(N.abs(elementradius_per_axis)).astype('int')
        filter_center = filter_radiuses
        comp_center = filter_center[self.__compatmask] \
                            * self.__elementsize[self.__compatmask]
        filter_mask = N.ones((filter_radiuses * 2) + 1, dtype='bool')

        # get coordinates of all elements
        f_coords = N.transpose(filter_mask.nonzero())

        # but start with empty mask
        filter_mask[:] = False

        # check all filter element
        for coord in f_coords:
            dist = self.__distance_function(
                        coord[self.__compatmask]
                            * self.__elementsize[self.__compatmask],
                        comp_center)
            # compare with radius
            if dist <= compat_radius:
                # zero too distant
                filter_mask[N.array(coord, ndmin=2).T.tolist()] = True


        self.__filter_coord = N.array( filter_mask.nonzero() ).T \
                                        - filter_center
        self.__filter_radius = radius


    def getNeighbors(self, origin, radius=0):
        """Returns coordinates of the neighbors which are within
        distance from coord.

        :Parameters:
          origin: 1D array
            The center coordinate of the neighborhood.
          radius: scalar | sequence
            If a scalar, the radius is treated as identical along all dimensions
            of the dataspace. If a sequence, it defines a per dimension radius,
            thus has to have the same number of elements as dimensions.
            Currently, only spherical neighborhoods are supported. Therefore,
            the radius has to be equal along all dimensions flagged as having
            compatible dataspace metrics. It is, however, possible to define
            variant radii for all other dimensions.
        """
        if len(origin) != self.__Ndims:
            raise ValueError("Obtained coordinates [%s] which have different "
                             "number of dimensions (%d) from known "
                             "elementsize" % (`origin`, self.__Ndims))

        # take care of postprocessing the radius the ensure validity of the next
        # conditional
        radius = self._expandRadius(radius)
        if N.any(radius != self.__filter_radius):
            self._computeFilter(radius)

        # for the ease of future references, it is better to transform
        # coordinates into tuples
        return origin + self.__filter_coord


    def _setFilter(self, filter_coord):
        """Lets allow to specify some custom filter to use
        """
        self.__filter_coord = filter_coord


    def _getFilter(self):
        """Lets allow to specify some custom filter to use
        """
        return self.__filter_coord

    def _setElementSize(self, v):
        # reset filter radius
        _elementsize = N.array(v, ndmin=1)
        # assure that it is read-only and it gets reassigned
        # only as a whole to trigger this house-keeping
        _elementsize.flags.writeable = False
        self.__elementsize = _elementsize
        self.__Ndims = len(_elementsize)
        self.__filter_radius = None

    filter_coord = property(fget=_getFilter, fset=_setFilter)
    elementsize = property(fget=lambda self: self.__elementsize,
                           fset=_setElementSize)

# Template for future classes
#
# class MeshMetric(Metric):
#     """Return list of neighboring points on a mesh
#     """
#     def getNeighbors(self, origin, distance=0):
#         """Return neighbors"""
#         raise NotImplementedError

