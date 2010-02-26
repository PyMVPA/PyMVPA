# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Neighborhood objects """

import numpy as np
from numpy import array
import operator
import sys

from mvpa.clfs.distance import cartesian_distance

if __debug__:
    from mvpa.base import debug

class Sphere(object):
    """N-Dimensional hypersphere.

    Use this if you want to obtain all the neighbors within a given
    radius from a point in a space with arbitrary number of dimensions
    assuming that the space is discrete.

    No validation of producing coordinates within any extent is done.

    Examples
    --------
    Create a Sphere of diameter 1 and obtain all coordinates within range for the
    coordinate (1,1,1).

    >>> s = Sphere(1)
    >>> s((2, 1))
    [(1, 1), (2, 0), (2, 1), (2, 2), (3, 1)]
    >>> s((1, ))
    [(0,), (1,), (2,)]

    If elements in discrete space have different sizes across
    dimensions, it might be preferable to specify element_sizes
    parameter.

    >>> s = Sphere(2, element_sizes=(1.5, 2.5))
    >>> s((2, 1))
    [(1, 1), (2, 1), (3, 1)]

    >>> s = Sphere(1, element_sizes=(1.5, 0.4))
    >>> s((2, 1))
    [(2, -1), (2, 0), (2, 1), (2, 2), (2, 3)]

    """

    def __init__(self, radius, element_sizes=None, distance_func=None):
        """ Initialize the Sphere

        Parameters
        ----------
        radius : float
          Radius of the 'sphere'.  If no `element_sizes` provided --
          radius would be effectively in number of voxels (if
          operating on MRI data).
        element_sizes : None or iterable of floats
          Sizes of elements in each dimension.  If None, it is equivalent
          to 1s in all dimensions.
        distance_func : None or lambda
          Distance function to use (choose one from `mvpa.clfs.distance`).
          If None, cartesian_distance to be used.
        """
        self._radius = radius
        # TODO: make ability to lookup in a dataset
        self._element_sizes = element_sizes
        if distance_func is None:
            distance_func = cartesian_distance
        self._distance_func = distance_func

        self._increments = None
        """Stored template of increments"""
        self._increments_ndim = None
        """Dimensionality of increments"""

    # Properties to assure R/O behavior for now
    @property
    def radius(self):
        return self._radius

    @property
    def element_sizes(self):
        return self._element_sizes

    @property
    def distance_func(self):
        return self._distance_func

    def _get_increments(self, ndim):
        """Creates a list of increments for a given dimensionality
        """
        # Set element_sizes
        element_sizes = self._element_sizes
        if element_sizes is None:
            element_sizes = np.ones(ndim)
        else:
            if (ndim != len(element_sizes)):
                raise ValueError, \
                      "Dimensionality mismatch: element_sizes %s provided " \
                      "to constructor had %i dimensions, whenever queried " \
                      "coordinate had %i" \
                      % (element_sizes, len(element_sizes), ndim)
        center = np.zeros(ndim)

        element_sizes = np.asanyarray(element_sizes)
        # What range for each dimension
        erange = np.ceil(self._radius / element_sizes).astype(int)

        tentative_increments = np.array(list(np.ndindex(tuple(erange*2 + 1)))) \
                               - erange
        # Filter out the ones beyond the "sphere"
        return array([x for x in tentative_increments
                      if self._distance_func(x * element_sizes, center)
                      <= self._radius])


    def train(self, dataset):
        # XXX YOH:  yeap -- BUT if you care about my note above on extracting
        #     somehow sizes -- some dataset.a might come handy may be?
        #     so actual template get constructed in train and _create_template
        #     could go away and just be returned in some R/O property
        #self.dataset = dataset
        # TODO: extract element_sizes
        pass

    # XXX YOH: should it have this at all?  may be sphere should just generate the
    #         "neighborhood template" -- all those offsets where to jump to get
    #         tentative neighbor... Otherwise there are double checks... some here
    #         some in the query engine... also imho Sphere should not even care about any extent
    def __call__(self, coordinate):
        """Get all coordinates within diameter

        Parameters
        ----------
        coordinate : sequence type of length 3 with integers

        Returns
        -------
        list of tuples of size 3

        """
        # type checking
        coordinate = np.asanyarray(coordinate)
        # XXX This might go into _train ...
        scalar = coordinate.ndim == 0
        if scalar:
            # we are dealing with scalars -- lets add a dimension
            # artificially
            coordinate = coordinate[None]
        # XXX This might go into _train ...
        ndim = len(coordinate)
        if self._increments is None  or self._increments_ndim != ndim:
            if __debug__:
                debug('NBH',
                      "Recomputing neighborhood increments for %dD Sphere"
                      % ndim)
            self._increments = self._get_increments(ndim)
            self._increments_ndim = ndim

        if __debug__:
            if coordinate.dtype.char not in np.typecodes['AllInteger']:
                raise ValueError("Sphere must be called on a sequence of "
                                 "integers of length %i, you gave %s "
                                 % (ndim, coordinate))
            #if dataset is None:
            #    raise ValueError("Sphere object has not been trained yet, use "
            #                     "train(dataset) first. ")

        # function call
        coord_array = (coordinate + self._increments)

        # XXX may be optionally provide extent checking?
        ## # now filter out illegal coordinates if they really are outside the
        ## # bounds
        ## if (coordinate - self.radius < 0).any() \
        ## or (coordinate + self.radius >= self.extent).any():
        ##     coord_array = array([c for c in coord_array \
        ##                            if (c >= 0).all()
        ##                            and (c < self.extent).all()])
        ## coord_array = coord_array.transpose()

        if scalar:
            # Take just 0th dimension since 1st was artificially introduced
            coord_array = coord_array[:, 0]
            return coord_array.tolist()
        else:
            # Note: converting first full array to list and then
            # "tuppling" it seems to be faster than tuppling each
            # sub-array
            return [tuple(x) for x in coord_array.tolist()]


class HollowSphere(Sphere):
    """N-Dimensional hypersphere with a hollow internal sphere

    See parent class `Sphere` for more information.

    Examples
    --------
    Create a Sphere of diameter 1 and obtain all coordinates within range for the
    coordinate (1,1,1).

    >>> s = HollowSphere(1, 0)
    >>> s((2, 1))
    [(1, 1), (2, 0), (2, 2), (3, 1)]
    >>> s((1, ))
    [(0,), (2,)]

    """

    def __init__(self, radius, inner_radius, **kwargs):
        """ Initialize the Sphere

        Parameters
        ----------
        radius : float
          Radius of the 'sphere'.  If no `element_sizes` provided --
          radius would be effectively in number of voxels (if
          operating on MRI data).
        inner_radius : float
          Inner radius of the 'sphere', describing where hollow
          part starts.  It is inclusive, so `inner_radius` of 0,
          would already remove the center element.
        **kwargs
          See `Sphere` for additional keyword arguments
        """
        if inner_radius > radius:
            raise ValueError, "inner_radius (got %g) should be smaller " \
                  "than the radius (got %g)" % (inner_radius, radius)
        Sphere.__init__(self, radius, **kwargs)
        self._inner_radius = inner_radius


    # Properties to assure R/O behavior for now
    @property
    def inner_radius(self):
        return self._inner_radius

    def _get_increments(self, ndim):
        """Creates a list of increments for a given dimensionality

        RF: lame yoh just cut-pasted and tuned up because everything
            depends on ndim...
        """
        # Set element_sizes
        element_sizes = self._element_sizes
        if element_sizes is None:
            element_sizes = np.ones(ndim)
        else:
            if (ndim != len(element_sizes)):
                raise ValueError, \
                      "Dimensionality mismatch: element_sizes %s provided " \
                      "to constructor had %i dimensions, whenever queried " \
                      "coordinate had %i" \
                      % (element_sizes, len(element_sizes), ndim)
        center = np.zeros(ndim)

        element_sizes = np.asanyarray(element_sizes)
        # What range for each dimension
        erange = np.ceil(self._radius / element_sizes).astype(int)

        tentative_increments = np.array(list(np.ndindex(tuple(erange*2 + 1)))) \
                               - erange
        # Filter out the ones beyond the "sphere"
        return array([x for x in tentative_increments
                      if self._inner_radius
                      < self._distance_func(x * element_sizes, center)
                      <= self._radius])


class QueryEngine(object):
    """Basic class defining interface for querying neighborhood in a dataset

    Derived classes provide specific implementations possibly with trade-offs
    between generality and performance

    XXX
    """

    def __init__(self, **kwargs):
        # XXX for example:
        # voxels=Sphere(diameter=3)
        self._queryobjs = kwargs
        self._queryattrs = {}


    def train(self, dataset):
        # reset first
        self._queryattrs.clear()
        # store all relevant attributes
        for space in self._queryobjs:
            self._queryattrs[space] = dataset.fa[space].value
        # execute subclass training
        self._train(dataset)


    def query_byid(self, fid):
        """Return feature ids of neighbors for a given feature id
        """
        kwargs = {}
        for space in self._queryattrs:
            kwargs[space] = self._queryattrs[space][fid]
        return self.query(**kwargs)

    #
    # aliases
    #

    def __call__(self, **kwargs):
        return self.query(**kwargs)


    def __getitem__(self, fid):
        return self.query_byid(fid)


class IndexQueryEngine(QueryEngine):
    """Provides efficient query engine for discrete spaces.

    Uses dictionary lookups for elements indices and presence in
    general.  Each space obtains a lookup dictionary which performs
    translation from given index/coordinate into the index within an
    index table (with a dimension per each space to search within).

    TODO:
    - extend documentation
    - repr
    """

    def __init__(self, sorted=True, **kwargs):
        """
        Parameters
        ----------
        sorted : bool
          Results of query get sorted
        """
        QueryEngine.__init__(self, **kwargs)
        self._spaceorder = None
        """Order of the spaces"""
        self._lookups = {}
        """Dictionary of lookup dictionaries per each space"""
        self._sliceall = {}
        """Precrafted indexes to cover ':' situation within ix_"""
        self._searcharray = None
        """Actual searcharray"""
        self.sorted = True
        """Either to sort the query results"""

    def _train(self, dataset):
        # local binding
        qattrs = self._queryattrs
        # in addition to the base class functionality we need to store the
        # order of the query-spaces
        self._spaceorder = qattrs.keys()
        # type check and determine mask dimensions
        dims = []                       # dimensionality of each space
        lookups = self._lookups = {}
        sliceall = self._sliceall = {}
        selector = []
        for space in self._spaceorder:
            # local binding for the attribute
            qattr = qattrs[space]
            # If it is >1D ndarray we need to transform to list of tuples,
            # since ndarray is not hashable
            # XXX would probably work for ANY discrete attribute
            if not qattr.dtype.char in np.typecodes['AllInteger']:
                pass
                #raise ValueError("IndexQueryEngine can only operate on "
                #                 "feature attributes with integer indices "
                #                 "(got: %s)." % str(qattr.dtype))
            if isinstance(qattr, np.ndarray) and len(qattr.shape) > 1:
                qattr = [tuple(x) for x in qattr]

            # determine the dimensions of this space
            # and charge the nonzero selector
            uqattr = list(set(qattr))
            dim = len(uqattr)
            dims.append(dim)
            # Lookup table for elements known to corresponding indices
            # in searcharray
            lookups[space] = lookup = \
                             dict([(u, i) for i, u in enumerate(uqattr)])
            # Precraft "slicing" for all elements for dummy numpy way
            # to select things ;)
            sliceall[space] = np.arange(dim)
            # And fill out selector using current values from qattr
            selector.append([lookup[x] for x in qattr])

        # now check whether we have sufficient information to put each feature
        # id into one unique search array element
        dims = np.array(dims)
        # we can deal with less features (e.g. masked dataset, but not more)
        # XXX (yoh): seems to be too weak of a check... pretty much you are trying
        #            to check either 2 features do not collide in the target
        #            "mask", right?
        if np.prod(dims) < dataset.nfeatures:
            raise ValueError("IndexQueryEngine has insufficient information "
                             "about the dataset spaces. It is required to "
                             "specify an ROI generator for each feature space "
                             "in the dataset (got: %s, #describable: %i, "
                             "#actual features: %i)."
                             % (str(self._spaceorder), np.prod(dims),
                                   dataset.nfeatures))
        # now we can create the search array
        self._searcharray = np.zeros(dims, dtype='int')
        # and fill it with feature ids, but start from ONE to be different from
        # the zeros
        self._searcharray[tuple(selector)] = np.arange(1, dataset.nfeatures + 1)
        # Lets do additional check -- now we should have same # of
        # non-zero elements as features
        if len(self._searcharray.nonzero()[0]) != dataset.nfeatures:
            # TODO:  Figure out how is the bad cow? sad there is no non-unique
            #        function in numpy
            raise ValueError("Multiple features carry the same set of "
                             "attributes %s.  %s engine cannot handle such "
                             "cases -- use another appropriate query engine"
                             % (self._spaceorder, self))


    def query(self, **kwargs):
        # construct the search array slicer
        # need to obey axis order
        slicer = []
        for space in self._spaceorder:
            lookup = self._lookups[space]
            # only generate ROI, if we have a generator
            # otherwise consider all of the unspecified space
            if space in kwargs:
                space_args = kwargs.pop(space)   # so we could check later on
                # if no ROI generator is available, take provided indexes
                # without any additional neighbors etc
                if self._queryobjs[space] is None:
                    roi = np.atleast_1d(space_args)
                else:
                    roi = self._queryobjs[space](space_args)
                # lookup and filter the results
                roi_ind = [lookup[i] for i in roi if (i in lookup)]
                # if no candidate is left, the whole thing does not match
                # regardless of the other spaces
                if not len(roi_ind):
                    return []
                slicer.append(roi_ind)
            else:
                # Provide ":" if no specialization was provided
                slicer.append(self._sliceall[space])
        # check if query had only legal spaces specified
        if len(kwargs):
            raise ValueError, "Do not know how to treat space(s) %s given " \
                  "in parameters of the query" % (kwargs.keys())
        # only ids are of interest -> flatten
        # and we need to back-transfer them into dataset ids by substracting 1
        res = self._searcharray[np.ix_(*slicer)].flatten() - 1
        res = res[res>=0]              # return only the known ones
        if self.sorted:
            return sorted(res)
        else:
            return res
