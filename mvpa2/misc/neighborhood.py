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
import sys
import itertools

from mvpa2.base import warning
from mvpa2.base.types import is_sequence_type
from mvpa2.base.dochelpers import borrowkwargs, borrowdoc, _repr_attrs, _repr
from mvpa2.clfs.distance import cartesian_distance

from mvpa2.misc.support import idhash as idhash_

if __debug__:
    from mvpa2.base import debug

class IdentityNeighborhood(object):
    """Trivial neighborhood.

    Use this if you want neighbors(i) == [i]
    """

    def __init__(self):
        """Initialize the neighborhood"""
        pass

    def __repr__(self):
        return self.__class__.__name__

    def train(self, dataset):
        pass

    def __call__(self, coordinate):
        """Return coordinate in a list

        Parameters
        ----------
        coordinate : sequence type of length 3 with integers

        Returns
        -------
        [tuple(coordinate)]

        """
        # type checking
        coordinate = np.asanyarray(coordinate)
        return [tuple(coordinate)]


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
          Distance function to use (choose one from `mvpa2.clfs.distance`).
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

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        prefixes_ = ['radius=%r' % (self._radius,)] + prefixes
        if self._element_sizes:
            prefixes_.append('element_sizes=%r' % (self._element_sizes,))
        if self._distance_func != cartesian_distance:
            prefixes_.append('distance_func=%r' % self._distance_func)
        return "%s(%s)" % (self.__class__.__name__, ', '.join(prefixes_))

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

        if len(self._increments):
            # function call
            coord_array = (coordinate + self._increments)
        else:
            # if no increments -- no neighbors -- empty list
            return []

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

    def __init__(self, radius, inner_radius, include_center=False, **kwargs):
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
        include_center : bool
          Flag indicating whether to include the center element.
          Center element is added as first feature. (Default: False)
        **kwargs
          See `Sphere` for additional keyword arguments
        """
        if inner_radius > radius:
            raise ValueError, "inner_radius (got %g) should be smaller " \
                  "than the radius (got %g)" % (inner_radius, radius)
        Sphere.__init__(self, radius, **kwargs)
        self._inner_radius = inner_radius
        self.include_center = include_center

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        return super(HollowSphere, self).__repr__(
            ['inner_radius=%r' % (self._inner_radius,)]
            + _repr_attrs(self, ['include_center'], default=False))

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
        res = array([x for x in tentative_increments
                      if self._inner_radius
                      < self._distance_func(x * element_sizes, center)
                      <= self._radius])

        if not len(res):
            warning("%s defines no neighbors" % self)
        return np.vstack([np.zeros(ndim,dtype='int'),res]) if self.include_center else res


class QueryEngineInterface(object):
    """Very basic class for `QueryEngine`\s defining the interface

    It should not be used directly, but is used to check either we are
    working with QueryEngine instances
    """

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        return _repr(self, *prefixes)

    def train(self, dataset):
        raise NotImplementedError


    def query_byid(self, fid):
        """Return feature ids of neighbors for a given feature id
        """
        raise NotImplementedError


    def query(self, **kwargs):
        """Return feature ids of neighbors given a specific query
        """
        raise NotImplementedError

    #
    # aliases
    #

    def __call__(self, **kwargs):
        return self.query(**kwargs)


    def __getitem__(self, fid):
        return self.query_byid(fid)



class QueryEngine(QueryEngineInterface):
    """Basic class defining interface for querying neighborhood in a dataset

    Derived classes provide specific implementations possibly with trade-offs
    between generality and performance.

    TODO: extend
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
          a dictionary of query objects. Something like
          dict(voxel_indices=Sphere(3))
        """
        super(QueryEngine, self).__init__()
        self._queryobjs = kwargs
        self._queryattrs = {}
        self._ids = None


    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        return super(QueryEngine, self).__repr__(
            prefixes=prefixes
            + ['%s=%r' % v for v in self._queryobjs.iteritems()])

    def __len__(self):
        return len(self._ids) if self._ids is not None else 0

    @property
    def ids(self):
        return self._ids

    def train(self, dataset):
        # reset first
        self._queryattrs.clear()
        # store all relevant attributes
        for space in self._queryobjs:
            self._queryattrs[space] = dataset.fa[space].value
        # execute subclass training
        self._train(dataset)
        # by default all QueryEngines should with all features of the dataset
        # Situation might be different in case of e.g. Surface-based
        # searchlights.
        self._ids = range(dataset.nfeatures)


    def query_byid(self, fid):
        """Return feature ids of neighbors for a given feature id
        """
        queryattrs = self._queryattrs
        kwargs = dict([(space, queryattrs[space][fid])
                       for space in queryattrs])
        return self.query(**kwargs)



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
        self.sorted = sorted
        """Either to sort the query results"""


    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        return super(IndexQueryEngine, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['sorted'], default=True))


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
        # and we need to back-transfer them into dataset ids by subtracting 1
        res = self._searcharray[np.ix_(*slicer)].flatten() - 1
        res = res[res>=0]              # return only the known ones
        if self.sorted:
            return sorted(res)
        else:
            return res


class CachedQueryEngine(QueryEngineInterface):
    """Provides caching facility for query engines.

    Notes
    -----

    This QueryEngine simply remembers the results of the previous
    queries.  Not much checking is done on either datasets it gets in
    :meth:`train` is the same as the on in previous sweep of queries,
    i.e. either none of the relevant for underlying QueryEngine
    feature attributes was modified.  So, CAUTION should be paid to
    avoid calling the same instance of `CachedQueryEngine` on
    different datasets (which might have different masking etc) .

    :func:`query_byid` should be working reliably and without
    surprises.

    :func:`query` relies on hashid of the queries, so there might be a
    collision! Thus consider it EXPERIMENTAL for now.
    """

    def __init__(self, queryengine):
        """
        Parameters
        ----------
        queryengine : QueryEngine
          Results of which engine to cache
        """
        super(CachedQueryEngine, self).__init__()
        self._queryengine = queryengine
        self._trained_ds_fa_hash = None
        """Will give information about either dataset's FA were changed
        """
        self._lookup_ids = None
        self._lookup = None

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        return super(CachedQueryEngine, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['queryengine']))


    def train(self, dataset):
        """'Train' `CachedQueryEngine`.

        Raises
        ------
        ValueError
          If `dataset`'s .fa were changed -- it would raise an
          exception telling to `untrain` explicitly, since the idea is
          to reuse CachedQueryEngine with the same engine and same
          dataset (up to variation of .sa, such as labels permutation)
        """
        ds_fa_hash = idhash_(dataset.fa) + ':%d' % dataset.fa._uniform_length
        if self._trained_ds_fa_hash is None:
            # First time is called
            self._trained_ds_fa_hash = ds_fa_hash
            self._queryengine.train(dataset)     # train the queryengine
            self._lookup_ids = [None] * dataset.nfeatures # lookup for query_byid
            self._lookup = {}           # generic lookup
            self.ids = self.queryengine.ids # used in GNBSearchlight??
        elif self._trained_ds_fa_hash != ds_fa_hash:
            raise ValueError, \
                  "Feature attributes of %s (idhash=%r) were changed from " \
                  "what this %s was trained on (idhash=%r). Untrain it " \
                  "explicitly if you like to reuse it on some other data." \
                  % (dataset, ds_fa_hash, self, self._trained_ds_fa_hash)
        else:
            pass

    def untrain(self):
        """Forgetting that CachedQueryEngine was already trained
        """
        self._trained_ds_fa_hash = None


    @borrowdoc(QueryEngineInterface)
    def query_byid(self, fid):
        v = self._lookup_ids[fid]
        if v is None:
            self._lookup_ids[fid] = v = self._queryengine.query_byid(fid)
        return v

    @borrowdoc(QueryEngineInterface)
    def query(self, **kwargs):
        def to_hashable(x):
            """Convert x to something which dict wouldn't mind"""
            try:
                # silly attempt
                d = {x: None}
                return x
            except TypeError:
                pass

            if isinstance(x, dict):
                # keys are already hashable
                # and sort for deterministic order
                return tuple((k, to_hashable(v))
                             for (k, v) in sorted(x.iteritems()))
            elif is_sequence_type(x):
                return tuple(i for i in x)
            elif np.isscalar(x):
                return x
            return x   # and then wait for the report for it to be added

        # idhash_ is somewhat inappropriate since also relies on id
        # (which we should allow to differ) but ATM failing to hash
        # ndarrays etc
        # k = idhash_(kwargs.items())

        # So let's use verbose version of the beastie (could have been
        # also as simple as __repr__ but afraid that order could be
        # changing etc).  This simple function should be sufficient
        # for our typical use, otherwise we might like to use hashing
        # facilities provided by joblib but for paranoid we would
        # still need to store actual values to resolve collisions
        # which would boil down to the same scenario
        k = to_hashable(kwargs)
        v = self._lookup.get(k, None)
        if v is None:
            self._lookup[k] = v = self._queryengine.query(**kwargs)
        return v

    queryengine = property(fget=lambda self: self._queryengine)



def scatter_neighborhoods(neighbor_gen, coords, deterministic=False):
    """Scatter neighborhoods over a coordinate list.

    Neighborhood seeds (or centers) are placed on coordinates drawn from a
    provided list so that no seed is part of any other neighborhood. Depending
    on the actual shape and size of the neighborhoods, their elements can be
    overlapping, only the seeds (or centers) are guaranteed to be
    non-overlapping with any other neighborhood. This can be used to perform
    sparse sampling of a given space.

    Parameters
    ==========

    neighbor_gen : neighborhood generator
      Callable that return a list of neighborhood element coordinates, when
      called with a seed coordinate (cf. Sphere)
    coords : list
      List of candidate coordinates that can serve as neighborhood seeds or
      elements.
    deterministic : bool
      If true, performs seed placement using an OrderedDict (available in
      Python 2.7 or later) to guarantee deterministic placement of neighborhood
      seeds in consecutive runs with identical input arguments.

    Returns
    =======
    coordinates, indices
      Two lists are returned. The first list contains the choosen seed
      coordinates (a subset of the input coordinates), the second list
      contains the indices of the respective seeds coordinates in the input
      coordinate list. If particular coordinates are present multiple times
      the index list will contain all indices corresponding to these
      coordinates.
    """
    hasher = dict
    if deterministic:
        from collections import OrderedDict
        hasher = OrderedDict

    # put coordinates into a dict for fast lookup
    try:
        # quick test to check whether the given coords are hashable. If not,
        # this test avoids a potentially long list zipping
        _ = {coords[0]: None}
        lookup = hasher()
        _ = [lookup.setdefault(c, list()).append(i) for i, c in enumerate(coords)]
    except TypeError:
        # maybe coords not hashable?
        lookup = hasher()
        _ = [lookup.setdefault(tuple(c), list()).append(i) for i, c in enumerate(coords)]

    seeds = []
    while len(lookup):
        # get any remaining coordinate
        # with OrderedDict popitem will return the last inserted item by default
        seed, idx = lookup.popitem()
        # remove all coordinates in the neighborhood
        _ = [lookup.pop(c, None) for c in neighbor_gen(seed)]
        # store seed
        seeds.append((seed, idx))
    # unzip coords and idx again
    coords, idx = zip(*seeds)
    # we need a flat idx list
    # yoh: sum trick replaced list(itertools.chain.from_iterable(idx))
    #      which is not python2.5-compatible
    idx = sum(idx, [])
    return coords, idx
