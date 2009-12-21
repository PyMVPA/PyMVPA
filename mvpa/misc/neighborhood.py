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
import sys

from mvpa.clfs.distance import cartesianDistance

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
    def __init__(self, diameter, extent=(sys.maxint, sys.maxint, sys.maxint)):
        """ Initialise the Sphere

        Parameters
        ----------
        diameter : odd int
            diameter of the sphere in voxels
        extent :  sequence of 3 ints
            maximum index to consider
            if this is not provided it will be the maximum value of an integer

        """
        self.extent = N.asanyarray(extent)
        if __debug__:
            # XXX YOH: hm...  I had in mind a bit better situation: didn't we
            #     scale with some dimension information? could not Sphere
            #     neighborhood generator be provided with such to generate an
            #     ellipsoid?
            if diameter % 2 != 1 or type(diameter) is not int:
                raise ValueError("Sphere diameter must be odd integer, but "
                                 "got %s of type %s"
                                 % (diameter, type(diameter)))
            # XXX YOH: may be sphere could easily be extended to become a
            #     hypersphere? ie generalize and be used for 2D, 3D, ... ND?
            #     Sure thing - using utility would simply puke (with some
            #     meaningful message) if dimensions mismatch
            if self.extent.size != 3 \
                or self.extent.dtype.char not in N.typecodes['AllInteger']:
                raise ValueError("Sphere extent must be 3 integers, was: %s"
                                % str(extent))
        self.diameter = diameter
        self.radius = diameter/2
        self.coord_list = self._create_template()
        self.dataset = None

    def _create_template(self):
        center = array((0, 0, 0))
        lr = range(-self.radius, self.radius+1) # linear range
        # TODO create additional distance metrics, for example manhatten
        # XXX YOH: we have those elsewhere... I guess we need to unify access to them
        # TODO create a way to specify shape of quantised sphere i.e. < vs <=
        return array([array((i, j, k)) for i in lr
                              for j in lr
                              for k in lr
                              if cartesianDistance(array((i, j, k)),center)
                                 <= self.radius])

    def train(self, dataset):
        # XXX techinically this is not needed
        # XXX YOH:  yeap -- BUT if you care about my note above on extracting
        #     somehow sizes -- some dataset.a might come handy may be?
        #     so actual template get constructed in train and _create_template
        #     could go away and just be returned in some R/O property
        self.dataset = dataset

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
        coordinate = N.asanyarray(coordinate)
        if __debug__:
            if coordinate.size != 3 \
            or coordinate.dtype.char not in N.typecodes['AllInteger']:
                raise ValueError("Sphere must be called on a sequence of "
                                 "integers of length 3, you gave %s "
                                 % coordinate)
            #if dataset is None:
            #    raise ValueError("Sphere object has not been trained yet, use "
            #                     "train(dataset) first. ")
        # function call
        coord_array = (coordinate + self.coord_list)
        # now filter out illegal coordinates if they really are outside the
        # bounds
        if (coordinate - self.radius < 0).any() \
        or (coordinate + self.radius >= self.extent).any():
            coord_array = array([c for c in coord_array \
                                   if (c >= 0).all()
                                   and (c < self.extent).all()])

        coord_array = coord_array.transpose()
        return zip(coord_array[0], coord_array[1], coord_array[2])


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



class IndexQueryEngine_(QueryEngine):
    """Provides efficient query engine for discrete spaces.

    TODO:
    - extend documentation
    - repr
    - RENAME -- Index... may be Discrete ? index might not imply
                order/distance may be?
    """

    def __init__(self, **kwargs):
        QueryEngine.__init__(self, **kwargs)
        self._spaceaxis = None          # XXX might not be needed
        self._spaceorder = None
        """Order of the spaces"""

    def _train(self, dataset):
        # local binding
        qattrs = self._queryattrs
        # in addition to the base class functionality we need to store the
        # order of the query-spaces
        self._spaceorder = qattrs.keys()
        # type check and determine mask dimensions
        max_ind = []
        dims = []                       # dimensionality of each space
        selector = []
        for space in self._spaceorder:
            # local binding for the attribute
            qattr = qattrs[space]
            if not qattr.dtype.char in N.typecodes['AllInteger']:
                raise ValueError("IndexQueryEngine can only operate on "
                                 "feature attributes with integer indices "
                                 "(got: %s)." % str(qattr.dtype))
            # determine the dimensions of this space
            # and charge the nonzero selector
            dim = qattr.max(axis=0)
            if N.isscalar(dim):
                max_ind.append([dim])
                dims.append(dim)
                selector.append(qattr.T)
            else:
                max_ind.append(dim)
                dims.extend(dim)
                selector.extend(qattr.T)
        # now check whether we have sufficient information to put each feature
        # id into one unique search array element
        dims = N.array(dims) + 1
        # we can deal with less features (e.g. masked dataset, but not more)
        # XXX (yoh): seems to be too weak of a check... pretty much you are trying
        #            to check either 2 features do not collide in the target
        #            "mask", right?
        if N.prod(dims) < dataset.nfeatures:
            raise ValueError("IndexQueryEngine has insufficient information "
                             "about the dataset spaces. It is required to "
                             "specify an ROI generator for each feature space "
                             "in the dataset (got: %s, #describale: %i, "
                             "#actual features: %i)."
                             % (str(self._spaceorder), N.prod(dims),
                                   dataset.nfeatures))
        # now we can create the search array
        self._searcharray = N.zeros(dims, dtype='int')
        # and fill it with feature ids, but start from ONE to be different from
        # the zeros
        self._searcharray[tuple(selector)] = N.arange(1, dataset.nfeatures + 1)
        # store the dimensions, and hence number of axis per space
        self._spaceaxis = zip(self._spaceorder, max_ind)


    def query(self, **kwargs):
        # construct the search array slicer
        # need to obey axis order
        slicer = []
        for space, max_ind in self._spaceaxis:
            # only generate ROI, if we have a generator
            # otherwise consider all of the unspecified space
            if space in kwargs:
                # if no ROI generator is available, take argument as is
                if self._queryobjs[space] is None:
                    # XXX this is wrong, it should be imporant how many axis
                    # that space covers and not if the slicing arg is scalar
                    if N.isscalar(kwargs[space]):
                        slicer.append(kwargs[space])
                    else:
                        slicer.extend(kwargs[space])
                else:
                    roi = self._queryobjs[space](kwargs[space])
                    # filter the results for validity
                    # XXX might be made conditional
                    roi = [i for i in roi if (i <= max_ind).all() and i >= 0]
                    # if no candidate is left, the whole thing does not match
                    # regardless of the other spaces
                    if not len(roi):
                        return []
                    # need to get access to per dim indices
                    roi = N.transpose(roi)
                    for i in xrange(len(max_ind)):
                        slicer.append(roi[i])
            else:
                # Provide ":" for all dimensions in a given space
                slicer += [slice(None)]*len(max_ind)
        # only ids are of interest -> flatten
        # and we need to back-transfer them into dataset ids by substracting 1
        return self._searcharray[slicer].flatten() - 1


class IndexQueryEngine(QueryEngine):
    """Provides efficient query engine for discrete spaces.

    Uses dictionary lookups for elements present/known to the
    attributes.

    TODO:
    - extend documentation
    - repr
    - RENAME -- Index... may be Discrete ? index might not imply
                order/distance may be?
    """

    def __init__(self, **kwargs):
        QueryEngine.__init__(self, **kwargs)
        self._spaceorder = None
        """Order of the spaces"""
        self._lookups = {}
        """Dictionary of lookup dictionaries per each space"""
        self._sliceall = {}
        """Precrafted indexes to cover ':' situation within ix_"""
        self._searcharray = None
        """Actual searcharray"""

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
            # XXX would probably work for ANY discrete attribute
            if not qattr.dtype.char in N.typecodes['AllInteger']:
                raise ValueError("IndexQueryEngine can only operate on "
                                 "feature attributes with integer indices "
                                 "(got: %s)." % str(qattr.dtype))
            # If it is >1D ndarray we need to transform to list of tuples,
            # since ndarray is not hashable
            if isinstance(qattr, N.ndarray) and len(qattr.shape) > 1:
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
            sliceall[space] = N.arange(dim)
            # And fill out selector using current values from qattr
            selector.append([lookup[x] for x in qattr])

        # now check whether we have sufficient information to put each feature
        # id into one unique search array element
        dims = N.array(dims)
        # we can deal with less features (e.g. masked dataset, but not more)
        # XXX (yoh): seems to be too weak of a check... pretty much you are trying
        #            to check either 2 features do not collide in the target
        #            "mask", right?
        if N.prod(dims) < dataset.nfeatures:
            raise ValueError("IndexQueryEngine has insufficient information "
                             "about the dataset spaces. It is required to "
                             "specify an ROI generator for each feature space "
                             "in the dataset (got: %s, #describale: %i, "
                             "#actual features: %i)."
                             % (str(self._spaceorder), N.prod(dims),
                                   dataset.nfeatures))
        # now we can create the search array
        self._searcharray = N.zeros(dims, dtype='int')
        # and fill it with feature ids, but start from ONE to be different from
        # the zeros
        self._searcharray[tuple(selector)] = N.arange(1, dataset.nfeatures + 1)
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
        for space_ind, space in enumerate(self._spaceorder):
            lookup = self._lookups[space]
            # only generate ROI, if we have a generator
            # otherwise consider all of the unspecified space
            if space in kwargs:
                # if no ROI generator is available, take provided indexes
                # without any additional neighbors etc
                if self._queryobjs[space] is None:
                    slicer.append(N.atleast_1d(kwargs[space]))
                else:
                    roi = self._queryobjs[space](kwargs[space])
                    # XXX -- convert to tuples
                    #if roi.shape > 1:
                    #    roi = [tuplex) for x in roi]
                    # filter the results for validity
                    roi_ind = [lookup[i] for i in roi if (i in lookup)]
                    # if no candidate is left, the whole thing does not match
                    # regardless of the other spaces
                    if not len(roi_ind):
                        return []
                    slicer.append(roi_ind)
            else:
                # Provide ":" for ... XXX
                slicer.append(self._sliceall[space])
        # only ids are of interest -> flatten
        # and we need to back-transfer them into dataset ids by substracting 1
        return self._searcharray[N.ix_(*slicer)].flatten() - 1
