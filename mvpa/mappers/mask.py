# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Data mapper which applies mask to the data"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.mappers.base import Mapper
from mvpa.base.dochelpers import enhancedDocString
from mvpa.misc.support import isInVolume

if __debug__:
    from mvpa.base import debug, warning
    from mvpa.misc.support import isSorted


class MaskMapper(Mapper):
    """Mapper which uses a binary mask to select "Features" """

    def __init__(self, mask, **kwargs):
        """Initialize MaskMapper

        :Parameters:
          mask : array
            an array in the original dataspace and its nonzero elements are
            used to define the features included in the dataset
        """
        Mapper.__init__(self, **kwargs)

        self.__mask = self.__maskdim = self.__masksize = \
                      self.__masknonzerosize = self.__forwardmap = \
                      self.__masknonzero = None # to make pylint happy
        self._initMask(mask)


    __doc__ = enhancedDocString('MaskMapper', locals(), Mapper)


    def __str__(self):
        return "MaskMapper: %d -> %d" \
            % (self.__masksize, self.__masknonzerosize)

    def __repr__(self):
        s = super(MaskMapper, self).__repr__()
        return s.replace("(", "(mask=%s," % self.__mask, 1)

# XXX
# XXX HAS TO TAKE CARE OF SUBCLASSES!!!
# XXX
#
#    def __deepcopy__(self, memo=None):
#        # XXX memo does not seem to be used
#        if memo is None:
#            memo = {}
#        from mvpa.support.copy import deepcopy
#        out = MaskMapper.__new__(MaskMapper)
#        Mapper.__init__(out)
#        out.__mask = self.__mask.copy()
#        out.__maskdim = self.__maskdim
#        out.__masksize = self.__masksize
#        out.__masknonzero = deepcopy(self.__masknonzero)
#        out.__masknonzerosize = self.__masknonzerosize
#        out.__forwardmap = self.__forwardmap.copy()
#
#        return out


    def _initMask(self, mask):
        """Initialize internal state with mask-derived information

        It is needed to initialize structures for the fast
        and reverse lookup to don't impose performance hit on any
        future operation
        """
        # NOTE: If any new class member are added here __deepcopy__() has to
        #       be adjusted accordingly!

        self.__mask = (mask != 0)
        self.__maskdim = len(mask.shape)
        self.__masksize = N.prod(mask.shape)

        # Following introduces space penalty but are needed
        # for efficient processing.
        # Store all coordinates for backward mapping
        self.__masknonzero = mask.nonzero()
        self.__masknonzerosize = len(self.__masknonzero[0])
        #from IPython.Shell import IPShellEmbed
        #ipshell = IPShellEmbed()
        #ipshell()
        #import pydb; pydb.debugger()
        # Store forward mapping (ie from coord into outId)
        # TODO to save space might take appropriate int type
        #     depending on masknonzerosize
        # it could be done with a dictionary, but since mask
        # might be relatively big, it is better to simply use
        # a chunk of RAM ;-)
        self.__forwardmap = N.zeros(mask.shape, dtype=N.int64)
        # under assumption that we +1 values in forwardmap so that
        # 0 can be used to signal outside of mask

        self.__forwardmap[self.__masknonzero] = \
            N.arange(self.__masknonzerosize)


    def _forward_data(self, data):
        """Map data from the original dataspace into featurespace.
        """
        data = N.asanyarray(data)          # assure it is an array
        datadim = len(data.shape)
        datashape = data.shape[(-1)*self.__maskdim:]
        if not datashape == self.__mask.shape:
            raise ValueError, \
                  "The shape of data to be mapped %s " % `datashape` \
                  + " does not match the mapper's mask shape %s" \
                    % `self.__mask.shape`

        if self.__maskdim == datadim:
            # we had to select by __masknonzero if we didn't sort
            # Ids and wanted to preserve the order
            #return data[ self.__masknonzero ]
            return data[ self.__mask ]
        elif self.__maskdim+1 == datadim:
            # XXX XXX XXX below line should be accomodated also
            # to make use of self.__masknonzero instead of
            # plain mask if we want to preserve the (re)order
            return data[ :, self.__mask ]
        else:
            raise ValueError, \
                  "Shape of the to be mapped data, does not match the " \
                  "mapper mask. Only one (optional) additional dimension " \
                  "exceeding the mask shape is supported."


    def _reverse_data(self, data):
        """Reverse map data from featurespace into the original dataspace.
        """
        data = N.asanyarray(data)
        datadim = len(data.shape)
        if not datadim in [1, 2]:
            raise ValueError, \
                  "Only 2d or 1d data can be reverse mapped. "\
                  "Got data of shape %s" % (data.shape,)

        if datadim == 1:
            # Verify that we are trying to reverse data of proper dimension.
            # In 1D case numpy would not complain and will broadcast
            # the values
            if __debug__ and  self.nfeatures != len(data):
                raise ValueError, \
                      "Cannot reverse map data with %d elements, whenever " \
                      "mask knows only %d" % (len(data), self.nfeatures)
            mapped = N.zeros(self.__mask.shape, dtype=data.dtype)
            mapped[self.__mask] = data
        elif datadim == 2:
            mapped = N.zeros(data.shape[:1] + self.__mask.shape,
                             dtype=data.dtype)
            mapped[:, self.__mask] = data

        return mapped


    def getInSize(self):
        """InShape is a shape of original mask"""
        return self.__masksize


    def getOutSize(self):
        """OutSize is a number of non-0 elements in the mask"""
        return self.__masknonzerosize


    def getMask(self, copy = True):
        """By default returns a copy of the current mask.

        If 'copy' is set to False a reference to the mask is returned instead.
        This shared mask must not be modified!
        """
        if copy:
            return self.__mask.copy()
        else:
            return self.__mask


    def getInId(self, outId):
        """Returns a features coordinate in the original data space
        for a given feature id.

        If this method is called with a list of feature ids it returns a
        2d-array where the first axis corresponds the dimensions in 'In'
        dataspace and along the second axis are the coordinates of the features
        on this dimension (like the output of NumPy.array.nonzero()).

        XXX it might become __get_item__ access method

        """
        # XXX Might be improved by storing also transpose of
        # __masknonzero
        return N.array([self.__masknonzero[i][outId]
                        for i in xrange(self.__maskdim)])


    def getInIds(self):
        """Returns a 2d array where each row contains the coordinate of the
        feature with the corresponding id.
        """
        return N.transpose(self.__masknonzero)


    def isValidInId(self, inId):
        mask = self.mask
        return (isInVolume(inId, mask.shape) and mask[tuple(inId)] != 0)


    def getOutId(self, coord):
        """Translate a feature mask coordinate into a feature ID.
        """
        # FIXME Since lists/arrays accept negative indexes to go from
        # the end -- we need to check coordinates explicitely. Otherwise
        # we would get warping effect
        try:
            tcoord = tuple(coord)
            if self.__mask[tcoord] == 0:
                raise ValueError, \
                      "The point %s didn't belong to the mask" % (`coord`)
            return self.__forwardmap[tcoord]
        except TypeError:
            raise ValueError, \
                  "Coordinates %s are of incorrect dimension. " % `coord` + \
                  "The mask has %d dimensions." % self.__maskdim
        except IndexError:
            raise ValueError, \
                  "Coordinates %s are out of mask boundary. " % `coord` + \
                  "The mask is of %s shape." % `self.__mask.shape`


    def selectOut(self, outIds):
        """Only listed outIds would remain.

        *Function assumes that outIds are sorted*. In __debug__ mode selectOut
        would check if obtained IDs are sorted and would warn the user if they
        are not.

        .. note::
          If you feel strongly that you need to remap features
          internally (ie to allow Ids with mixed order) please contact
          developers of mvpa to discuss your use case.

          The function used to accept a matrix-mask as the input but now
          it really has to be a list of IDs

        Feature/Bug:
         * Negative outIds would not raise exception - just would be
           treated 'from the tail'
        """
        if __debug__ and 'CHECK_SORTEDIDS' in debug.active:
            # per short conversation with Michael -- we should not
            # allow reordering since we saw no viable use case for
            # it. Thus -- warn user is outIds are not in sorted order
            # and no sorting was requested may be due to performance
            # considerations
            if not isSorted(outIds):
                warning("IDs for selectOut must be provided " +
                        "in sorted order, otherwise .forward() would fail"+
                        " on the data with multiple samples")

        # adjust mask and forwardmap
        discarded = N.array([ True ] * self.nfeatures)
        discarded[outIds] = False    # create a map of discarded Ids
        discardedin = tuple(self.getInId(discarded))
        # XXX need to perform copy on write; implement properly during rewrite
        #self.__mask[discardedin] = False
        mask = self.__mask.copy()
        mask[discardedin] = False
        self.__mask = mask

        # XXX for now do evil expanding of slice args
        # this should be implemented properly when rewriting MaskMapper into
        # a 1D -> 1D Mapper
        if isinstance(outIds, slice):
            outIds = range(*outIds.indices(self.getOutSize()))

        # XXX for now do evil expanding of selection arrays
        # this should be implemented properly when rewriting MaskMapper into
        # a 1D -> 1D Mapper
        if isinstance(outIds, N.ndarray):
            outIds = outIds.nonzero()[0]

        self.__masknonzerosize = len(outIds)
        self.__masknonzero = [ x[outIds] for x in self.__masknonzero ]

        # adjust/remap not discarded in forwardmap
        # since we merged _tent/maskmapper-init-noloop it is not necessary
        # to zero-out discarded entries since we anyway would check with mask
        # in getOutId(s)
        # XXX this also needs to do copy on write; reimplement properly during
        # rewrite
        #self.__forwardmap[self.__masknonzero] = \
        #    N.arange(self.__masknonzerosize)
        fmap = self.__forwardmap.copy()
        fmap[self.__masknonzero] = N.arange(self.__masknonzerosize)
        self.__forwardmap = fmap


    def discardOut(self, outIds):
        """Listed outIds would be discarded

        """

        # adjust mask and forwardmap
        discardedin = tuple(self.getInId(outIds))
        self.__mask[discardedin] = False
        # since we merged _tent/maskmapper-init-noloop it is not necessary
        # to zero-out discarded entries since we anyway would check with mask
        # in getOutId(s)
        # self.__forwardmap[discardedin] = 0

        self.__masknonzerosize -= len(outIds)
        self.__masknonzero = [ N.delete(x, outIds)
                               for x in self.__masknonzero ]

        # adjust/remap not discarded in forwardmap
        self.__forwardmap[self.__masknonzero] = \
                                              N.arange(self.__masknonzerosize)

        # OPT: we can adjust __forwardmap only for ids which are higher than
        # the smallest outId among discarded. Similar strategy could be done
        # for selectOut but such index has to be figured out first there
        #      ....


# comment out for now... introduce when needed
#    def getInEmpty(self):
#        """Returns empty instance of input object"""
#        raise NotImplementedError
#
#
#    def getOutEmpty(self):
#        """Returns empty instance of output object"""
#        raise NotImplementedError


    def convertOutIds2OutMask(self, outIds):
        """Returns a boolean mask with all features in `outIds` selected.

        :Parameters:
            outIds: list or 1d array
                To be selected features ids in out-space.

        :Returns:
            ndarray: dtype='bool'
                All selected features are set to True; False otherwise.
        """
        fmask = N.repeat(False, self.nfeatures)
        fmask[outIds] = True

        return fmask


    def convertOutIds2InMask(self, outIds):
        """Returns a boolean mask with all features in `ouIds` selected.

        This method works exactly like Mapper.convertOutIds2OutMask(), but the
        feature mask is finally (reverse) mapped into in-space.

        :Parameters:
            outIds: list or 1d array
                To be selected features ids in out-space.

        :Returns:
            ndarray: dtype='bool'
                All selected features are set to True; False otherwise.
        """
        return self.reverse(self.convertOutIds2OutMask(outIds))


    # Read-only props
    mask = property(fget=lambda self:self.getMask(False))


    # TODO Unify tuple/array conversion of coordinates. tuples are needed
    #      for easy reference, arrays are needed when doing computation on
    #      coordinates: for some reason numpy doesn't handle casting into
    #      array from tuples while performing arithm operations...


class FeatureSubsetMapper(Mapper):
    """Mapper to select a subset of features.

    This mapper only operates on one-dimensional samples or two-dimensional
    samples matrices. If necessary it can be combined for FlattenMapper to
    handle multidimensional data.
    """
    def __init__(self, mask):
        """
        Parameters
        ----------
        mask : array
          This is a one-dimensional array whos non-zero elements define both
          the feature subset and the data shape the mapper is going to handle.
        """
        Mapper.__init__(self)
        self.__forwardmap = None

        if not len(mask.shape) == 1:
            raise ValueError("The mask has to be a one-dimensional vector. "
                             "For multidimensional data consider FlattenMapper "
                             "before running SubsetMapper.")
        self.__mask = (mask != 0)
        self.__masknonzerosize = N.sum(self.__mask) # number of non-zeros


    def _init_forwardmap(self):
        """Init the forward coordinate map needed for id translation.

        This is a separate function for lazy-computation of this map
        """
        # Store forward mapping (ie from coord into outId)
        # we do not initialize to save some CPU-cycles, especially for
        # large datasets -- This should be save, since we only need it
        # for coordinate lookup that is limited to non-zero mask elements
        # which are initialized next
        self.__forwardmap = N.empty(self.__mask.shape, dtype=N.int64)
        self.__forwardmap[self.__mask] = N.arange(self.__masknonzerosize)


    def _forward_data(self, data):
        """Map data from the original dataspace into featurespace.

        Parameters
        ----------
        data : array-like
          Either one-dimensional sample or two-dimensional samples matrix.
        """
        datadim = len(data.shape)
        # single sample and matching data shape
        if datadim == 1 and data.shape == self.__mask.shape:
            return data[self.__mask]
        # multiple samples and matching sample shape
        elif datadim == 2 and data.shape[1:] == self.__mask.shape:
            return data[:, self.__mask]
        else:
            raise ValueError(
                  "Shape of the to be mapped data, does not match the mapper "
                  "mask %s. Only one (optional) additional dimension (for "
                  "multiple samples) exceeding the mask shape is supported."
                    % `self.__mask.shape`)


    def _reverse_data(self, data):
        """Reverse map data from featurespace into the original dataspace.

        Parameters
        ----------
        data : array-like
          Either one-dimensional sample or two-dimensional samples matrix.
        """
        datadim = len(data.shape)

        # single sample, matching shape
        if datadim == 1 and len(data) == self.__masknonzerosize:
            # Verify that we are trying to reverse data of proper dimension.
            # In 1D case numpy would not complain and will broadcast
            # the values
            mapped = N.zeros(self.__mask.shape, dtype=data.dtype)
            mapped[self.__mask] = data
            return mapped
        # multiple samples, matching shape
        elif datadim == 2 and len(data[0]) == self.__masknonzerosize:
            mapped = N.zeros(data.shape[:1] + self.__mask.shape,
                             dtype=data.dtype)
            mapped[:, self.__mask] = data
            return mapped
        else:
            raise ValueError(
                  "Only 2d or 1d data can be reverse mapped. Additionally, "
                  "each sample vector has to match the size of the mask. "
                  "Got data of shape %s, when mask is %s"
                  % (data.shape, self.__mask.shape))


    def get_outsize(self):
        """OutSize is a number of non-0 elements in the mask"""
        return self.__masknonzerosize


    def get_mask(self, copy=True):
        """By default returns a copy of the current mask.

        Parameters
        ----------
        copy : bool
          If False a reference to the mask is returned, a copy otherwise.
          This shared mask must not be modified!
        """
        if copy:
            return self.__mask.copy()
        else:
            return self.__mask


    def is_valid_outid(self, id):
        """Return whether a particular id is a valid output id/coordinate.

        Parameters
        ----------
        id : int
        """
        return id >=0 and id < self.get_outsize()


    def is_valid_inid(self, id):
        """Return whether a particular id is a valid input id/coordinate.

        Parameters
        ----------
        id : int
        """
        # if it exceeds range it cannot be right
        if not (id >=0 and id < len(self.__mask)):
            return False
        # otherwise look into the mask
        return self.__mask[id]


    def get_outids(self, in_id):
        """Translate an input id/coordinate into an output id/coordinate.

        Parameters
        ----------
        in_id : int

        Returns
        -------
        list
          A one-element list with an integer id.
        """
        if not self.is_valid_inid(in_id):
            raise ValueError("Input id/coordinated '%s' is not mapped into the "
                             "the output space (i.e. not part of the mask)."
                             % str(in_id))
        # lazy forward mapping
        if self.__forwardmap is None:
            self._init_forwardmap()
        return [self.__forwardmap[in_id]]


    def select_out(self, slicearg, cow=True):
        """Limit the feature subset selection.

        Parameters
        ----------
        slicearg : array(bool), list, slice
          Any valid Numpy slicing argument defining a subset of the current
          feature set.
        cow: bool
          For internal use only!
          If `True`, it is safe to call the function on a shallow copy of
          another FeatureSubsetMapper instance without affecting the original
          mapper instance. If `False`, modifications done to one instance
          invalidate the other.
        """
        # create a boolean mask covering all present 'out' features
        # before subset selection
        submask = N.zeros(self.__masknonzerosize, dtype='bool')
        # now apply the slicearg to it to enable the desired features
        # everything slicing that numpy supports is possible and even better
        # the order of ids in the case of a list argument is irrelevant
        submask[slicearg] = True

        if cow:
            mask = self.__mask.copy()
            # do not update the forwardmap right away but wait till it becomes
            # necessary
            self.__forwardmap = forwardmap = None
        else:
            mask = self.__mask
            forwardmap = self.__forwardmap

        # adjust the mask: self-masking will give current feature set, assigning
        # the desired submask will do incremental masking
        mask[mask] = submask
        self.__masknonzerosize = N.sum(submask)
        # regenerate forward map if not COW and lazy computing -- old values
        # can remain since we only consider values with a corresponding mask
        # element that is True
        if not forwardmap is None:
            forwardmap[mask] = N.arange(self.__masknonzerosize)

        # reassign
        self.__mask = mask
        self.__forwardmap = forwardmap

