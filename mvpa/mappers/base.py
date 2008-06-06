#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Data mapper"""

__docformat__ = 'restructuredtext'

import numpy as N
from mvpa.misc.vproperty import VProperty
from mvpa.base.dochelpers import enhancedDocString

if __debug__:
    from mvpa.misc import warning
    from mvpa.misc.support import isSorted



class Mapper(object):
    """Interface to provide mapping between two spaces: in and out.
    Methods are prefixed correspondingly. forward/reverse operate
    on the entire dataset. get(In|Out)Id[s] operate per element::

              forward
        in   ---------> out
             <--------/
               reverse

    Subclasses should define 'dsshape' and 'nfeatures' properties that point to
    `getInShape` and `getOutSize` respectively. This cannot be
    done in the baseclass as standard Python properties would still point to
    the baseclass methods.
    """
    def __init__(self):
        """Does nothing."""
        pass


    __doc__ = enhancedDocString('Mapper', locals())


    def forward(self, data):
        """Map data from the original dataspace into featurespace.
        """
        raise NotImplementedError


    def __call__(self, data):
        """Calls the mappers forward() method.
        """
        return self.forward(data)


    def reverse(self, data):
        """Reverse map data from featurespace into the original dataspace.
        """
        raise NotImplementedError


    def train(self, dataset):
        """Sub-classes have to override this method if the mapper need
        training.
        """
        pass


    def getInShape(self):
        """Returns the dimensionality specification of the original dataspace.

        XXX -- should be deprecated and  might be substituted
        with functions like  getEmptyFrom / getEmptyTo
        """
        raise NotImplementedError


    def getOutShape(self):
        """
        Returns the shape (or other dimensionality speicification)
        of the destination dataspace.
        """
        raise NotImplementedError


    def getInSize(self):
        """Returns the size of the entity in input space"""
        raise NotImplementedError


    def getOutSize(self):
        """Returns the size of the entity in output space"""
        raise NotImplementedError


    def selectOut(self, outIds):
        """Remove some elements and leave only ids in 'out'/feature space"""
        raise NotImplementedError


    nfeatures = VProperty(fget=getOutSize)



class MaskMapper(Mapper):
    """Mapper which uses a binary mask to select "Features" """

    def __init__(self, mask):
        """Initialize MaskMapper

        :Parameters:
          mask : array
            an array in the original dataspace and its nonzero elements are
            used to define the features included in the dataset
        """
        Mapper.__init__(self)


        self.__mask = self.__maskdim = self.__masksize = \
                      self.__masknonzerosize = self.__forwardmap = \
                      self.__masknonzero = None # to make pylint happy
        self._initMask(mask)


    __doc__ = enhancedDocString('MaskMapper', locals(), Mapper)


    def __str__(self):
        return "MaskMapper: %d -> %d" \
            % (self.__masksize, self.__masknonzerosize)


# XXX
# XXX HAS TO TAKE CARE OF SUBCLASSES!!!
# XXX
#
#    def __deepcopy__(self, memo=None):
#        # XXX memo does not seem to be used
#        if memo is None:
#            memo = {}
#        from copy import deepcopy
#        out = MaskMapper.__new__(MaskMapper)
#        Mapper.__init__(out)
#        out.__mask = self.__mask.copy()
#        out.__maskdim = self.__maskdim
#        out.__masksize = self.__masksize
#        out.__masknonzero = deepcopy(self.__masknonzero)
#        out.__masknonzerosize = self.__masknonzerosize
#        out.__forwardmap = self.__forwardmap.copy()

        return out


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


    def forward(self, data):
        """Map data from the original dataspace into featurespace.
        """
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


    def reverse(self, data):
        """Reverse map data from featurespace into the original dataspace.
        """
        datadim = len(data.shape)
        if not datadim in [1, 2]:
            raise ValueError, \
                  "Only 2d or 1d data can be reverse mapped."

        if datadim == 1:
            mapped = N.zeros(self.__mask.shape, dtype=data.dtype)
            mapped[self.__mask] = data
        elif datadim == 2:
            mapped = N.zeros(data.shape[:1] + self.__mask.shape,
                             dtype=data.dtype)
            mapped[:, self.__mask] = data

        return mapped


    def getInShape(self):
        """InShape is a shape of original mask"""
        return self.__mask.shape


    def getInSize(self):
        """InShape is a shape of original mask"""
        return self.__masksize


    def getOutShape(self):
        """OutShape is a shape of target dataset"""
        # should worry about state-full class.
        # TODO: add exception 'InvalidStateError' which is raised
        #       by some class if state is not yet defined:
        #         classifier has not yet been trained
        #         mapped yet see the dataset
        raise NotImplementedError


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


    def selectOut(self, outIds, sort=False):
        """Only listed outIds would remain.

        The function used to accept a matrix-mask as the input but now
        it really has to be a list of IDs

        Function assumes that outIds are sorted. If not - please set
        sort to True. While in __debug__ mode selectOut would check if
        obtained IDs are sorted and would warn the user if they are
        not.

        If you feel strongly that you need to remap features
        internally (ie to allow Ids with mixed order) please contact
        developers of mvpa to discuss your use case.

        See `tests.test_maskmapper.testSelectOrder` for basic testing

        Feature/Bug:
         * Negative outIds would not raise exception - just would be
           treated 'from the tail'

        Older comments on 'order' - might be useful in future if
        reordering gets ressurrected
        Order will be taken into account -- ie items will be
        remapped if order was changed... need to check if neighboring
        still works... no -- it doesn't. For the data without samples
        .forward can be easily adjusted by using masknonzero instead of
        plain mask, but for data with samplesI don't see a clean way...
        see forward() above... there is no testcase for order preservation
        for DIM+1 case
        """
        if sort:
            outIds.sort()
        elif __debug__:
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
        self.__mask[discardedin] = False

        self.__masknonzerosize = len(outIds)
        self.__masknonzero = [ x[outIds] for x in self.__masknonzero ]

        # adjust/remap not discarded in forwardmap
        # since we merged _tent/maskmapper-init-noloop it is not necessary
        # to zero-out discarded entries since we anyway would check with mask
        # in getOutId(s)
        self.__forwardmap[self.__masknonzero] = \
            N.arange(self.__masknonzerosize)


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
    # TODO: refactor the property names? make them vproperty?
    dsshape = property(fget=getInShape)
    mask = property(fget=lambda self:self.getMask(False))


    # TODO Unify tuple/array conversion of coordinates. tuples are needed
    #      for easy reference, arrays are needed when doing computation on
    #      coordinates: for some reason numpy doesn't handle casting into
    #      array from tuples while performing arithm operations...


