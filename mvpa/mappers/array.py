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
from operator import isSequenceType

from mvpa.mappers.metric import MetricMapper
from mvpa.datasets.metric import DescreteMetric, cartesianDistance
from mvpa.base.dochelpers import enhancedDocString

if __debug__:
    from mvpa.misc import warning
    from mvpa.misc.support import isSorted



class DenseArrayMapper(MetricMapper):
    """Mapper for equally spaced dense arrays."""

    def __init__(self, mask, metric=None,
                 distance_function=cartesianDistance, elementsize=None):
        """Initialize DenseArrayMapper

        :Parameters:
          mask : array
            an array in the original dataspace and its nonzero elements are
            used to define the features included in the dataset
          metric : Metric
            Corresponding metric for the space. No attempt is made to
            determine whether a certain metric is reasonable for this
            mapper. If `metric` is None -- `DescreteMetric`
            is constructed that assumes an equal (1) spacing of all mask
            elements with a `distance_function` given as a parameter listed
            below.
          distance_function : functor
            Distance function to use as the parameter to
            `DescreteMetric` if `metric` is not specified,
          elementsize : list or scalar
            Determines spacing within `DescreteMetric`. If it is given as a
            scalar, corresponding value is assigned to all dimensions, which
            are found within `mask`

        :Note: parameters `elementsize` and `distance_function` are relevant
               only if `metric` is None
        """
        if metric == None:
            if elementsize is None:
                elementsize = [1]*len(mask.shape)
            else:
                if isSequenceType(elementsize):
                    if len(elementsize) != len(mask.shape):
                        raise ValueError, \
                              "Number of elements in elementsize [%d]" % \
                              len(elementsize) + " doesn't match shape " + \
                              "of the mask [%s]" % (`mask.shape`)
                else:
                    elementsize = [ elementsize ] * len(mask.shape)
            metric = DescreteMetric(elementsize=[1]*len(mask.shape),
                                     distance_function=distance_function)

        MetricMapper.__init__(self, mask, metric)


    __doc__ = enhancedDocString('DenseArrayMapper', locals(), MetricMapper)


    def __str__(self):
        return "DenseArrayMapper: %d -> %d" \
            % (self.getInSize(), self.getOutSize())


#    def __deepcopy__(self, memo=None):
#        # XXX memo does not seem to be used
#        if memo is None:
#            memo = {}
#        from mvpa.misc.copy import deepcopy
#        # XXX might be necessary to deepcopy 'self.metric' as well
#        # to some degree reimplement the constructor to prevent calling the
#        # expensive _initMask() again
#        out = MaskMapper.__new__(MaskMapper)
#        MetricMapper.__init__(out, self.metric)
#        out.__mask = self.__mask.copy()
#        out.__maskdim = self.__maskdim
#        out.__masksize = self.__masksize
#        out.__masknonzero = deepcopy(self.__masknonzero)
#        out.__masknonzerosize = self.__masknonzerosize
#        out.__forwardmap = self.__forwardmap.copy()
#
#        return out


    def getNeighborIn(self, inId, radius=0):
        """Return the list of coordinates for the neighbors.
        XXX See TODO below: what to return -- list of arrays or list of tuples?
        """
        mask = self.mask
        maskshape = mask.shape
        # TODO Check dimensionality of inId
        for neighbor in self.metric.getNeighbor(inId, radius):
            tneighbor = tuple(neighbor)
            if ( isInVolume(neighbor, maskshape) and
                 self.mask[tneighbor] != 0 ):
                yield neighbor


    def getNeighbor(self, outId, radius=0):
        """Return the list of Ids for the neighbors.

        Returns a list of outIds
        """
        # TODO Check dimensionality of outId
        inId = self.getInId(outId)
        for inId in self.getNeighborIn(inId, radius):
            yield self.getOutId(inId)



def isInVolume(coord, shape):
    """For given coord check if it is within a specified volume size.

    Returns True/False. Assumes that volume coordinates start at 0.
    No more generalization (arbitrary minimal coord) is done to save
    on performance

    XXX: should move somewhere else.
    """
    for i in xrange(len(coord)):
        if coord[i] < 0 or coord[i] >= shape[i]:
            return False
    return True
