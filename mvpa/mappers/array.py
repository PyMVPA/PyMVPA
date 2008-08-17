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

from mvpa.mappers.mask import MaskMapper
from mvpa.datasets.metric import DescreteMetric, cartesianDistance
from mvpa.base.dochelpers import enhancedDocString

if __debug__:
    from mvpa.base import warning
    from mvpa.misc.support import isSorted



class DenseArrayMapper(MaskMapper):
    """Mapper for equally spaced dense arrays."""

    """TODO: yoh thinks we should move that 'metric' assignment into
    MaskMapper, based on the fact if distance_function is given either
    as an argument or may be class variable. That would pretty much
    remove the need for a separate class of DenseArrayMapper and it
    could become just a sugaring helper function which would initiate
    MaskMapper (or some other mapper may be with appropriate
    distance_function and/or mapper

    Otherwise it is again -- orthogonality -- will we need to device
    NonmaskedArrayMapper which has no mask assigned but might be a
    good cartesian cube on its own or smth like that?
    """

    def __init__(self, mask, metric=None, distance_function=cartesianDistance,
                 elementsize=None, **kwargs):
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

        MaskMapper.__init__(self, mask, metric=metric, **kwargs)

        # We must have metric assigned
        if self.metric == None:
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
            self.metric = DescreteMetric(elementsize=[1]*len(mask.shape),
                                         distance_function=distance_function)


    __doc__ = enhancedDocString('DenseArrayMapper', locals(), MaskMapper)


    def __str__(self):
        return "DenseArrayMapper: %d -> %d" \
            % (self.getInSize(), self.getOutSize())


    # No need to overrride because all arguments just to assign a
    # metric, which would be visible from Mapper class
    #def __repr__(self):
    #    s = super(DenseArrayMapper, self).__str__()
    #    return s.sub("(", "?????", 1)


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


