# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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
from mvpa.mappers.metric import DescreteMetric, cartesianDistance
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

    def __init__(self, mask=None, metric=None,
                 distance_function=cartesianDistance,
                 elementsize=None, shape=None, **kwargs):
        """Initialize DenseArrayMapper

        :Parameters:
          mask : array
            an array in the original dataspace and its nonzero elements are
            used to define the features included in the dataset. alternatively,
            the `shape` argument can be used to define the array dimensions.
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
          shape: tuple
            The shape of the array to be mapped. If `shape` is provided instead
            of `mask`, a full mask (all True) of the desired shape is
            constructed. If `shape` is specified in addition to `mask`, the
            provided mask is extended to have the same number of dimensions.

        :Note: parameters `elementsize` and `distance_function` are relevant
               only if `metric` is None
        """
        if mask is None:
            if shape is None:
                raise ValueError, \
                      "Either `shape` or `mask` have to be specified."
            else:
                # make full dataspace mask if nothing else is provided
                mask = N.ones(shape, dtype='bool')
        else:
            if not shape is None:
                # expand mask to span all dimensions but first one
                # necessary e.g. if only one slice from timeseries of volumes is
                # requested.
                mask = N.array(mask, ndmin=len(shape))
                # check for compatibility
                if not shape == mask.shape:
                    raise ValueError, \
                        "The mask dataspace shape %s is not " \
                        "compatible with the provided shape %s." \
                        % (mask.shape, shape)

        # configure the baseclass with the processed mask
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
