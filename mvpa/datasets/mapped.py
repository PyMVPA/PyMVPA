# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Mapped dataset"""

__docformat__ = 'restructuredtext'

import mvpa.support.copy as copy

from mvpa.datasets import Dataset
from mvpa.base.dochelpers import enhancedDocString
from mvpa.misc.exceptions import DatasetError

class MappedDataset(Dataset):
    """A `Dataset` which is created by applying a `Mapper` to the data.

    Upon contruction `MappedDataset` uses a `Mapper` to transform the
    samples from their original into the two-dimensional matrix
    representation that is required by the `Dataset` class.

    This class enhanced the `Dataset` interface with two additional
    methods: `mapForward()` and `mapReverse()`. Both take arbitrary data
    arrays (with matching shape) and transform them using the embedded
    mapper from the original dataspace into a one- or two-dimensional
    representation (for arrays corresponding to the shape of a single or
    multiple samples respectively) or vice versa.

    Most likely, this class will not be used directly, but rather
    indirectly through one of its subclasses (e.g. `MaskedDataset`).
    """

    def __init__(self, samples=None, mapper=None, dsattr=None, **kwargs):
        """
        If `samples` and `mapper` arguments are not `None` the mapper is
        used to forward-map the samples array and the result is passed
        to the `Dataset` constructor.

        :Parameters:
          mapper: Instance of `Mapper`
            This mapper will be embedded in the dataset and is used and
            updated, by all subsequent mapping or feature selection
            procedures.
          **kwargs:
            All other arguments are simply passed to and handled by
            the constructor of `Dataset`.
        """
        # there are basically two mode for the constructor:
        # 1. internal mode - only data and dsattr dict
        # 2. user mode - samples != None # and mapper != None

        # see if dsattr is none, if so, set to empty dict
        if dsattr is None:
            dsattr = {}

        # if a mapper was passed, store it in dsattr dict that gets passed
        # to base Dataset
        if not mapper is None:
            # TODO: check mapper for compliance with dimensionality within _data
            #       may be only within __debug__
            dsattr['mapper'] = mapper

        # if the samples are passed to the special arg, use the mapper to
        # transform them.
        if not samples is None:
            if not dsattr.has_key('mapper') or dsattr['mapper'] is None:
                raise DatasetError, \
                      "Constructor of MappedDataset requires a mapper " \
                      "if unmapped samples are provided."
            Dataset.__init__(self,
                             samples=mapper.forward(samples),
                             dsattr=dsattr,
                             **(kwargs))
        else:
            Dataset._checkCopyConstructorArgs(samples=samples,
                                              dsattr=dsattr,
                                              **kwargs)
            Dataset.__init__(self, dsattr=dsattr, **(kwargs))


    __doc__ = enhancedDocString('MappedDataset', locals(), Dataset)


    def mapForward(self, data):
        """Map data from the original dataspace into featurespace.
        """
        return self.mapper.forward(data)


    def mapReverse(self, data):
        """Reverse map data from featurespace into the original dataspace.
        """
        return self.mapper.reverse(data)


    def mapSelfReverse(self):
        """Reverse samples from featurespace into the original dataspace.
        """
        return self.mapper.reverse(self.samples)

    def selectFeatures(self, ids, plain=False, sort=False):
        """Select features given their ids.

        The methods behaves similar to Dataset.selectFeatures(), but
        additionally takes care of adjusting the embedded mapper
        appropriately.

        :Parameters:
          ids: sequence
            Iterable container to select ids
          plain: boolean
            Flag whether to return MappedDataset (or just Dataset)
          sort: boolean
            Flag whether to sort Ids. Order matters and selectFeatures
            assumes incremental order. If not such, in non-optimized
            code selectFeatures would verify the order and sort
        """

        # call base method to get selected feature subset
        if plain:
            sdata = Dataset(self._data, self._dsattr, check_data=False,
                            copy_samples=False, copy_data=False,
                            copy_dsattr=False)
            return sdata.selectFeatures(ids, sort)
        else:
            sdata = Dataset.selectFeatures(self, ids)
            # since we have new DataSet we better have a new mapper
            sdata._dsattr['mapper'] = copy.deepcopy(sdata._dsattr['mapper'])
            if sort:
                sdata._dsattr['mapper'].selectOut(sorted(ids))
            else:
                sdata._dsattr['mapper'].selectOut(ids)
            return sdata


    # read-only class properties
    mapper = property(fget=lambda self: self._dsattr['mapper'])
    samples_original = property(fget=mapSelfReverse,
                                doc="Return samples in the original shape")

    # syntactic sugarings
    O = property(fget=mapSelfReverse,
                 doc="Return samples in the original shape")
