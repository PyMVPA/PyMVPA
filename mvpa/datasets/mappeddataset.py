#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Mapped dataset"""

__docformat__ = 'restructuredtext'

import copy

from mvpa.datasets.dataset import Dataset


class MappedDataset(Dataset):
    """A `Dataset` which is created by applying a `Mapper` to the data.

    It uses a mapper to transform samples from their original
    dataspace into the feature space. Various mappers can be used. The
    "easiest" one is `MaskMapper` which allows to select the features
    (voxels) to be used in the analysis: see `MaskedDataset`
    """

    def __init__(self, samples=None, mapper=None, dsattr=None, **kwargs):
        """Initialize `MaskedDataset`

        :Parameters:
          - `mapper`: Instance of `Mapper` used to map input data

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
            if dsattr['mapper'] is None:
                raise ValueError, \
                      "Constructor of MappedDataset requires a mapper " \
                      "if unmapped samples are provided."
            Dataset.__init__(self,
                             samples=mapper.forward(samples),
                             dsattr=dsattr,
                             **(kwargs))
        else:
            Dataset.__init__(self, dsattr=dsattr, **(kwargs))



    def mapForward(self, data):
        """Map data from the original dataspace into featurespace.
        """
        return self.mapper.forward(data)


    def mapReverse(self, data):
        """Reverse map data from featurespace into the original dataspace.
        """
        return self.mapper.reverse(data)


    def selectFeatures(self, ids, plain=False, sort=False):
        """Select features given their ids.

        :Parameters:
          - `ids`: iterable container to select ids
          - `plain`: `bool`, if to return MappedDataset (or just Dataset)
          - `sort`: `bool`, if to sort Ids. Order matters and selectFeatures
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
            sdata._dsattr['mapper'].selectOut(ids, sort)
            return sdata


    # read-only class properties
    mapper = property(fget=lambda self: self._dsattr['mapper'])
