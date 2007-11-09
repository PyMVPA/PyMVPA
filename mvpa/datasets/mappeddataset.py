#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Mapped dataset"""


from mvpa.datasets.dataset import Dataset


class MappedDataset(Dataset):
    """
    """
    def __init__(self, unmapped_samples=None, mapper=None, dsattr={}, **kwargs):
        """A `Dataset` that uses a mapper to transform samples from their
        original dataspace into the feature space.kwargs are passed to
        `Dataset`.
        """
        # there are basically two mode for the constructor:
        # 1. internal mode - only data and dsattr dict
        # 2. user mode - unmapped_samples and mapper != None

        # if a mapper was passed, store it in dsattr dict that gets passed
        # to base Dataset
        if not mapper is None:
            dsattr['mapper'] = mapper

        # if the samples are passed to the special arg, use the mapper to
        # transform them.
        if not unmapped_samples is None:
            if dsattr['mapper'] is None:
                raise ValueError,
                      "Constructor of MappedDataset requires a mapper " \
                      "if unmapped samples are provided."
            Dataset.__init__(self,
                             samples=mapper.forward(unmapped_samples),
                             dsattr=dsattr,
                             **(kwargs))
        else:
            Dataset.__init__(self, dsattr=dsattr, **(kwargs))


    def mapForward(self, data):
        """ Map data from the original dataspace into featurespace.
        """
        return self.mapper.forward(data)


    def mapReverse(self, data):
        """ Reverse map data from featurespace into the original dataspace.
        """
        return self.mapper.reverse(data)


    def selectFeatures(self, ids):
        """
        """
        # has to be reimplemented because the mapper has to be adjusted when
        # the features space is modified

        # call base method to get selected feature subset
        sdata = Dataset.selectFeatures(self, ids)

        raise NotImplementedError

        sdata._dsattr['mapper'] = self._dsattr['mapper'].WHAT_IS_YOUR_NAME?

        return sdata


    # read-only class properties
    mapper = property(fget=lambda self: self._data['mapper'])
