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
    def __init__(self, samples=None, mapper=None, dsattr={}, **kwargs):
        """A `Dataset` that uses a mapper to transform samples from their
        original dataspace into the feature space.kwargs are passed to
        `Dataset`.
        """
        # there are basically two mode for the constructor:
        # 1. internal mode - only data and dsattr dict
        # 2. user mode - samples != None # and mapper != None

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
        """ Map data from the original dataspace into featurespace.
        """
        return self.mapper.forward(data)


    def mapReverse(self, data):
        """ Reverse map data from featurespace into the original dataspace.
        """
        return self.mapper.reverse(data)


    def selectFeatures(self, ids, plain=False, bymask=False):
        """

        XXX: for now it sorts ids in numerical orders. This should be
        replaced with properly working mapper's selectFeatures which
        would take care about changed order of ids

        """
        # TODO :has to be reimplemented because the mapper has to be
        # adjusted when the features space is modified
        if bymask == False:
            ids = sorted(ids)


        # call base method to get selected feature subset
        if plain:
            sdata = Dataset(self._data, self._dsattr, check_data=False,
                            copy_samples=False, copy_data=False,
                            copy_dsattr=False)
            return sdata.selectFeatures(ids)
        else:
            sdata = Dataset.selectFeatures(self, ids)
            sdata._dsattr['mapper'].selectOut(ids)
            return sdata


    # read-only class properties
    mapper = property(fget=lambda self: self._dsattr['mapper'])
