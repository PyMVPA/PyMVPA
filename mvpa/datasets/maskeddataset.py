#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset with applied mask"""

import operator

import numpy as N

from mvpa.datasets.mappeddataset import MappedDataset
from mvpa.datasets.maskmapper import MaskMapper
from mvpa.datasets.dataset import Dataset


class MaskedDataset(MappedDataset):
    """
    """
    def __init__(self, samples=None, mask=None, **kwargs):
        """
        `mask` is usually an ndarray where to chosen features equal the
        non-zero mask elements. A special value for `mask` is recognized: If
        `mask` is set to 'full' a mask covering the whole dataspace is created.
        """
        # need if clause here as N.array(None) != None
        if not samples is None:
            samples = N.array(samples)
            if mask is None:
                # make full dataspace mask if nothing else is provided
                mask = N.ones(samples.shape[1:], dtype='bool')
        if not mask is None:
            if samples is None:
                raise ValueError, \
                      "Constructor of MaskedDataset requires both a samples " \
                      "array and a mask if one of both is provided."
             # check for compatibility
            if not samples.shape[1:] == mask.shape:
                raise ValueError, "The mask dataspace shape [%s] is not " \
                                  "compatible with the shape of the provided " \
                                  "data samples [%s]." % (`mask.shape`,
                                                          `samples.shape[1:]`)
            # init base class -- MappedDataset takes care of all the forward
            # mapping stuff
            MappedDataset.__init__(self,
                                   samples=samples,
                                   mapper=MaskMapper(mask),
                                   **(kwargs))

        else:
            MappedDataset.__init__(self, **(kwargs))



#    def selectFeatures(self, ids, plain=False):
#        """ Select a number of features from the current set.
#
#        `ids` is a list of feature IDs
#        `plain`=`True` directs to return a simple Dataset
#        if `plain`==False -- returns a new MaskedDataset object
#
#        Return object is a view of the original data (no copying is
#        performed).
#        """
#
#        if plain:
#            return Dataset.selectFeatures(self, ids)
#
#        # HEY, ATTENTION: the list of selected features needs to be sorted
#        # otherwise the feature mask will become inconsistent with the
#        # dataset and you need 2 days to discover the bug
#        ids = sorted( ids )
#
#        # create feature mask of the new subset to create a new mapper out
#        # of it
#        mask =  self.mapper.buildMaskFromFeatureIds( ids )
#
#        return MaskedDataset( self.samples[:, ids],
#                              self.labels,
#                              self.chunks,
#                              mask = MaskMapper( mask ) )
#

    def selectFeaturesByMask(self, mask, plain=False):
        """ Use a mask array to select features from the current set.

        The final selection mask only contains features that are present in the
        current feature mask AND the selection mask passed to this method.

        @mask       selects features
        @plain=True directs to return a simple Dataset
        if @plain=False -- returns a new MaskedDataset object

        if @plain=True return
        Returns a new MaskedDataset object with a view of the original pattern
        array (no copying is performed).
        """
        # AND new and old mask to get the common features
        comb_mask = N.logical_and(mask != 0,
                                  self.mapper.getMask(copy=False) != 0)

        # transform mask into feature space
        fmask = self.mapper.forward( comb_mask != 0 )

        return self.selectFeatures(fmask)
