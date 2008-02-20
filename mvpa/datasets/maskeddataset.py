#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset with applied mask"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.datasets.mappeddataset import MappedDataset
from mvpa.datasets.maskmapper import MaskMapper

if __debug__:
    from mvpa.misc import debug

class MaskedDataset(MappedDataset):
    """Helper class which is `MappedDataset` with using `MaskMapper`.

    TODO: since what it does is simply some checkes/data_mangling in the
    constructor, it might be absorbed inside generic `MappedDataset`

    """

    def __init__(self, samples=None, mask=None, **kwargs):
        """Initialize `MaskedDataset` instance

        :Parameters:
          - `mask`: an ndarray where the chosen features equal the non-zero
            mask elements.

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
            # expand mask to span all dimensions but first one
            # necessary e.g. if only one slice from timeseries of volumes is
            # requested.
            mask = N.array(mask, ndmin=len(samples.shape[1:]))
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


    def selectFeaturesByMask(self, mask, plain=False):
        """Use a mask array to select features from the current set.

        :Parameters:
          mask : ndarray
            input mask
          plain : bool
            `True` directs to return a simple `Dataset`,
            `False` -- a new `MaskedDataset` object

        Returns a new MaskedDataset object with a view of the original pattern
        array (no copying is performed).
        The final selection mask only contains features that are present in the
        current feature mask AND the selection mask passed to this method.
        """
        # AND new and old mask to get the common features
        comb_mask = N.logical_and(mask != 0,
                                  self.mapper.getMask(copy=False) != 0)
        if __debug__:
            debug('DS', "VERY SUBOPTIMAL - do not rely on performance")
        # transform mask into feature space
        fmask = self.mapper.forward( comb_mask != 0 )
        #TODO all this will be gone soon anyway -- need proper selectIn within
        # a mapper
        return self.selectFeatures(fmask.nonzero()[0], plain=plain)


