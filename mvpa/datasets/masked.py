# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset with applied mask"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.datasets.mapped import MappedDataset
from mvpa.mappers.array import DenseArrayMapper

if __debug__:
    from mvpa.base import debug

class MaskedDataset(MappedDataset):
    """Helper class which is `MappedDataset` with using `MaskMapper`.

    TODO: since what it does is simply some checkes/data_mangling in the
    constructor, it might be absorbed inside generic `MappedDataset`

    """

    def __init__(self, samples=None, mask=None, **kwargs):
        """
        :Parameters:
          mask: ndarray
            the chosen features equal the non-zero mask elements.
        """
        # might contain the default mapper
        mapper = None

        # need if clause here as N.array(None) != None
        if not samples is None:
            # XXX should be asanyarray? but then smth segfaults on unittests
            samples = N.asarray(samples)
            mapper = DenseArrayMapper(mask=mask,
                                      shape=samples.shape[1:])

        if not mapper is None:
            if samples is None:
                raise ValueError, \
                      "Constructor of MaskedDataset requires both a samples " \
                      "array and a mask if one of both is provided."
            # init base class -- MappedDataset takes care of all the forward
            # mapping stuff
            MappedDataset.__init__(
                self,
                samples=samples,
                mapper=mapper,
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


