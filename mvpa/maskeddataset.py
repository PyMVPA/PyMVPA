### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA:
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2 of the License, or (at your option) any later version.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
# Modified by Yaroslav Halchenko 2007
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as N
from mappeddataset import *
from maskmapper import *


# XXX should be really provided by any Mapper + MappedDataset tandem
class MaskedDataset(MappedDataset):
    """
    """
    def __init__( self, samples, regs, chunks, mask=None ):
        """
        'mask' can be:
            * None: Mask using full dataspace is generated
            * ndarray: features equal non-zero mask elements
            * MaskMapper: existing mapper is used and 'samples' are assumed to
                          be already mapped by that mapper
        """
        samples = N.array( samples )

        if mask == None:
            # assume full dataspace mask
            mask = N.ones( samples.shape[1:], dtype='bool' )

        if isinstance( mask, N.ndarray ):
            # map samples with mask
            mapper = MaskMapper( mask )
            mapped_samples = mapper.forward( samples )
        elif isinstance( mask, MaskMapper ):
            # if mask is mapper assume the samples are already mapped
            # used to mimic a copy constructor
            mapper = mask
            mapped_samples = samples
        else:
            raise ValueError, "'mask' can only be None, ndarray or a " \
                              "MaskMapper instance."

        MappedDataset.__init__( self,
                                mapped_samples,
                                regs,
                                chunks,
                                mapper )


    def __add__( self, other ):
        """
        When adding two MaskedDatasets the mask of the dataset left of the
        operator is used for the merged dataset.
        """
        out = MaskedDataset( self.samples,
                             self.regs,
                             self.chunks,
                             self.mapper )

        out += other

        return out


    def selectFeatures( self, ids ):
        """ Select a number of features from the current set.

        'ids' is a list of feature IDs

        Returns a new MaskedDataset object with a view of the original data
        (no copying is performed).
        """
        # HEY, ATTENTION: the list of selected features needs to be sorted
        # otherwise the feature mask will become inconsistent with the
        # dataset and you need 2 days to discover the bug
        ids = sorted( ids )

        # create feature mask of the new subset to create a new mapper out
        # of it
        mask =  self.mapper.buildMaskFromFeatureIds( ids )

        return MaskedDataset( self.samples[:, ids],
                              self.regs,
                              self.chunks,
                              mask = MaskMapper( mask ) )


    def selectFeaturesByMask( self, mask ):
        """ Use a mask array to select features from the current set.

        The final selection mask only contains features that are present in the
        current feature mask AND the selection mask passed to this method.

        Returns a new MaskedDataset object with a view of the original pattern
        array (no copying is performed).
        """
        # AND new and old mask to get the common features
        comb_mask = N.logical_and( mask>0, self.mapper.getMask(copy=False)>0 )

        # transform mask into feature space
        fmask = self.mapper.forward( comb_mask > 0 )

        return MaskedDataset( self.samples[:, fmask],
                              self.regs,
                              self.chunks,
                              mask = MaskMapper( comb_mask ) )


    def selectSamples( self, mask ):
        """ Choose a subset of samples.

        Returns a new MaskedDataset object containing the selected sample
        subset.
        """
        # without having a sequence a index the masked sample array would
        # loose its 2d layout
        if not operator.isSequenceType( mask ):
            mask = [mask]

        return MaskedDataset( self.samples[mask,],
                              self.regs[mask,],
                              self.chunks[mask,],
                              self.mapper )


