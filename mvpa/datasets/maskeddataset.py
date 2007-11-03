#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: PyMVPA:"""

import operator

import numpy as N

from mvpa.datasets.mappeddataset import MappedDataset
from mvpa.datasets.maskmapper import MaskMapper
from mvpa.datasets.dataset import Dataset


# XXX should be really provided by any Mapper + MappedDataset tandem
# Michael: I agree, but what about keeping mapper and dataset synchronized
#          when a subset of features is selected? Some part of the code has to
#          take care and I currently cannot see a solution that would scale
#          to any kind of mapper. Mapper interface would need
#          improve/enhacements...
class MaskedDataset(MappedDataset):
    """
    """
    def __init__( self, samples, labels, chunks, mask=None, dtype=None ):
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
            # check for compatibility
            if not samples.shape[1:] == mask.shape:
                raise ValueError, "The mask dataspace shape [%s] is not " \
                                  "compatible with the shape of the provided " \
                                  "data samples [%s]." % (`mask.shape`,
                                                          `samples.shape[1:]`)
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
                                labels,
                                chunks,
                                mapper,
                                dtype )


    def __add__( self, other ):
        """
        When adding two MaskedDatasets the mask of the dataset left of the
        operator is used for the merged dataset.
        """
        out = MaskedDataset( self.samples,
                             self.labels,
                             self.chunks,
                             self.mapper )

        out += other

        return out


    def selectFeatures(self, ids, plain=False):
        """ Select a number of features from the current set.

        @ids is a list of feature IDs
        @plain=True directs to return a simple Dataset
        if @plain=False -- returns a new MaskedDataset object

        Return object is a view of the original data (no copying is
        performed).
        """

        if plain:
            return Dataset.selectFeatures(self, ids)

        # HEY, ATTENTION: the list of selected features needs to be sorted
        # otherwise the feature mask will become inconsistent with the
        # dataset and you need 2 days to discover the bug
        ids = sorted( ids )

        # create feature mask of the new subset to create a new mapper out
        # of it
        mask =  self.mapper.buildMaskFromFeatureIds( ids )

        return MaskedDataset( self.samples[:, ids],
                              self.labels,
                              self.chunks,
                              mask = MaskMapper( mask ) )


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
        comb_mask = N.logical_and( mask != 0, self.mapper.getMask(copy=False) != 0 )

        # transform mask into feature space
        fmask = self.mapper.forward( comb_mask != 0 )

        if plain:
            return Dataset( self.samples[:, fmask],
                            self.labels,
                            self.chunks)
        else:
            return MaskedDataset( self.samples[:, fmask],
                                  self.labels,
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

        return MaskedDataset( self.samples[mask, ],
                              self.labels[mask, ],
                              self.chunks[mask, ],
                              self.mapper )


