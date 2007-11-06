#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset that gets its samples from a NIfTI file"""

from nifti import NiftiImage

from mvpa.datasets.dataset import Dataset
from mvpa.datasets.maskeddataset import MaskedDataset
from mvpa.datasets.metric import DescreteMetric, cartesianDistance
from mvpa.datasets.maskmapper import MaskMapper


class NiftiDataset(MaskedDataset):
    """
    """
    def __init__(self, filename, labels, chunks, mask=None, dtype=None):
        """
        """
        # we have to handle the nifti elementsize at the end if
        # mask is not already a MaskMapper
        set_elementsize = False
        if not isinstance(mask, MaskMapper):
            set_elementsize = True

        # default way to use the constructor: with NIfTI image filename
        if isinstance(filename, str):
            # open the nifti file
            self.__nifti = NiftiImage(filename)
            samples = self.__nifti.data

        # internal mode for copyconstructors: tuple(NiftiImage, samples_matrix)
        elif isinstance(filename, tuple):
            self.__nifti = filename[0]
            samples = filename[1]

        else:
            raise ValueError, "Unrecognized value in 'filename' argument."

        if isinstance(mask, str):
            # if mask is also a nifti file open, it and take the image array
            # use a copy of the mask data as otherwise segfault will embarass
            # you, once the 'mask' NiftiImage get deleted
            mask = NiftiImage(mask).asarray()

        # by default init the dataset now
        # if mask is a MaskMapper already, this is a cheap init. This is
        # important as this is the default mode for the copy constructor
        # and might be called really often!!
        MaskedDataset.__init__(self,
                               samples,
                               labels,
                               chunks,
                               mask,
                               dtype)


        if set_elementsize:
            # in case the MaskMapper wasn't already passed to the constructor
            # overwrite the default metric of it here to take the NIfTI element
            # properties into account

            # NiftiDataset uses a MaskMapper with DescreteMetric with cartesian
            # distance and element size from the NIfTI header 

            # 'voxdim' is (x,y,z) while 'samples' are (t,z,y,x)
            elementsize = [i for i in reversed(self.__nifti.voxdim)]
            self.mapper.setMetric(
                        DescreteMetric(elementsize=elementsize,
                                       distance_function=cartesianDistance))


    @staticmethod
    def _fromMaskedDataset(md, nifti):
        """Init a NiftiDataset from a MaskedDataset.

        This is an internal utility function -- not meant to be used by
        outsiders!

        It merges a separate NiftiImage object with a MaskedDataset by calling
        the NiftiDataset constructor with the right arguments. No checks are
        perform -- use with care!
        """
        return NiftiDataset((nifti, md.samples),
                            md.labels,
                            md.chunks,
                            md.mapper)


    def __add__(self, other):
        """Adds to NiftiDatasets.

        When adding the mask and NIfTI header information of the dataset left
        of the operator are used for the merged dataset.
        """
        merged = MaskedDataset.__add__(self, other)
        return NiftiDataset._fromMaskedDataset(merged, self.__nifti)


    def selectFeatures(self, ids, plain=False):
        """ Select a number of features from the current set.

        @ids is a list of feature IDs
        @plain=True directs to return a simple Dataset
        if @plain=False -- returns a new NiftiDataset object

        Return object is a view of the original data
        (no copying is performed).
        """
        if plain:
            return Dataset.selectFeatures(self, ids)

        sub = MaskedDataset.selectFeatures(self, ids)
        return NiftiDataset._fromMaskedDataset(sub, self.__nifti)


    def selectFeaturesByMask(self, mask, plain=False):
        """ Use a mask array to select features from the current set.

        The final selection mask only contains features that are present in the
        current feature mask AND the selection mask passed to this method.

        @ids is a list of feature IDs
        @plain=True directs to return a simple Dataset
        if @plain=False -- returns a new NiftiDataset object

        Return object is a view of the original data (no copying is
        performed).
        """
        if plain:
            raise NotImplementedError #return Dataset.selectFeatures(self, ids)

        sub = MaskedDataset.selectFeaturesByMask(self, mask)
        return NiftiDataset._fromMaskedDataset(sub, self.__nifti)


    def selectSamples( self, mask ):
        """ Choose a subset of samples.

        Returns a new NiftiDataset object containing the selected sample
        subset.
        """
        sub = MaskedDataset.selectSamples(self, mask)
        return NiftiDataset._fromMaskedDataset(sub, self.__nifti)


    def map2Nifti(self, data):
        """Maps a data vector into the dataspace and wraps it with a
        NiftiImage. The header data of this object is used to initialize
        the new NiftiImage.
        """
        dsarray = self.mapper.reverse(data)
        return NiftiImage(dsarray, self.niftihdr)


    niftihdr = property(fget=lambda self: self.__nifti.header,
                        doc='Access to the NIfTI header dictionary.')

