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
    def __init__(self, samples=None, mask=None, dsattr={}, **kwargs):
        """
        - `samples`: Filename of a NIfTI image or a `NiftiImage` object.
        """
        # we have to handle the nifti elementsize at the end if
        # mask is not already a MaskMapper
        set_elementsize = False
        if not dsattr.has_key('mapper'):
            set_elementsize = True

        # default way to use the constructor: with NIfTI image filename
        if not samples is None:
            if isinstance(samples, str):
                # open the nifti file
                nifti = NiftiImage(samples)
            elif isinstance(samples, NiftiImage):
                # nothing special
                nifti = samples
            else:
                raise ValueError, \
                      "NiftiDataset constructor takes the filename of a " \
                      "NIfTI image or a NiftiImage object as 'samples' " \
                      "argument."

            samples = nifti.data
            # do not put the whole NiftiImage in the dict as this will most
            # likely be deepcopy'ed at some point and ensuring data integrity
            # of the complex Python-C-Swig hybrid might be a tricky task.
            # Only storing the header dict should achieve the same and is more
            # memory efficient and even simpler
            dsattr['niftihdr'] = nifti.header

            if isinstance(mask, str):
                # if mask is also a nifti file open, it and take the image array
                # use a copy of the mask data as otherwise segfault will
                # embarass you, once the 'mask' NiftiImage get deleted
                mask = NiftiImage(mask).asarray()
            elif isinstance(mask, NiftiImage):
                # just use data array as masl
                mask = mask.asarray()

        # by default init the dataset now
        # if mask is a MaskMapper already, this is a cheap init. This is
        # important as this is the default mode for the copy constructor
        # and might be called really often!!
        MaskedDataset.__init__(self,
                               samples=samples,
                               mask=mask,
                               dsattr=dsattr,
                               **(kwargs))


        if set_elementsize:
            # in case the MaskMapper wasn't already passed to the constructor
            # overwrite the default metric of it here to take the NIfTI element
            # properties into account

            # NiftiDataset uses a MaskMapper with DescreteMetric with cartesian
            # distance and element size from the NIfTI header 

            # 'voxdim' is (x,y,z) while 'samples' are (t,z,y,x)
            elementsize = [i for i in reversed(nifti.voxdim)]
            self.mapper.setMetric(
                        DescreteMetric(elementsize=elementsize,
                                       distance_function=cartesianDistance))


    def map2Nifti(self, data):
        """Maps a data vector into the dataspace and wraps it with a
        NiftiImage. The header data of this object is used to initialize
        the new NiftiImage.
        """
        dsarray = self.mapper.reverse(data)
        return NiftiImage(dsarray, self.niftihdr)


    niftihdr = property(fget=lambda self: self._dsattr['niftihdr'],
                        doc='Access to the NIfTI header dictionary.')

