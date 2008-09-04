#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset that gets its samples from a NIfTI file"""

__docformat__ = 'restructuredtext'

# little trick to be able to import 'nifti' package (which has same name)
oldname = __name__
# crazy name with close to zero possibility to cause whatever
__name__ = 'iaugf9zrkjsbdv89'
from nifti import NiftiImage
# restore old settings
__name__ = oldname

from mvpa.datasets.base import Dataset
from mvpa.datasets.masked import MaskedDataset
from mvpa.datasets.metric import DescreteMetric, cartesianDistance
from mvpa.base import warning


class NiftiDataset(MaskedDataset):
    """Dataset based on NiftiImage provided by pynifti.

    See http://niftilib.sourceforge.net/pynifti/ for more information
    about pynifti.
    """
    # XXX: Every dataset should really have an example of howto instantiate
    #      it (necessary parameters).
    def __init__(self, samples=None, mask=None, dsattr=None, **kwargs):
        """Initialize NiftiDataset.

        :Parameters:
          - `samples`: Filename (string) of a NIfTI image or a `NiftiImage`
            object
          - `mask`: Filename (string) of a NIfTI image or a `NiftiImage`
            object

        """
        # if dsattr is none, set it to an empty dict
        if dsattr is None:
            dsattr = {}

        # we have to handle the nifti elementsize at the end if
        # mask is not already a Mapper
        set_elementsize = False
        if not dsattr.has_key('mapper'):
            set_elementsize = True

        # default way to use the constructor: with NIfTI image filename
        if not samples is None:
            if isinstance(samples, str):
                # open the nifti file
                try:
                    nifti = NiftiImage(samples)
                except RuntimeError, e:
                    warning("ERROR: NiftiDatasets: Cannot open samples file %s" \
                            % samples) # should we make also error?
                    raise e
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
                try:
                    mask = NiftiImage(mask).asarray()
                except RuntimeError, e:
                    warning("ERROR: NiftiDatasets: Cannot open mask file %s" \
                            % mask)
                    raise e

            elif isinstance(mask, NiftiImage):
                # just use data array as masl
                mask = mask.asarray()

        # by default init the dataset now
        # if mask is a Mapper already, this is a cheap init. This is
        # important as this is the default mode for the copy constructor
        # and might be called really often!!
        MaskedDataset.__init__(self,
                               samples=samples,
                               mask=mask,
                               dsattr=dsattr,
                               **(kwargs))


        if set_elementsize:
            # in case the Mapper wasn't already passed to the constructor
            # overwrite the default metric of it here to take the NIfTI element
            # properties into account

            # NiftiDataset uses a MetricMapper with DescreteMetric with cartesian
            # distance and element size from the NIfTI header 

            # 'voxdim' is (x,y,z) while 'samples' are (t,z,y,x)
            elementsize = [i for i in reversed(nifti.voxdim)]
            self.mapper.setMetric(
                        DescreteMetric(elementsize=elementsize,
                                       distance_function=cartesianDistance))


    def map2Nifti(self, data=None):
        """Maps a data vector into the dataspace and wraps it with a
        NiftiImage. The header data of this object is used to initialize
        the new NiftiImage.

        :Parameters:
          data : ndarray or Dataset
            The data to be wrapped into NiftiImage. If None (default), it
            would wrap samples of the current dataset. If it is a Dataset
            instance -- takes its samples for mapping
        """
        if data is None:
            data = self.samples
        elif isinstance(data, Dataset):
            # ease users life
            data = data.samples
        dsarray = self.mapper.reverse(data)
        return NiftiImage(dsarray, self.niftihdr)


    niftihdr = property(fget=lambda self: self._dsattr['niftihdr'],
                        doc='Access to the NIfTI header dictionary.')

