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

from mvpa.base import externals
externals.exists('nifti', raiseException=True)

# little trick to be able to import 'nifti' package (which has same name)
oldname = __name__
# crazy name with close to zero possibility to cause whatever
__name__ = 'iaugf9zrkjsbdv89'
from nifti import NiftiImage
# restore old settings
__name__ = oldname

from mvpa.datasets.base import Dataset
from mvpa.datasets.mapped import MappedDataset
from mvpa.mappers.metric import DescreteMetric, cartesianDistance
from mvpa.mappers.array import DenseArrayMapper
from mvpa.base import warning


class NiftiDataset(MappedDataset):
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
        # if in copy constructor mode
        if not dsattr is None and dsattr.has_key('mapper'):
            MappedDataset.__init__(self,
                                   samples=samples,
                                   dsattr=dsattr,
                                   **kwargs)
            return

        #
        # the following code only deals with contructing fresh datasets from
        # scratch
        #

        # figure out what type the samples are
        if isinstance(samples, str):
            # open the nifti file
            try:
                nifti = NiftiImage(samples)
            except RuntimeError, e:
                warning("ERROR: NiftiDatasets: Cannot open samples file %s" \
                        % samples)
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
        dsattr = {'niftihdr': nifti.header}

        # figure out what the mask is, but onyl handle known cases, the rest
        # goes directly into the mapper which maybe knows more
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
            # just use data array as mask
            mask = mask.asarray()

        # build an appropriate mapper that knows about the metrics of the NIfTI
        # data
        # NiftiDataset uses a DescreteMetric with cartesian
        # distance and element size from the NIfTI header 

        # 'voxdim' is (x,y,z) while 'samples' are (t,z,y,x)
        elementsize = [i for i in reversed(nifti.voxdim)]
        mapper = DenseArrayMapper(mask=mask, shape=samples.shape[1:],
                    metric=DescreteMetric(elementsize=elementsize,
                                          distance_function=cartesianDistance))

        MappedDataset.__init__(self,
                               samples=samples,
                               mapper=mapper,
                               dsattr=dsattr,
                               **kwargs)


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

