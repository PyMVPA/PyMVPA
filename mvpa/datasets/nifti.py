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

import numpy as N
from mvpa.misc.copy import deepcopy

# little trick to be able to import 'nifti' package (which has same name)
oldname = __name__
# crazy name with close to zero possibility to cause whatever
__name__ = 'iaugf9zrkjsbdv89'
from nifti import NiftiImage
# restore old settings
__name__ = oldname

from mvpa.datasets.base import Dataset
from mvpa.datasets.mapped import MappedDataset
from mvpa.datasets.event import EventDataset
from mvpa.mappers.base import CombinedMapper
from mvpa.mappers.metric import DescreteMetric, cartesianDistance
from mvpa.mappers.array import DenseArrayMapper
from mvpa.base import warning


def getNiftiFromAnySource(src):
    """Load/access NIfTI data from files or instances.

    :Parameter:
      src: str | NiftiImage
        Filename of a NIfTI image or a `NiftiImage` instance.

    :Returns:
      NiftiImage | None
        If the source is not supported None is returned.
    """
    nifti = None

    # figure out what type
    if isinstance(src, str):
        # open the nifti file
        try:
            nifti = NiftiImage(src)
        except RuntimeError, e:
            warning("ERROR: NiftiDatasets: Cannot open NIfTI file %s" \
                    % src)
            raise e
    elif isinstance(src, NiftiImage):
        # nothing special
        nifti = src

    return nifti


def getNiftiData(nim):
    """Convenience function to extract the data array from a NiftiImage

    This function will make use of advanced features of PyNIfTI to prevent
    unnecessary copying if a sufficent version is available.
    """
    if externals.exists('nifti >= 0.20081017.1'):
        return nim.data
    else:
        return nim.asarray()


class NiftiDataset(MappedDataset):
    """Dataset loading its samples from a NIfTI image or file.

    Samples can be loaded from a NiftiImage instance or directly from a NIfTI
    file. This class stores all relevant information from the NIfTI file header
    and provides information about the metrics and neighborhood information of
    all voxels.

    Most importantly it allows to map data back into the original data space
    and format via :meth:`~mvpa.datasets.nifti.NiftiDataset.map2Nifti`.

    This class allows for convenient pre-selection of features by providing a
    mask to the constructor. Only non-zero elements from this mask will be
    considered as features.

    NIfTI files are accessed via PyNIfTI. See
    http://niftilib.sourceforge.net/pynifti/ for more information about
    pynifti.
    """
    # XXX: Every dataset should really have an example of howto instantiate
    #      it (necessary parameters).
    def __init__(self, samples=None, mask=None, dsattr=None, **kwargs):
        """
        :Parameters:
          samples: str | NiftiImage
            Filename of a NIfTI image or a `NiftiImage` instance.
          mask: str | NiftiImage
            Filename of a NIfTI image or a `NiftiImage` instance.
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

        # load the samples
        niftisamples = getNiftiFromAnySource(samples)
        samples = niftisamples.data

        # do not put the whole NiftiImage in the dict as this will most
        # likely be deepcopy'ed at some point and ensuring data integrity
        # of the complex Python-C-Swig hybrid might be a tricky task.
        # Only storing the header dict should achieve the same and is more
        # memory efficient and even simpler
        dsattr = {'niftihdr': niftisamples.header}


        # figure out what the mask is, but onyl handle known cases, the rest
        # goes directly into the mapper which maybe knows more
        niftimask = getNiftiFromAnySource(mask)
        if not niftimask is None:
            mask = getNiftiData(niftimask)

        # build an appropriate mapper that knows about the metrics of the NIfTI
        # data
        # NiftiDataset uses a DescreteMetric with cartesian
        # distance and element size from the NIfTI header 

        # 'voxdim' is (x,y,z) while 'samples' are (t,z,y,x)
        elementsize = [i for i in reversed(niftisamples.voxdim)]
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



class ERNiftiDataset(EventDataset):
    """Dataset with event-defined samples from a NIfTI timeseries image.

    This is a convenience dataset to facilitate the analysis of event-related
    fMRI datasets. Boxcar-shaped samples are automatically extracted from the
    full timeseries using :class:`~mvpa.misc.support.Event` definition lists.
    For each event all volumes covering that particular event in time
    (including partial coverage) are used to form the corresponding sample.

    The class supports the conversion of events defined in 'realtime' into the
    descrete temporal space defined by the NIfTI image. Moreover, potentially
    varying offsets between true event onset and timepoint of the first selected
    volume can be stored as an additional feature in the dataset.

    Additionally, the dataset supports masking. This is done similar to the
    masking capabilities of :class:`~mvpa.datasets.nifti.NiftiDataset`. However,
    the mask can either be of the same shape as a single NIfTI volume, or
    can be of the same shape as the generated boxcar samples, i.e.
    a samples consisting of three volumes with 24 slices and 64x64 inplane
    resolution needs a mask with shape (3, 24, 64, 64). In the former case the
    mask volume is automatically expanded to be identical in a volumes of the
    boxcar.
    """
    def __init__(self, samples=None, events=None, mask=None, evconv=False,
                 storeoffset=False, tr=None, **kwargs):
        """
        :Paramaters:
          evconv: bool
            Convert event definitions using `onset` and `duration` in some
            temporal unit into #sample notation.
          storeoffset: Bool
            Whether to store temproal offset information when converting
            Events into descrete time. Only considered when evconv == True.
          tr: float
            Temporal distance of two adjacent NIfTI volumes. This can be used
            to override the corresponding value in the NIfTI header.
        """
        # check if we are in copy constructor mode
        if events is None:
            EventDataset.__init__(self, samples=samples, events=events,
                                  mask=mask, **kwargs)
            return

        nifti = getNiftiFromAnySource(samples)
        # no copying
        samples = nifti.data

        # do not put the whole NiftiImage in the dict as this will most
        # likely be deepcopy'ed at some point and ensuring data integrity
        # of the complex Python-C-Swig hybrid might be a tricky task.
        # Only storing the header dict should achieve the same and is more
        # memory efficient and even simpler
        dsattr = {'niftihdr': nifti.header}

        # NiftiDataset uses a DescreteMetric with cartesian
        # distance and element size from the NIfTI header 
        # 'voxdim' is (x,y,z) while 'samples' are (t,z,y,x)
        elementsize = [i for i in reversed(nifti.voxdim)]
        # XXX metric might be inappropriate if boxcar has length 1
        # might move metric setup after baseclass init and check what has
        # really happened
        metric = DescreteMetric(elementsize=elementsize,
                                distance_function=cartesianDistance)

        # convert EVs if necessary -- not altering original
        if evconv:
            # determine TR, take from NIfTI header by default
            dt = nifti.rtime
            # override if necessary
            if not tr is None:
                dt = tr
            if dt == 0:
                raise ValueError, "'dt' cannot be zero when converting Events"

            events = [ev.asDescreteTime(dt, storeoffset) for ev in events]
        else:
            # do not touch the original
            events = deepcopy(events)

            # forcefully convert onset and duration into integers, as expected
            # by the baseclass
            for ev in events:
                oldonset = ev['onset']
                oldduration = ev['duration']
                ev['onset'] = int(ev['onset'])
                ev['duration'] = int(ev['duration'])
                if not oldonset == ev['onset'] \
                   or not oldduration == ev['duration']:
                    warning("Loosing information during automatic integer "
                            "conversion of EVs. Consider an explicit conversion"
                            " by setting `evconv` in ERNiftiDataset().")

        # pull mask array from NIfTI (if present)
        if not mask is None:
            mask_nim = getNiftiFromAnySource(mask)
            if not mask_nim is None:
                mask = getNiftiData(mask_nim)
            else:
                raise ValueError, "Cannot load mask from '%s'" % mask

        # finally init baseclass
        EventDataset.__init__(self, samples=samples, events=events,
                              mask=mask, dametric=metric, dsattr=dsattr,
                              **kwargs)


    def map2Nifti(self, data=None):
        """Maps a data vector into the dataspace and wraps it with a
        NiftiImage. The header data of this object is used to initialize
        the new NiftiImage.

        .. note::
          Only the features corresponding to voxels are mapped back -- not
          any additional features passed via the Event definitions.

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

        mr = self.mapper.reverse(data)

        # trying to determine which part should go into NiftiImage
        if isinstance(self.mapper, CombinedMapper):
            # we have additional feature in the dataset -- ignore them
            mr = mr[0]
        else:
            pass

        return NiftiImage(mr, self.niftihdr)


    niftihdr = property(fget=lambda self: self._dsattr['niftihdr'],
                        doc='Access to the NIfTI header dictionary.')
