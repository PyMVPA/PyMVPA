# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset that gets its samples from a NIfTI file"""

__docformat__ = 'restructuredtext'

from mvpa.base import externals

import sys
import numpy as N
from mvpa.support.copy import deepcopy

if __debug__:
    from mvpa.base import debug

if externals.exists('nifti', raiseException=True):
    if sys.version_info[:2] >= (2, 5):
        # enforce absolute import
        NiftiImage = __import__('nifti', globals(), locals(), [], 0).NiftiImage
    else:
        # little trick to be able to import 'nifti' package (which has same
        # name)
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


def getNiftiFromAnySource(src, ensure=False, enforce_dim=None):
    """Load/access NIfTI data from files or instances.

    :Parameters:
      src: str | NiftiImage
        Filename of a NIfTI image or a `NiftiImage` instance.
      ensure : bool
        If True, through ValueError exception if cannot be loaded.
      enforce_dim : int or None
        If not None, it is the dimensionality of the data to be enforced,
        commonly 4D for the data, and 3D for the mask in case of fMRI.

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
    elif (isinstance(src, list) or isinstance(src, tuple)) \
        and len(src)>0 \
        and (isinstance(src[0], str) or isinstance(src[0], NiftiImage)):
        # load from a list of given entries
        if enforce_dim is not None: enforce_dim_ = enforce_dim - 1
        else:                       enforce_dim_ = None
        srcs = [getNiftiFromAnySource(s, ensure=ensure,
                                      enforce_dim=enforce_dim_)
                for s in src]
        if __debug__:
            # lets check if they all have the same dimensionality
            shapes = [s.data.shape for s in srcs]
            if not N.all([s == shapes[0] for s in shapes]):
                raise ValueError, \
                      "Input volumes contain variable number of dimensions:" \
                      " %s" % (shapes,)
        # Combine them all into a single beast
        nifti = NiftiImage(N.array([s.asarray() for s in srcs]),
                           srcs[0].header)
    elif ensure:
        raise ValueError, "Cannot load NIfTI from %s" % (src,)

    if nifti is not None and enforce_dim is not None:
        shape, new_shape = nifti.data.shape, None
        lshape = len(shape)

        # check if we need to tune up shape
        if lshape < enforce_dim:
            # if we are missing required dimension(s)
            new_shape = (1,)*(enforce_dim-lshape) + shape
        elif lshape > enforce_dim:
            # if there are bogus dimensions at the beginning
            bogus_dims = lshape - enforce_dim
            if shape[:bogus_dims] != (1,)*bogus_dims:
                raise ValueError, \
                      "Cannot enforce %dD on data with shape %s" \
                      % (enforce_dim, shape)
            new_shape = shape[bogus_dims:]

        # tune up shape if needed
        if new_shape is not None:
            if __debug__:
                debug('DS_NIFTI', 'Enforcing shape %s for %s data from %s' %
                      (new_shape, shape, src))
            nifti.data.shape = new_shape

    return nifti


def getNiftiData(nim):
    """Convenience function to extract the data array from a NiftiImage

    This function will make use of advanced features of PyNIfTI to prevent
    unnecessary copying if a sufficent version is available.
    """
    if externals.exists('nifti >= 0.20090205.1'):
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
    def __init__(self, samples=None, mask=None, dsattr=None,
                 enforce_dim=4, **kwargs):
        """
        :Parameters:
          samples: str | NiftiImage
            Filename of a NIfTI image or a `NiftiImage` instance.
          mask: str | NiftiImage | ndarray
            Filename of a NIfTI image or a `NiftiImage` instance or an ndarray
            of appropriate shape.
          enforce_dim : int or None
            If not None, it is the dimensionality of the data to be enforced,
            commonly 4D for the data, and 3D for the mask in case of fMRI.
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
        niftisamples = getNiftiFromAnySource(samples, ensure=True,
                                             enforce_dim=enforce_dim)
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
        if niftimask is None:
            pass
        elif isinstance(niftimask, N.ndarray):
            mask = niftimask
        else:
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


    def getDt(self):
        """Return the temporal distance of two samples/volumes.

        This method tries to be clever and always returns `dt` in seconds, by
        using unit information from the NIfTI header. If such information is
        not present the assumed unit will also be `seconds`.
        """
        # plain value
        hdr = self.niftihdr
        TR = hdr['pixdim'][4]

        # by default assume seconds as unit and do not scale
        scale = 1.0

        # figure out units, if available
        if hdr.has_key('time_unit'):
            unit_code = hdr['time_unit'] / 8
        elif hdr.has_key('xyzt_unit'):
            unit_code = int(hdr['xyzt_unit']) / 8
        else:
            warning("No information on time units is available. Assuming "
                    "seconds")
            unit_code = 0

        # handle known units
        # XXX should be refactored to use actual unit labels from pynifti
        # when version 0.20090205 or later is assumed to be available on all
        # machines
        if unit_code in [0, 1, 2, 3]:
            if unit_code == 0:
                warning("Time units were not specified in NiftiImage. "
                        "Assuming seconds.")
            scale = [ 1.0, 1.0, 1e-3, 1e-6 ][unit_code]
        else:
            warning("Time units are incorrectly coded: value %d whenever "
                    "allowed are 8 (sec), 16 (millisec), 24 (microsec). "
                    "Assuming seconds." % (unit_code * 8,)
                    )
        return TR * scale


    niftihdr = property(fget=lambda self: self._dsattr['niftihdr'],
                        doc='Access to the NIfTI header dictionary.')

    dt = property(fget=getDt,
                  doc='Time difference between two samples (in seconds). '
                  'AKA TR in fMRI world.')

    samplingrate = property(fget=lambda self: 1.0 / self.dt,
                          doc='Sampling rate (based on .dt).')


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
                 storeoffset=False, tr=None, enforce_dim=4, **kwargs):
        """
        :Paramaters:
          mask: str | NiftiImage | ndarray
            Filename of a NIfTI image or a `NiftiImage` instance or an ndarray
            of appropriate shape.
          evconv: bool
            Convert event definitions using `onset` and `duration` in some
            temporal unit into #sample notation.
          storeoffset: Bool
            Whether to store temproal offset information when converting
            Events into descrete time. Only considered when evconv == True.
          tr: float
            Temporal distance of two adjacent NIfTI volumes. This can be used
            to override the corresponding value in the NIfTI header.
          enforce_dim : int or None
            If not None, it is the dimensionality of the data to be enforced,
            commonly 4D for the data, and 3D for the mask in case of fMRI.
        """
        # check if we are in copy constructor mode
        if events is None:
            EventDataset.__init__(self, samples=samples, events=events,
                                  mask=mask, **kwargs)
            return

        nifti = getNiftiFromAnySource(samples, ensure=True,
                                      enforce_dim=enforce_dim)
        # no copying
        samples = nifti.data

        # do not put the whole NiftiImage in the dict as this will most
        # likely be deepcopy'ed at some point and ensuring data integrity
        # of the complex Python-C-Swig hybrid might be a tricky task.
        # Only storing the header dict should achieve the same and is more
        # memory efficient and even simpler
        dsattr = {'niftihdr': nifti.header}

        # determine TR, take from NIfTI header by default
        dt = nifti.rtime
        # override if necessary
        if not tr is None:
            dt = tr

        # NiftiDataset uses a DescreteMetric with cartesian
        # distance and element size from the NIfTI header
        # 'voxdim' is (x,y,z) while 'samples' are (t,z,y,x)
        elementsize = [dt] + [i for i in reversed(nifti.voxdim)]
        # XXX metric might be inappropriate if boxcar has length 1
        # might move metric setup after baseclass init and check what has
        # really happened
        metric = DescreteMetric(elementsize=elementsize,
                                distance_function=cartesianDistance)

        # convert EVs if necessary -- not altering original
        if evconv:
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
        if mask is None:
            pass
        elif isinstance(mask, N.ndarray):
            # plain array can be passed on to base class
            pass
        else:
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
