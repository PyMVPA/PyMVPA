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
from mvpa.base.collections import DatasetAttribute
from mvpa.base.dataset import _expand_attribute

if __debug__:
    from mvpa.base import debug

if externals.exists('nifti', raiseException=True):
    from nifti import NiftiImage

from mvpa.datasets.base import Dataset
from mvpa.mappers.flatten import FlattenMapper
from mvpa.mappers.boxcar import BoxcarMapper
from mvpa.base import warning


def getNiftiFromAnySource(src, ensure=False, enforce_dim=None):
    """Load/access NIfTI data from files or instances.

    Parameters
    ----------
    src : str or NiftiImage
      Filename of a NIfTI image or a `NiftiImage` instance.
    ensure : bool, optional
      If True, throw ValueError exception if cannot be loaded.
    enforce_dim : int or None
      If not None, it is the dimensionality of the data to be enforced,
      commonly 4D for the data, and 3D for the mask in case of fMRI.

    Returns
    -------
    NiftiImage or None
      If the source is not supported None is returned.

    Raises
    ------
    ValueError
      If there is a problem with data (variable dimensionality) or
      failed to load data and ensure=True.
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
    if externals.exists('nifti ge 0.20090205.1'):
        return nim.data
    else:
        return nim.asarray()


class NiftiDataset(Dataset):
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
    def map2nifti(dataset, data=None):
        """Maps a data vector into the dataspace and wraps it with a
        NiftiImage. The header data of this object is used to initialize
        the new NiftiImage.

        Parameters
        ----------
        data : ndarray or Dataset, optional
          The data to be wrapped into NiftiImage. If None (default), it
          would wrap samples of the current dataset. If it is a Dataset
          instance -- takes its samples for mapping.
        """
        if data is None:
            data = dataset.samples
        elif isinstance(data, Dataset):
            # ease users life
            data = data.samples
        # call the appropriate function to map single samples or multiples
        if len(data.shape) > 1:
            dsarray = dataset.a.mapper.reverse(data)
        else:
            dsarray = dataset.a.mapper.reverse1(data)
        return NiftiImage(dsarray, dataset.a.imghdr)


def fmri_dataset(samples, labels=None, chunks=None, mask=None,
                 events=None, tr=None,
                 sprefix='voxel', tprefix='time', eprefix='event'):
    """Constructs a `Dataset` given 4D fMRI file


    ALSO


    Dataset with event-defined samples from a NIfTI timeseries image.

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

    Parameters
    ----------
    tr : float
      Temporal distance of two adjacent NIfTI volumes. This can be used
      to override the corresponding value in the NIfTI header.


    TODO: extend
    """
    # TODO: Create detrending mapper and allow a mapper to be applied before
    # boxcaring -- we can only resonably detrend before boxcaring...

    # load the samples
    niftisamples = getNiftiFromAnySource(samples, ensure=True, enforce_dim=4)
    samples = niftisamples.data

    # figure out what the mask is, but onyl handle known cases, the rest
    # goes directly into the mapper which maybe knows more
    niftimask = getNiftiFromAnySource(mask)
    if niftimask is None:
        pass
    elif isinstance(niftimask, N.ndarray):
        mask = niftimask
    else:
        mask = getNiftiData(niftimask)

    # compile the samples attributes
    sa = {}
    if not labels is None:
        sa['labels'] = _expand_attribute(labels, samples.shape[0], 'labels')
    if not chunks is None:
        sa['chunks'] = _expand_attribute(chunks, samples.shape[0], 'chunks')

    # create a dataset
    ds = NiftiDataset(samples, sa=sa)
    if sprefix is None:
        inspace = None
    else:
        inspace = sprefix + '_indices'
    ds = ds.get_mapped(FlattenMapper(shape=samples.shape[1:], inspace=inspace))

    # now apply the mask if any
    if not mask is None:
        flatmask = ds.a.mapper.forward1(mask != 0)
        # direct slicing is possible, and it is potentially more efficient,
        # so let's use it
        #mapper = FeatureSliceMapper(flatmask)
        #ds = ds.get_mapped(FeatureSliceMapper(flatmask))
        ds = ds[:, flatmask]

    # store interesting props in the dataset
    # do not put the whole NiftiImage in the dict as this will most
    # likely be deepcopy'ed at some point and ensuring data integrity
    # of the complex Python-C-Swig hybrid might be a tricky task.
    # Only storing the header dict should achieve the same and is more
    # memory efficient and even simpler
    ds.a['imghdr'] = niftisamples.header
    # If there is a space assigned , store the extent of that space
    if sprefix is not None:
        ds.a[sprefix + '_dim'] = samples.shape[1:]
        # 'voxdim' is (x,y,z) while 'samples' are (t,z,y,x)
        ds.a[sprefix + '_eldim'] = \
                tuple([i for i in reversed(niftisamples.voxdim)])
        # TODO extend with the unit
    if tprefix is not None:
        ds.sa[tprefix + '_indices'] = N.arange(len(ds), dtype='int')
        ds.sa[tprefix + '_coords'] = N.arange(len(ds), dtype='float') \
                                     * niftisamples.header['pixdim'][4]
        # TODO extend with the unit

    # exit here if there are no events specified
    if events is None:
        return ds

    #
    # Post-processing for event handling
    #
    # determine TR, take from NIfTI header by default
    dt = niftisamples.header['pixdim'][4]
    # override if necessary
    if not tr is None:
        dt = tr
    # convert all onsets into descrete integer values representing volume ids
    # but storing any possible offset to the real event onset as an additional
    # feature of that event -- these features will be stored as sample
    # attributes
    descr_events = [ev.asDescreteTime(dt, storeoffset=True) for ev in events]

    # convert the event specs into the format expected by BoxcarMapper
    # take the first event as an example of contained keys
    evvars = {}
    for k in descr_events[0]:
        try:
            evvars[k] = [e[k] for e in descr_events]
        except KeyError:
            raise ValueError("Each event property must be present for all "
                             "events (could not find '%s'" % k)
    # checks
    for p in ['onset', 'duration']:
        if not p in evvars:
            raise ValueError("'%s' is a required property for all events."
                             % p)
    boxlength = max(evvars['duration'])
    if __debug__:
        if not max(evvars['duration']) == min(evvars['duration']):
            warning('Boxcar mapper will use maximum boxlength (%i) of all '
                    'provided Events.'% boxlength)

    # finally create, train und use the boxcar mapper
    bcm = BoxcarMapper(evvars['onset'], boxlength, inspace=eprefix)
    bcm.train(ds)
    ds = ds.get_mapped(bcm)
    # at last reflatten the dataset
    # could we add some meaningful attribute during this mapping, i.e. would 
    # assigning 'inspace' do something good?
    ds = ds.get_mapped(FlattenMapper(shape=ds.samples.shape[1:]))
    # add samples attributes for the events, simply dump everything as a samples
    # attribute
    for a in evvars:
        # special case: we want the non-descrete, original onset and duration
        if a in ['onset', 'duration']:
            ds.sa[eprefix + '_attrs_' + a] = [e[a] for e in events]
        else:
            ds.sa[eprefix + '_attrs_' + a] = evvars[a]
    return ds
