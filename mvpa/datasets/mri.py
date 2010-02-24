# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset for magnetic resonance imaging (MRI) data.

This module offers functions to import MRI data in the NIfTI format into
PyMVPA, and export PyMVPA datasets back into NIfTI files.

Currently NIfTI file access is based on PyNIfTI_.

.. _PyNIfTI: http://niftilib.sourceforge.net/pynifti
"""

__docformat__ = 'restructuredtext'

from mvpa.base import externals

import sys
import numpy as np
from mvpa.support.copy import deepcopy
from mvpa.misc.support import Event
from mvpa.base.collections import DatasetAttribute
from mvpa.base.dataset import _expand_attribute

if __debug__:
    from mvpa.base import debug

if externals.exists('nifti', raiseException=True):
    from nifti import NiftiImage

from mvpa.datasets.base import Dataset
from mvpa.mappers.fx import _uniquemerge2literal
from mvpa.mappers.flatten import FlattenMapper
from mvpa.mappers.boxcar import BoxcarMapper
from mvpa.base import warning


def map2nifti(dataset, data=None, imghdr=None):
    """Maps data(sets) into the original dataspace and wraps it in a NiftiImage.

    Parameters
    ----------
    dataset : Dataset
      The mapper of this dataset is used to perform the reverse-mapping.
    data : ndarray or Dataset, optional
      The data to be wrapped into NiftiImage. If None (default), it
      would wrap samples of the current dataset. If it is a Dataset
      instance -- takes its samples for mapping.
    imghdr : dict
      Image header data. If None, the header is taken from `dataset`.

    Returns
    -------
    NiftiImage
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

    if imghdr is None:
        imghdr = dataset.a.imghdr

    return NiftiImage(dsarray, imghdr)


def fmri_dataset(samples, targets=None, chunks=None, mask=None,
                 sprefix='voxel', tprefix='time', add_fa=None,):
    """Create a dataset from an fMRI timeseries image.

    The timeseries image serves as the samples data, with each volume becoming
    a sample. All 3D volume samples are flattened into one-dimensional feature
    vectors, optionally being masked (i.e. subset of voxels corresponding to
    non-zero elements in a mask image).

    In addition to (optional) samples attributes for targets and chunks the
    returned dataset contains a number of additional attributes:

    Samples attributes (per each volume):

      * volume index (time_indices)
      * volume acquisition time (time_coord)

    Feature attributes (per each voxel):

      * voxel indices (voxel_indices), sometimes referred to as ijk

    Dataset attributes:

      * dump of the NIfTI image header data (imghdr)
      * volume extent (voxel_dim)
      * voxel extent (voxel_eldim)

    The default attribute name is listed in parenthesis, but may be altered by
    the corresponding prefix arguments. The validity of the attribute values
    relies on correct settings in the NIfTI image header.

    Parameters
    ----------
    samples : str or NiftiImage or list
      fMRI timeseries, specified either as a filename (single file 4D image),
      an image instance (4D image), or a list of filenames or image instances
      (each list item corresponding to a 3D volume).
    targets : scalar or sequence
      Label attribute for each volume in the timeseries, or a scalar value that
      is assigned to all samples.
    chunks : scalar or sequence
      Chunk attribute for each volume in the timeseries, or a scalar value that
      is assigned to all samples.
    mask : str or NiftiImage
      Filename or image instance of a 3D volume mask. Voxels corresponding to
      non-zero elements in the mask will be selected. The mask has to be in the
      same space (orientation and dimensions) as the timeseries image
    sprefix : str or None
      Prefix for attribute names describing spatial properties of the
      timeseries. If None, no such attributes are stored in the dataset.
    tprefix : str or None
      Prefix for attribute names describing temporal properties of the
      timeseries. If None, no such attributes are stored in the dataset.
    add_fa : dict or None
      Optional dictionary with additional volumetric data that shall be stored
      as feature attributes in the dataset. The dictionary key serves as the
      feature attribute name. Each value might be of any type supported by the
      'mask' argument of this function.

    Returns
    -------
    Dataset
    """
    # load the samples
    niftisamples = _load_anynifti(samples, ensure=True, enforce_dim=4)
    samples = niftisamples.data

    # figure out what the mask is, but onyl handle known cases, the rest
    # goes directly into the mapper which maybe knows more
    niftimask = _load_anynifti(mask)
    if niftimask is None:
        pass
    elif isinstance(niftimask, np.ndarray):
        mask = niftimask
    else:
        mask = _get_nifti_data(niftimask)

    # compile the samples attributes
    sa = {}
    if not targets is None:
        sa['targets'] = _expand_attribute(targets, samples.shape[0], 'targets')
    if not chunks is None:
        sa['chunks'] = _expand_attribute(chunks, samples.shape[0], 'chunks')

    # create a dataset
    ds = Dataset(samples, sa=sa)
    if sprefix is None:
        inspace = None
    else:
        inspace = sprefix + '_indices'
    ds = ds.get_mapped(FlattenMapper(shape=samples.shape[1:], inspace=inspace))

    # now apply the mask if any
    if not mask is None:
        flatmask = ds.a.mapper.forward1(mask)
        # direct slicing is possible, and it is potentially more efficient,
        # so let's use it
        #mapper = FeatureSliceMapper(flatmask)
        #ds = ds.get_mapped(FeatureSliceMapper(flatmask))
        ds = ds[:, flatmask != 0]

    # load and store additional feature attributes
    if not add_fa is None:
        for fattr in add_fa:
            value = _get_nifti_data(_load_anynifti(add_fa[fattr]))
            ds.fa[fattr] = ds.a.mapper.forward1(value)

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
        ds.sa[tprefix + '_indices'] = np.arange(len(ds), dtype='int')
        ds.sa[tprefix + '_coords'] = np.arange(len(ds), dtype='float') \
                                     * niftisamples.header['pixdim'][4]
        # TODO extend with the unit

    return ds


def _get_nifti_data(nim):
    """Convenience function to extract the data array from a NiftiImage

    This function will make use of advanced features of PyNIfTI to prevent
    unnecessary copying if a sufficent version is available.
    """
    if externals.exists('nifti ge 0.20090205.1'):
        return nim.data
    else:
        return nim.asarray()


def _load_anynifti(src, ensure=False, enforce_dim=None):
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
            warning("ERROR: Cannot open NIfTI file %s" \
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
        srcs = [_load_anynifti(s, ensure=ensure,
                                      enforce_dim=enforce_dim_)
                for s in src]
        if __debug__:
            # lets check if they all have the same dimensionality
            shapes = [s.data.shape for s in srcs]
            if not np.all([s == shapes[0] for s in shapes]):
                raise ValueError, \
                      "Input volumes contain variable number of dimensions:" \
                      " %s" % (shapes,)
        # Combine them all into a single beast
        nifti = NiftiImage(np.array([s.asarray() for s in srcs]),
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



