# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Support for surface-based GIFTI data IO.

This module offers functions to import into PyMVPA surface-based GIFTI
data using NiBabel_, and export PyMVPA surface-based datasets back into
GIFTI.

The current implementation supports data associated with nodes, and
node indices for such data. There is no support for meta-data,
or non-identity affine transformations.

This module supports node data, i.e. each node on the surface has N
values associated with it (with N>=1). Typical examples include
time series data and statistical maps.

Optionally, anatomical information (vertices and faces) can be stored,
so that FreeSurfer's mris_convert can read data written by map2gifti.

.. _NiBabel: http://nipy.sourceforge.net/nibabel
"""

from mvpa2.base import externals

externals.exists('nibabel', raise_=True)

from nibabel.gifti import gifti, giftiio

from mvpa2.base.collections import FeatureAttributesCollection, \
    SampleAttributesCollection
from mvpa2.base.dataset import AttrDataset
from mvpa2.datasets.base import Dataset
from mvpa2.base import warning
from mvpa2.support.nibabel.surf import from_any as surf_from_any
from mvpa2.support.nibabel.surf_gifti import to_gifti_image as \
    anat_surf_to_gifti_image

import numpy as np



def _gifti_intent_niistring(intent_code):
    return gifti.intent_codes.niistring[intent_code]



def _gifti_intent_is_data(intent_string):
    # exclude a set of intents
    not_data_intent_strings = ('NIFTI_INTENT_GENMATRIX',
                               'NIFTI_INTENT_SYMMATRIX',
                               'NIFTI_INTENT_DISPVECT',
                               'NIFTI_INTENT_VECTOR',
                               'NIFTI_INTENT_POINTSET',
                               'NIFTI_INTENT_TRIANGLE',
                               'NIFTI_INTENT_QUATERNION',
                               'NIFTI_INTENT_DIMLESS',
                               'NIFTI_INTENT_NODE_INDEX',
                               'NIFTI_INTENT_SHAPE')

    return intent_string not in not_data_intent_strings



def _gifti_intent_is_node_indices(intent_string):
    return intent_string == 'NIFTI_INTENT_NODE_INDEX'



def _get_gifti_image(samples):
    if isinstance(samples, basestring):
        samples = giftiio.read(samples)

    required_class = gifti.GiftiImage
    if not isinstance(samples, required_class):
        raise TypeError('Input of type %s must be a %s' %
                        (samples, required_class))

    return samples



def gifti_dataset(samples, targets=None, chunks=None):
    """
    Parameters
    ----------
    samples : str or GiftiImage
      GIFTI surface-based data, specified either as a filename or an image.
    targets : scalar or sequence
      Label attribute for each volume in the timeseries.
    chunks : scalar or sequence
      Chunk attribute for each volume in the timeseries.
    """
    node_indices = None
    data_vectors = []
    intents = []

    image = _get_gifti_image(samples)

    for darray in image.darrays:
        intent_string = _gifti_intent_niistring(darray.intent)

        if _gifti_intent_is_data(intent_string):
            data_vectors.append(darray.data)
            intents.append(intent_string)

        elif _gifti_intent_is_node_indices(intent_string):
            node_indices = darray.data

    samples = np.asarray(data_vectors)
    nsamples, nfeatures = samples.shape

    # set sample attributes
    sa = SampleAttributesCollection(length=nsamples)

    sa['intents'] = intents

    if targets is not None:
        sa['targets'] = targets

    if chunks is not None:
        sa['chunks'] = chunks

    # set feature attributes
    fa = FeatureAttributesCollection(length=nfeatures)

    if node_indices is not None:
        fa['node_indices'] = node_indices

    return Dataset(samples=samples, sa=sa, fa=fa)



def map2gifti(ds, filename=None, encoding='GIFTI_ENCODING_B64GZ',
              surface=None):
    """Maps data(sets) into a GiftiImage, and optionally saves it to disc.

    Parameters
    ----------
    ds : AttrDataset or numpy.ndarray
      The data to be mapepd
    filename : basestring or None, optional
      Filename to which the GiftiImage is stored
    encoding : "ASCII" or "Base64Binary" or "GZipBase64Binary", optional
      Encoding format of data
    surface : mvpa2.surf.nibabel.surf.Surface or str, optional
      Optional anatomical Surface object, or filename of anatomical surface
      file, to be stored together with the data. This should allow
      FreeSurfer's mris_convert to read files written by this function

    Returns
    -------
    img : GiftiImage
      dataset contents represented in GiftiImage
    """

    darrays = []

    if isinstance(ds, np.ndarray):
        samples = ds
    elif isinstance(ds, AttrDataset):
        samples = ds.samples
        _warn_if_fmri_dataset(ds)
    else:
        raise TypeError('first argument must be AttrDataset or numpy.ndarray')

    [nsamples, nfeatures] = samples.shape

    def _get_attribute_value(ds, attr_name, keys_):
        if isinstance(ds, np.ndarray):
            # no attributes
            return None

        attr_collection = ds.__dict__.get(attr_name)

        if isinstance(keys_, basestring):
            keys_ = (keys_,)

        for key in keys_:
            if key in attr_collection:
                return attr_collection[key].value
        return None

    def _build_array(data, intent, encoding=encoding):
        is_integer = intent == 'NIFTI_INTENT_NODE_INDEX'
        dtype = np.int32 if is_integer else np.float32

        arr = gifti.GiftiDataArray.from_array(data.astype(dtype), intent,
                                              encoding=encoding)
        # Setting the coordsys argument the constructor would set the matrix
        # to the 4x4 identity matrix, which is not desired. Instead the
        # coordsys is explicitly set to None afterwards
        arr.coordsys = None

        return arr

    node_indices_labels = ('node_indices', 'center_ids', 'ids', 'roi_ids')
    node_indices = _get_attribute_value(ds, 'fa', node_indices_labels)

    if node_indices is not None:
        darray = _build_array(node_indices, 'NIFTI_INTENT_NODE_INDEX')
        darrays.append(darray)

    intents = _get_attribute_value(ds, 'sa', 'intents')
    for i, sample in enumerate(samples):
        intent = 'NIFTI_INTENT_NONE' if intents is None else intents[i]
        darray = _build_array(sample, intent)
        darrays.append(darray)

    # if there is a surface, add it
    if surface is not None:
        surface_object = surf_from_any(surface, )
        anat_image = anat_surf_to_gifti_image(surface_object, add_indices=False)

        for darray in anat_image.darrays:
            darrays.append(darray)

    image = gifti.GiftiImage(darrays=darrays)

    if filename is not None:
        giftiio.write(image, filename)

    return image



def _warn_if_fmri_dataset(ds):
    assert (isinstance(ds, AttrDataset))

    fmri_fields = set(('imgaffine', 'imgtype', 'imghdr'))

    ds_fmri_fields = set.intersection(set(ds.a.keys()), fmri_fields)

    if len(ds_fmri_fields) > 0:
        warning('dataset attribute .a has fields %s, which suggest it is an '
                'volumetric dataset. Converting this dataset to GIFTI '
                'format will most likely result in unvisualiable '
                '(and potentially, un-analysable) data. Consider using '
                'map2nifti instead' % (', '.join(ds_fmri_fields)))
