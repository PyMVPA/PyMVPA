# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Support for storage using the NeuroImaging Markup Language (NIML).

Supports storing most typical values (samples, feature attributes, sample
attributes, dataset attributes) that are in a dataset in NIML format, as
long as these values are array-like.

Notes
-----
No support for 'sophisticated' values such as Mappers

.. versionadded:: 2.3.0

"""

__docformat__ = 'restructuredtext'

import numpy as np

import os

from mvpa2.support.nibabel import afni_niml_dset as niml_dset

from mvpa2.base.collections import SampleAttributesCollection, \
    FeatureAttributesCollection, DatasetAttributesCollection, \
    ArrayCollectable

from mvpa2.base import warning, debug, externals
from mvpa2.datasets.base import Dataset
from mvpa2.base import dataset

if externals.exists('h5py'):
    from mvpa2.base.hdf5 import h5save, h5load

_PYMVPA_PREFIX = 'PYMVPA'
_PYMVPA_SEP = '_'



def _as_vector_if_matrix_with_single_column(x):
    '''Helper function'''
    if isinstance(x, np.ndarray) and \
                    len(x.shape) == 2 and x.shape[1] == 1:
        return x.ravel()
    else:
        return x



def from_niml(dset, fa_labels=None, sa_labels=None, a_labels=None):
    '''Convert a NIML dataset to a Dataset

    Parameters
    ----------
    dset: dict
        Dictionary with NIML key-value pairs, such as obtained from
        mvpa2.support.nibabel.afni_niml_dset.read()
    fa_labels: list
        Keys in dset that are enforced to be feature attributes
    sa_labels: list
        Keys in dset that are enforced to be sample attributes
    a_labels: list
        Keys in dset that are enforced to be dataset attributes

    Returns
    -------
    dataset: mvpa2.base.Dataset
        a PyMVPA Dataset
    '''
    if fa_labels is None:
        fa_labels = []
    if sa_labels is None:
        sa_labels = []
    if a_labels is None:
        a_labels = []

    # check for singleton element
    if type(dset) is list and len(dset) == 1:
        # recursive call
        return from_niml(dset[0])

    if not type(dset) is dict:
        raise ValueError("Expected a dict")

    if not 'data' in dset:
        raise ValueError("dset with no data?")

    data = dset['data']
    if len(data.shape) == 1:
        nfeatures = data.shape[0]
        nsamples = 1
    else:
        nfeatures, nsamples = data.shape

    # some labels have predefined destinations
    sa_labels_ = ['labels', 'stats', 'chunks', 'targets'] + sa_labels
    fa_labels_ = ['node_indices', 'center_ids'] + fa_labels
    a_labels_ = ['history'] + a_labels
    ignore_labels = ('data', 'dset_type')

    sa = SampleAttributesCollection(length=nsamples)
    fa = FeatureAttributesCollection(length=nfeatures)
    a = DatasetAttributesCollection()

    labels_collections = [(sa_labels_, sa),
                          (fa_labels_, fa),
                          (a_labels_, a)]

    infix2collection = {'sa': sa,
                        'fa': fa,
                        'a': a}

    infix2length = {'sa': nsamples, 'fa': nfeatures}

    for k, v in dset.iteritems():
        if k in ignore_labels:
            continue

        if k.startswith(_PYMVPA_PREFIX + _PYMVPA_SEP):
            # special PYVMPA field - do the proper conversion
            k_split = k.split(_PYMVPA_SEP)
            if len(k_split) > 2:
                infix = k_split[1].lower()
                collection = infix2collection.get(infix, None)
                if collection is not None:
                    short_k = _PYMVPA_SEP.join(k_split[2:])
                    expected_length = infix2length.get(infix, None)
                    if expected_length:
                        if isinstance(v, np.ndarray) and np.dtype == np.str_:
                            v = str(v)

                        while isinstance(v, basestring):
                            # strings are seperated by ';'
                            # XXX what if this is part of the value
                            # intended by the user?
                            v = v.split(';')

                        if expected_length != len(v):
                            raise ValueError("Unexpected length: %d != %d" %
                                             (expected_length, len(v)))

                        v = _as_vector_if_matrix_with_single_column(v)

                        v = ArrayCollectable(v, length=expected_length)

                    collection[short_k] = v
                    continue

        v = _as_vector_if_matrix_with_single_column(v)

        found_label = False

        for label, collection in labels_collections:
            if k in label:
                collection[k] = v
                found_label = True
                break

        if found_label:
            continue


        # try to be smart and deduce this from dimensions.
        # this only works if nfeatures!=nsamples otherwise it would be
        # ambiguous
        # XXX is this ugly?
        if nfeatures != nsamples:
            try:
                n = len(v)
                if n == nfeatures:
                    fa[k] = v
                    continue
                elif n == nsamples:
                    sa[k] = v
                    continue
            except:
                pass

        # don't know what this is - make it a general attribute
        a[k] = v

    ds = Dataset(np.transpose(data), sa=sa, fa=fa, a=a)

    return ds



def to_niml(ds):
    '''Convert a Dataset to a NIML dataset

    Parameters
    ----------
    dataset: mvpa2.base.Dataset
        A PyMVPA Dataset

    Returns
    -------
    dset: dict
        Dictionary with NIML key-value pairs, such as obtained from
        mvpa2.support.nibabel.afni_niml_dset.read()
     '''
    if isinstance(ds, np.ndarray):
        ds = Dataset(ds)

    dset = dict(data=np.transpose(ds.samples))

    node_indices_labels = ('node_indices', 'center_ids', 'ids', 'roi_ids')
    node_indices = _find_node_indices(ds, node_indices_labels)
    if node_indices is not None:
        dset['node_indices'] = node_indices

    sample_labels = ('labels', 'targets')
    labels = _find_sample_labels(ds, sample_labels)
    if labels is not None:
        dset['labels'] = labels

    attr_labels = ('a', 'fa', 'sa')

    # a few labels are directly used in NIML dsets
    # without prefixing it with a pyMVPA string
    # for (dataset, feature, sample) attributes
    # here we define two for sample attributes
    attr_special_labels = ([], [], ['labels', 'stats'])

    for i, attr_label in enumerate(attr_labels):
        attr = getattr(ds, attr_label)
        special_labels = attr_special_labels[i]
        for k in attr.keys():
            v = attr[k]
            if hasattr(v, 'value'):
                v = v.value

            if k in special_labels:
                long_key = k
            else:
                long_key = _PYMVPA_SEP.join((_PYMVPA_PREFIX,
                                             attr_label.upper(), k))

            dset[long_key] = v

    return dset



def hstack(dsets, pad_to_feature_index=None, hstack_method='drop_nonunique',
           set_empty_value=0.):
    '''Stacks NIML datasets while considering node indices

    Parameters
    ----------
    dsets: list
        datasets to be stacked
    pad_to_feature_index: list or int or None
        If a list then it should be of the same length as dsets and indicates
        to which node index the input should be padded. A single int means
        that the same value is used for all dset in dsets. None means
        no padding, and is only allowed for non-sparse datasets.
    hstack_method: str:
        How datasets are stacked; see dataset.hstack.
    set_empty_value: float
        Value to which empty (padded) dataset values are set.

    Returns
    dset: Dataset
        Data combined from all dset in dsets.
    '''

    n = len(dsets)

    # make sure pad_to_feature_index has n values
    if pad_to_feature_index is None or type(pad_to_feature_index) is int:
        pad_to_feature_index = [pad_to_feature_index] * n
    elif len(pad_to_feature_index) != n:
        raise ValueError("illegal pad_to_feature_index: expected list or int")

    # labels that can contain node indices
    node_indices_labels = ('node_indices', 'center_ids', 'ids', 'roi_ids')
    node_indices = []

    # allocate space for output
    padded_dsets = []
    hstack_indices = []
    first_node_index = 0
    for i, (dset, pad_to) in enumerate(zip(dsets, pad_to_feature_index)):
        # get node indices in this dataset
        node_index = _find_node_indices(dset, node_indices_labels)
        if node_index is None:
            node_index = np.arange(dset.nfeatures)
        max_node_index = np.max(node_index)

        # make a stripped version - without node index labels
        stripped_dset = dset.copy()
        for label in node_indices_labels:
            if label in stripped_dset.fa:
                stripped_dset.fa.pop(label)

        # see if padding is needed
        if pad_to is None or pad_to == max_node_index + 1:
            if not np.array_equal(np.arange(max_node_index + 1), np.sort(node_index)):
                raise ValueError("Sparse input %d: need pad_to input" % (i + 1))
            padded_dset = stripped_dset
            other_index = np.arange(0)
        else:
            # have to use empty values
            nfeatures_empty = pad_to - dset.nfeatures
            if nfeatures_empty < 0:
                raise ValueError("Dataset has %d features, cannot pad "
                                 "to %d" % (dset.nfeatures, pad_to))

            # make empty array
            empty_arr = np.zeros((dset.nsamples, nfeatures_empty),
                                 dtype=dset.samples.dtype) + set_empty_value
            empty_dset = Dataset(empty_arr, sa=stripped_dset.sa.copy(deep=True))

            # combine current dset and empty array
            padded_dset = dataset.hstack((stripped_dset, empty_dset), hstack_method)

            # set the proper node indices
            other_index = np.setdiff1d(np.arange(pad_to), node_index)

        # sanity check to make sure that indices are ok
        # XXX could be more informative
        if len(np.setdiff1d(node_index, np.arange(pad_to or max_node_index + 1))):
            raise ValueError("Illegal indices")

        hstack_index = node_index + first_node_index
        hstack_other_index = other_index + first_node_index
        first_node_index += pad_to or (max_node_index + 1)  # prepare for next iteration

        padded_dsets.append(padded_dset)
        hstack_indices.append(hstack_index)
        if len(other_index):
            hstack_indices.append(hstack_other_index)

    hstack_dset = dataset.hstack(padded_dsets, hstack_method)
    hstack_indices = np.hstack(hstack_indices)

    hstack_dset.fa[node_indices_labels[0]] = hstack_indices

    return hstack_dset



def _find_sample_labels(dset, sample_labels):
    '''Helper function to find labels in this dataset.
    Looks for any in sample_labels and returns the first one
    that matches '''
    use_label = None

    dset_keys = dset.sa.keys()
    for label in sample_labels:
        if label in dset_keys:
            sample_label = dset.sa[label].value
            if isinstance(sample_label, basestring):
                # split using
                sample_label = sample_label.split(';')

            # they can be of any type so ensure they are strings
            sample_label_list = [str(i) for i in sample_label]
            if len(sample_label_list) != dset.nsamples:
                # unlike node indices here we are more lenient
                # so not throw an exception but just continue
                continue

            use_label = label

            # do not look for any other labels
            break

    return None if use_label is None else sample_label_list



def _find_node_indices(dset, node_indices_labels):
    '''Helper function to find node indices in this dataset
    Sees if any of the node_indices_labels is a feature attribute
    in the dataset and returns it. If they are multiple matches
    ensure they are identical, otherwise raise an error.
    A use case is searchlight results that assignes center_ids as
    a feature attributes, but it should be named node_indices
    before conversion to NIML format'''

    use_label = None

    dset_keys = dset.fa.keys()
    for label in node_indices_labels:
        if label in dset_keys:
            if use_label is None:
                # make vector and ensure all integer values
                node_indices = dset.fa[label].value
                node_indices = np.asarray(node_indices).ravel()
                if len(node_indices) != dset.nfeatures:
                    raise ValueError("Node indices mismatch: found %d values "
                                     " but dataset has %d features" %
                                     (len(node_indices, dset.nfeatures)))
                node_indices_int = np.asarray(node_indices, dtype=np.int)
                if not np.array_equal(node_indices_int, node_indices):
                    raise ValueError("Node indices should have integer values")
                use_label = label

            else:
                if not np.array_equal(dset.fa[label].value, node_indices_int):
                    raise ValueError("Different indices for feature attributes"
                                     " %s and %s" % (use_label, label))

    return None if use_label is None else node_indices_int



def write(fn, ds, form='binary'):
    '''Write a Dataset to a file in NIML format

    Parameters
    ----------
    fn: str
        Filename
    ds: mvpa2.base.Dataset
        Dataset to be stored
    form: str
        Data format: 'binary' or 'text' or 'base64'
    '''
    niml_ds = to_niml(ds)
    niml_dset.write(fn, niml_ds, form=form)



def read(fn):
    '''Read a Dataset from a file in NIML format

    Parameters
    ----------
    fn: str
        Filename
    '''

    readers_converters = {('.dset',): (niml_dset.read, from_niml)}

    if externals.exists('h5py'):
        readers_converters[('.h5py', '.hdf')] = (h5load, None)

    keys = [exts for exts in readers_converters.iterkeys()
            if any(fn.endswith(ext) for ext in exts)]

    n_keys = len(keys)

    if n_keys == 1:
        # single extension matches
        key = keys[0]
        reader, converter = readers_converters[key]

        r = reader(fn)
        if converter is not None:
            r = converter(r)
        return r

    else:
        # unclear extension, try all readers and throw less informative
        # error message
        for reader, converter in readers_converters.itervalues():
            try:
                r = reader(fn)
                if converter:
                    r = converter(r)
                return r

            except:
                pass

        raise ValueError("Unable to read %s with unclear extension" % fn)



def from_any(x):
    '''Get a Dataset from the input

    Parameters
    ----------
    x: str or dict or Dataset
        Filename, or NIML-dictionary, or a Dataset itself

    Returns
    -------
    ds: mvpa2.base.Dataset
        Dataset instance
    '''
    if isinstance(x, basestring):
        return read(x)
    elif isinstance(x, dict):
        return from_niml(x)
    elif isinstance(x, Dataset):
        return x

    raise ValueError("Not supported: %r" % (x,))
