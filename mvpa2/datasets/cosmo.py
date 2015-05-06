# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset from CoSMoMVPA_

This module provides basic I/O support for datasets in CoSMoMVPA_.
The current implementation provides (1) loading and saving a CoSMoMVPA dataset
struct, which is converted to a PyMVPA Dataset object; and (2) loading
a CoSMoMVPA neighborhood struct, which is converted to a CoSMoQueryEngine
object that inherits from QueryEngineInterface.

A use case is running searchlights on MEEG data, e.g.:

1. FieldTrip_ is used to preprocess MEEG data
2. CoSMoMVPA is used to convert the preprocessed MEEG data to a CoSMoMVPA
   dataset struct and generate neighborhood information for the searchlight
3. this module (mvpa2.datasets.cosmo) is used to import the preprocessed MEEG
   data and the neighborhood information into PyMVPA-objects
4. PyMVPA is used to run a searchlight with a Measure of interest
5. this module (mvpa2.datasets.cosmo) is used to export the searchlight output
   to a CoSMoMVPA dataset struct
6. CoSMoMVPA is used to convert the CoSMoMVPA dataset struct with the
   searchlight output to a FieldTrip struct
7. FieldTrip is used to visualize the results


Example
=======

Suppose that in Matlab using CoSMoMVPA, two structs were created:

ds (with fields .samples, .sa, .fa and .a)
   containing a dataset (e.g. an fMRI, MEEG, or surface-based dataset).
   Such a struct is typically defined in CoSMoMVPA using
   cosmo_{fmri,meeg,surface}_dataset
nbrhood (with fields .neighbors, .fa and .a)
   containing neighborhood information for each feature in ds.
   Such a struct is typically defined in CoSMoMVPA using cosmo_neighborhood.

Alternatively they can be defined in Matlab directly without use of CoSMoMVPA
functionality.  For a toy example, consider the following Matlab code::

  >> ds=struct();
  >> ds.samples=[1 2 3; 4 5 6];
  >> ds.a.name='input';
  >> ds.fa.i=[1 2 3];
  >> ds.fa.j=[1 2 2];
  >> ds.sa.chunks=[2 2]';
  >> ds.sa.targets=[1 2]';
  >> ds.sa.labels={'a','b','c','d';'e','f','g','h'};
  >> save('ds_tiny.mat','-struct','ds');

  >> nbrhood=struct();
  >> nbrhood.neighbors={1, [1 3], [1 2 3], [2 2]};
  >> nbrhood.fa.k=[4 3 2 1];
  >> nbrhood.a.name='output';
  >> save('nbrhood_tiny.mat','-struct','nbrhood');


These can be stored in Matlab by::

  >> save('ds.mat','-struct','ds')
  >> save('nbrhood.mat','-struct','nbrhood')

and loaded in Python using::

>>> import mvpa2
>>> import os
>>> from mvpa2.datasets.cosmo import from_any, CosmoSearchlight
>>> from mvpa2.mappers.fx import mean_feature
>>> data_path=os.path.join(mvpa2.pymvpa_dataroot,'cosmo')
>>> fn_mat_ds=os.path.join(data_path,'ds_tiny.mat')
>>> fn_mat_nbrhood=os.path.join(data_path,'nbrhood_tiny.mat')
>>> ds=from_any(fn_mat_ds)
>>> print ds
<Dataset: 2x3@float64, <sa: chunks,labels,targets>, <fa: i,j>, <a: name>>
>>> qe=from_any(fn_mat_nbrhood)
>>> print qe
CosmoQueryEngine(4 center ids (0 .. 3), <fa: k>, <a: name>

where ds is a :class:`~mvpa2.datasets.base.Dataset` and qe a
:class:`~mvpa2.datasets.cosmo.CosmoQueryEngine`.

A :class:`~mvpa2.measures.base.Measure` of choice can be used for a searchlight;
here the measure simply takes the mean over features in each searchlight::
>>> measure=mean_feature()

A searchlight can be run the CosmoQueryEngine
>>> sl=CosmoSearchlight(measure, qe)
>>> ds_sl=sl(ds)
>>> print ds_sl
<Dataset: 2x4@float64, <sa: chunks,labels,targets>, <fa: k>, <a: name>>

Note that the output dataset has the feature and sample attributes taken
from the queryengine, *not* the dataset.

Alternatively it is possible to run the searchlight directly using the
filename of the neighborhood .mat file::

>>> sl=CosmoSearchlight(measure, fn_mat_nbrhood)
>>> ds_sl=sl(ds)
>>> print ds_sl
<Dataset: 2x4@float64, <sa: chunks,labels,targets>, <fa: k>, <a: name>>

which gives the same result as above.

Leaving the doctest format here, subsequently the result can be
stored in Python using::

  >> map2cosmo(ds_sl,'ds_sl.mat')

and loaded in Matlab using::

  >> ds_sl=importdata('ds_sl.mat')

so that in Matlab ds_sl is a dataset struct with the output
of applying measure to the neighborhoods defined in nbrhood.

Notes
=====

- This function does not provide or deal with mappers associated with a dataset.
  For this reason map2nifti does not work on PyMVPA fmri datasets that were
  imported from CoSMoMVPA using this module.  Instead, CoSMoMVPA's
  map2fmri in Matlab can be used to map results to nifti and other formats
- The main difference between the searchlight approach in CoSMoMVPA versus
  PyMVPA is that CoSMoMVPA allows for setting feature (.fa) and dataset
  (.a) attributes for the output explicitly in its QueryEngineInterface.
  Use cases are (a) surface-based searchlight of fMRI data (with .fa set
  to the node indices of the output) and (b) MEEG searchlight of timelocked
  data that uses all (or a subset of the) sensors for the input and provides
  a time-course of MVP results for the output; where the input data has
  features of time by sensor, while the output data has only time.

.. _CoSMoMVPA: http://www.github.com/CoSMoMVPA
.. _FieldTrip: http://fieldtrip.fcdonders.nl/
"""

__docformat__ = 'restructuredtext'

from mvpa2.base import externals
externals.exists('scipy', raise_=True)

from scipy.io import loadmat, savemat, matlab
import numpy as np

from mvpa2.datasets.base import Dataset
from mvpa2.featsel.base import StaticFeatureSelection
from mvpa2.mappers.base import ChainMapper
from mvpa2.base.collections import Collection #ArrayCollectable
from mvpa2.misc.neighborhood import QueryEngineInterface
from mvpa2.measures.searchlight import Searchlight
from mvpa2.base import debug, warning


def _numpy_array_astype_unsafe(arr, type_):
    '''
    Helper function to deal with API change in numpy.

    Rationale: prior to version 1.7 it seems the 'casting' argument is not
    supported, whereas later versions do support it.
    '''
    casting_support_version = '1.7'

    if np.__version__ < casting_support_version:
        return arr.astype(type_)
    else:
        return arr.astype(type_, casting='unsafe')

def _from_singleton(x, ndim=2):
    '''
    If x is an array of shape (1,1,...1) with ndim dimensions it returns the
    one single element in it. For other shapes it returns an error
    '''
    s = (1,) * ndim
    if x.shape != s:
        raise ValueError("Expected singleton shape %s for %s, found %s" %
                                (s, x, x.shape,))
    return x[(0,) * ndim]



# dictionary indicating whether elements in .fa, .sa and .a must be
# transposed for CoSMoMVPA data representation. As in CoSMoMVPA a dataset
# with .samples PxQ the elements in .fa must be ?xQ, (only) .fa requires
# transpose
_attr_fieldname2do_transpose = dict(fa=True, sa=False, a=False)
_is_private_key = lambda s: s.startswith('__') and s.endswith('__')

def _loadmat_internal(fn):
    '''
    Helper function to load matlab data

    Parameters
    ----------
    fn: basestring
        Filename of Matlab .mat file

    Returns
    -------
    mat: dict
        Data in fn

    Notes
    -----
    Data is loaded with mat_dtype=True so that e.g. data stored in float
    (in Matlab) are not converted to int if all values are integers.
    '''

    return loadmat(fn, mat_dtype=True)



def _attributes_cosmo2dict(cosmo):
    '''
    Converts CoSMoMVPA-like attributes to a dictionary form

    Parameters
    ----------
    cosmo: dict
        Dictionary that may contains fields 'sa', 'fa', 'a'. For any of these
        fields the contents can be a dict, np.ndarray (object array as returned
        by loadmat) or ArrayCollectable (from a PyMVPA Dataset's .a, .fa or .sa)

    Returns
    -------
    pymvpa_attributes: dict
        Data represented in cosmo with fields 'sa', 'fa' and 'a'. Each element
        in pymvpa_attributes[key] is a dict itself mapping an attribute name
        to a value.
    '''

    # space for output
    pymvpa_attributes = dict()

    # go over 'sa', 'fa' and 'a'
    for fieldname, do_transpose in _attr_fieldname2do_transpose.iteritems():
        attrs = dict()

        if fieldname in cosmo:
            v = cosmo[fieldname]

            if type(v) is dict:
                # copy the data over
                attrs.update(v)

            elif isinstance(v, np.ndarray):
                # extract singleton element
                fsa_mat = _from_singleton(v)

                if fsa_mat is not None:
                    # assume an object array
                    fsa_keys = fsa_mat.dtype.names

                    for fsa_key in fsa_keys:
                        dim = fsa_mat[fsa_key]

                        if do_transpose:
                            # feature attribute case, to match dimensionality
                            # in second dimension
                            dim = dim.T

                        # transform row-vectors in matrix form (shape=(1,P))
                        # to vectors (shape=(P,))
                        if len(dim.shape) == 2 and dim.shape[1] == 1:
                            dim = dim.ravel()

                        attrs[fsa_key] = dim

            elif isinstance(v, Collection):
                # from PyMVPA Dataset, extract keys and values
                attrs.update((k, v[k].value) for k in v)

            elif v is None:
                pass

            else:
                raise TypeError('Unsupported input %s' % v)

        pymvpa_attributes[fieldname] = attrs

    return pymvpa_attributes


def _attributes_dict2cosmo(ds):
    '''
    Converts attributes in Dataset to CoSMoMVPA-like attributes

    Parameters
    ----------
    ds: Dataset
        input dataset with .sa, .fa and .a

    Returns
    -------
    cosmo: dict
        dictionary with keys 'sa', 'fa' and 'a', each for which the
        corresponding values are np.array objects with the fieldnames
        corresponding to the fieldnames of the input dataset

    Notes
    -----
    The output can be used by savemat to store a matlab file
    '''

    cosmo = dict()

    # go over 'sa', 'fa' and 'a'
    for fieldname, do_transpose in _attr_fieldname2do_transpose.iteritems():
        attr_collection = getattr(ds, fieldname)

        if attr_collection:
            # only store if non-empty

            dtypes = []
            values = []

            for k in attr_collection:
                value = attr_collection[k].value

                if len(value.shape) == 1:
                    # transform vectors (shape=(P,)) to vectors in matrix
                    # form (shape=(1,P))
                    value = np.reshape(value, (-1, 1))

                if do_transpose:
                    # feature attribute case, to match dimensionality
                    # in second dimension
                    value = value.T

                dtypes.append((k, 'O'))
                values.append(value)

            dtype = np.dtype(dtypes)
            arr = np.array([[tuple(values)]], dtype=dtype)

            cosmo[fieldname] = arr

    return cosmo



def _mat_replace_matlab_function_by_string(x):
    '''
    Replace matlab function handles (as read by matread) by a string.

    Parameters
    ----------
    x : object

    Returns
    -------
    y : object
        if x is a scipy.io.matlab.mio5_params.MatlabFunction then
        y is a string representation of x, otherwise y is equal to x

    Notes
    -----
    scipy can read but not write Matlab function hanndles; the use case of
    this function is to replace such function handles by something that scipy
    can write
    '''

    if isinstance(x, matlab.mio5_params.MatlabFunction):
        return np.asarray('%s' % x)

    return None


def _mat_make_saveable(x, fixer=_mat_replace_matlab_function_by_string):
    '''
    Make a Matlab data structure saveable by scipy's matsave

    Parameters
    ----------
    x : object
        Input to be made saveable
    fixer: callable (default: _mat_replace_matlab_function_by_string)
        Function that can replace un-saveable elements by a saveable version

    Returns
    -------
    y : object
        Saveable version of x.

    Notes
    -----
    scipy can read but not write Matlab function handles; the use case of
    this function is to replace such function handles by something that scipy
    can write.

    The present implementation processes the input recursively and makes a
    copy of the entire input.
    With the default fixer, function handles are replaced by a string
    representation of the funciton handle.
    '''

    y = fixer(x)
    if y is not None:
        # x has been fixed; return result
        return y

    # x was not of a type that could be fixed; see if it has
    # elements that can be fixed through recursion

    if type(x) is dict:
        # use recursion
        return dict((k, _mat_make_saveable(v, fixer=fixer))
                        for k, v in x.iteritems())

    elif isinstance(x, np.ndarray) and x.dtype.names is None:
        # standard array or object array
        n = x.size

        # get tuple indices for N-dimensional array
        idxs = zip(*np.unravel_index(np.arange(n), x.shape))

        # only object arrays need fixing. Other types, e.g. float arrays
        # can be ignored
        needs_fixing = x.dtype == np.dtype('O')

        vs = x.copy()
        if needs_fixing:
            for idx in idxs:
                # use recursion
                vs[idx] = _mat_make_saveable(x[idx], fixer=fixer)

        return vs

    elif isinstance(x, np.ndarray) and x.dtype.names is not None:
        # cell array
        arr = x.copy()
        for k in x.dtype.names:
            # use recursion
            arr[k] = _mat_make_saveable(x[k], fixer=fixer)

        return arr

    elif isinstance(x, basestring):
        # anything else, e.g. __header__
        return x

    else:
        raise TypeError('Unexpected type %s in %s' % (type(x), x))


def _check_cosmo_dataset(cosmo):
    '''
    Helper function to ensure a cosmo input for cosmo_dataset is valid.
    Currently does two things:
    (1) raise an error if there are no samples
    (2) raise a warning if samples have very large or very small values. A use
        case is certain MEEG datasets with very small sample values
        (in the order of 1e-25) which affects some classifiers
    '''

    samples = cosmo.get('samples', None)

    if samples is None:
        raise KeyError("Missing field .samples in %s" % cosmo)

    # check for extreme values
    warn_for_extreme_values_decimals = 10

    # ignore NaNs and infinity
    nonzero_msk = np.logical_and(np.isfinite(samples), samples != 0)
    max_nonzero = np.max(np.abs(samples[nonzero_msk]))

    # see how many decimals in the largest absolute value
    decimals_nonzero = np.log10(max_nonzero)

    if abs(decimals_nonzero) > warn_for_extreme_values_decimals:
        msg = ('Samples have extreme values, maximum absolute value is %s; '
             'This may affect some analyses. Considering scaling the samples, '
             'e.g. by a factor of 10**%d ' % (max_nonzero, -decimals_nonzero))
        warning(msg)


def cosmo_dataset(cosmo):
    '''
    Construct Dataset from CoSMoMVPA format

    Parameters
    ----------
    cosmo: str or Dataset-like or dict
        If a str it is treated as a filename of a .mat file with a matlab
        struct used in CoSMoMVPA, i.e. a struct with fields .samples, .sa,
        .fa, and .a.
        If a dict is is treated like the result from scipy's loadmat of
        a matlab struct used in CoSMoMVPA.

    Returns
    -------
    ds : Dataset
        PyMVPA Dataset object with values in .samples, .fa., .sa and .a
        based on the input
    '''

    if isinstance(cosmo, basestring):
        # load file
        cosmo = _loadmat_internal(cosmo)

    # do some sanity checks
    _check_cosmo_dataset(cosmo)

    # store samples
    args = dict(samples=cosmo['samples'])

    # set dataset, feature and sample attributes
    args.update(_attributes_cosmo2dict(cosmo))

    # build dataset using samples, fa, sa and a arguments
    return Dataset(**args)


def map2cosmo(ds, filename=None):
    '''
    Convert PyMVPA Dataset to CoSMoMVPA struct saveable by scipy's savemat

    Parameters
    ----------
    ds : Dataset
        PyMVPA dataset to be converted
    filename: None or basestring
        If not None, the conversion result is saved to the file named
        filename using scipy's savemat

    Returns
    -------
    cosmo : dict
        dictionary that can be saved using scipy's savemat
    '''

    # set samples
    cosmo = dict(samples=ds.samples)

    # set feature, sample and dataset attributes
    cosmo.update(_attributes_dict2cosmo(ds))

    # remove elements not saveable by scipy (e.g. function handles)
    cosmo_fixed = _mat_make_saveable(cosmo)

    # do some sanity checks
    _check_cosmo_dataset(cosmo_fixed)

    # optionally store to disc
    if filename is not None:
        savemat(filename, cosmo_fixed)

    return cosmo_fixed


def from_any(x):
    '''
    Load CoSMoMVPA dataset or neighborhood

    Parameters
    ----------
    x : basestring or dict or Dataset or CosmoQueryEngine
        If a basestring it is interpreted as a filename of a .mat file
        with a CoSMoMVPA dataset or neighborhood struct, and the contents
        are returned.
        If a dict it is interpreted as the result from scipy's loadmat and its
        contents are returned
        If a Dataset or CosmoQueryEngine then x is returned immediately

    Returns
    -------
    y : Dataset or CosmoQueryEngine
        If x refers to a Dataset (has .samples) then a Dataset is
        returned with .fa., .sa and .a taken from the input (if present)
        If x refers to a CosmoQueryEngine (has .neighbors) then a
        CosmoQueryEngine is returned.
    '''

    if isinstance(x, basestring):
        x = _loadmat_internal(x)

    if isinstance(x, dict):
        x_keys = x.keys()
        # remove private headers so that CosmoQueryEngine.from_mat won't choke
        x = dict((k, v) for k, v in x.iteritems() if not _is_private_key(k))

        for depth in (0, 1):
            if 'samples' in x:
                return cosmo_dataset(x)
            elif 'neighbors' in x:
                return CosmoQueryEngine.from_mat(**x)
            elif len(x) == 1 and depth == 0:
                # case of using 'save' in matlab with '-struct' with single
                # variable; get the value of the variable and try again
                _, v = x.popitem()

                x = dict((k, _from_singleton(v[k])) for k in v.dtype.names)

                continue
            else:
                raise ValueError('Unrecognized dict with keys %s' % x_keys)

    elif isinstance(x, (Dataset, CosmoQueryEngine)):
        # already good type, return it directly
        return x

    raise ValueError('Unrecognized input %s' % x)


class CosmoQueryEngine(QueryEngineInterface):
    '''
    queryengine for neighborhoods defined in CoSMoMVPA.
    This class behaves like a normal QueryEngine, and its use is intended
    with a searchlight. It differs in that it contains the dataset (.a)
    and feature (.fa) attributes for the output of a searchlight. This is
    implemented by the method set_output_dataset_attributes.
    Although the standard Searchlight can be used with this function,
    using CosmoSearchlight automatically calls this method so that the
    dataset attributes for the output are properly set.

    Example
    -------
    # ds.mat      is a dataset      struct saved in matlab using CoSMoMVPA
    # nbrhood.mat "  " neighborhood "                                    "
    >> ds=from_any('ds.mat')       # PyMVPA Dataset
    >> qe=from_any('nbrhood.mat')  # PyMVPA CosmoQueryEngine

    # alternative to define query engine:
    >> qe_alt=CosmoQueryEngine.from_mat('nbrhood.mat')

    # define measure
    >> measure=mean_features()

    # define searchlight
    >> sl=CosmoSearchlight(measure, qe)

    # run searchlight
    >> ds_res=sl(ds)

    # store result
    >> map2cosmo(ds_res,'result.mat')

    '''
    def __init__(self, mapping, a=None, fa=None):
        '''
        Parameters
        ----------
        mapping: dict
            mapping from center ids (int) to array of feature ids (numpy
            array of datatype int)
        a: None or dict or ArrayCollectable
            dataset attributes to be used for the output of a Searchlight
        fa: None or dict or ArrayCollectable
            dataset attributes to be used for the output of a Searchlight
        '''

        super(CosmoQueryEngine, self).__init__()

        # check and store mapping
        self._check_mapping(mapping)
        self._mapping = mapping

        # store center ids
        self._ids = ids = np.asarray(mapping.keys())

        # get feature and dataset attributes
        attributes = _attributes_cosmo2dict(dict(a=a, fa=fa))

        # see how many features there are for the output
        fa = attributes.get('fa', None)

        if not fa:
            nfeatures = np.max(ids) + 1
        else:
            nfeatures = len(fa[next(iter(fa))])

        # make a template dataset, so that feature and dataset attributes
        # are stored properly (with slicing support through Dataset).
        # the method set_output_dataset_attributes uses this template
        # dataset so that after running CosmoSearchlight the dataset (.a)
        # and feature (.fa) attributes can be set for the output.
        #
        # Note: a Dataset is used for convenience here, because it provides
        #       (1) automatic checks for proper size of feature attributes
        #       relative to the size of samples, and  (2) slicing
        attributes['samples'] = np.zeros((0, nfeatures), dtype=np.int_)
        self._dataset_template = Dataset(**attributes)[:, ids]

    @staticmethod
    def _check_mapping(mapping):
        '''
        Simple checks to ensure the provided mapping is kosher
        '''
        if not isinstance(mapping, dict):
            raise TypeError('Mapping must be dict, found %s' % type(mapping))

        for k, v in mapping.iteritems():
            if not np.isscalar(k):
                raise ValueError('Key %s not a scalar' % k)
            if not isinstance(k, int):
                raise ValueError('Keys %s must be int, found %s' % (k, type(k)))
            if not isinstance(v, np.ndarray):
                raise TypeError('Value %s for key %s must be numpy array' %
                                                                    (v, k))
            if not np.issubdtype(np.int_, v.dtype):
                raise ValueError('Value %s for key %s must be int' % (v, k))


    @classmethod
    def from_mat(cls, neighbors, a=None, fa=None):
        '''
        Create CosmoQueryEngine from mat struct

        Parameters
        ----------
        neighbors: numpy.object
            Object from scipy's matload; must have been a Px1 cell
            with in each cell a vector with indices of neighboring features
            in base 1. Typically this is from a CoSMoMVPA neighborhood struct.
        a: None or dict or ArrayCollectable
            dataset attributes to be used for the output of a Searchlight
        fa: None or dict or ArrayCollectable
            dataset attributes to be used for the output of a Searchlight

        Notes
        -----
        Empty elements are ignored
        '''

        neighbors_vec = neighbors.ravel()
        mapping = dict()

        for id, nbr_fids in enumerate(neighbors_vec):
            nbr_fids_vec = nbr_fids.ravel()

            if len(nbr_fids_vec):

                if min(nbr_fids_vec) < 1:
                    raise ValueError('Negative index for id %s' % id)

                if not np.all(np.equal(nbr_fids, np.round(nbr_fids))):
                    raise ValueError('Non-integer indices for id %s' % id)

                nbr_fids_vec_int = _numpy_array_astype_unsafe(nbr_fids_vec,
                                                              np.int_)

                # store mapping, convert base 1 (Matlab) to base 0 (Python)
                mapping[int(id)] = nbr_fids_vec_int - 1

        return cls(mapping, a=a, fa=fa)


    def __repr__(self):
        '''
        Return representation of this instance
        '''
        template = self._dataset_template

        return '%s(mapping=%r, a=%r, sa=%r)' % (self.__class__.__name__,
                                                self._mapping,
                                                template.a,
                                                template.fa)

    def __str__(self):
        '''
        Return string summary of this instance
        '''

        template = self._dataset_template
        ids = self.ids

        return ('%s(%d center ids (%d .. %d), <fa: %s>, <a: %s>' %
                        (self.__class__.__name__, len(ids),
                         np.min(ids), np.max(ids),
                         ', '.join(template.fa.keys()),
                         ', '.join(template.a.keys())))

    def __reduce__(self):
        '''
        Return state of the instance that can be pickled
        '''
        template = self._dataset_template
        return (self.__class__, (self._mapping, template.a, template.fa))

    def __len__(self):
        '''
        Return number of ids (keys)
        '''
        return len(self.ids)

    def train(self, dataset):
        '''
        This method does nothing
        '''
        pass

    def untrain(self):
        '''
        This method does nothing
        '''
        pass

    def query(self, **kwargs):
        raise NotImplementedError

    def query_byid(self, id):
        '''
        Returns
        ------
        fids : np.ndarray
            vector with feature indices of neighbors of the feature indexed
            by id
        '''

        return self._mapping[id]

    @property
    def ids(self):
        '''
        Returns
        -------
        keys: npndarray
            vector with feature indices that can be used as keys
        '''

        return self._ids

    @property
    def a(self):
        '''
        Returns
        -------
        a : DatasetAttributesCollection
            Dataset attributes for the output dataset from using this instance
        '''
        return self._dataset_template.a

    @property
    def fa(self):
        '''
        Returns
        -------
        fa : FeatureAttributesCollection
            Feature attributes for the output dataset from using this instance.
            It has as many elements as self.ids
        '''
        return self._dataset_template.fa

    def set_output_dataset_attributes(self, ds):
        '''
        Set attributes to output dataset (e.g. after running a searchlight)

        Parameters
        ----------
        ds : Dataset
            dataset with ds.fa.center_ids containing the center id of each
            feature

        Returns
        ds_copy : Dataset
            copy of ds, but with feature (.fa) and dataset (.a)
            attributes provided to the contstructor of this instance.
            The .fa and .a from the input ds are removed first.
        '''
        if not 'center_ids' in ds.fa:
            raise KeyError('Dataset seems not to be the result from '
                            'running a searchlight: missing .fa.center_ids')

        center_ids = ds.fa.center_ids

        ds_template = self._dataset_template

        if center_ids is not None:
            # a subset of features was used as center; slice the template
            # dataset accordingly to get sliced .fa
            center_ids_arr = np.asarray(center_ids)
            ds_template = ds_template[:, center_ids_arr]

        # make a copy of the input
        ds_copy = ds.copy()

        # apply dataset and feature attributes
        ds_copy.a.clear()
        ds_copy.a.update(ds_template.a)

        ds_copy.fa.clear()
        ds_copy.fa.update(ds_template.fa)

        return ds_copy


class CosmoSearchlight(Searchlight):
    '''
    Implement a standard Saerchlight measure, but with a separate
    postprocessing step that involves setting feature (.fa)
    and dataset (.a) attributes after the searchlight call has been made.

    A typical use case is in combination with a neighborhood from CoSMoMVPA,
    either from a .mat matlab file or through a CosmoQueryEngine
    '''

    def __init__(self, datameasure, nbrhood, add_center_fa=False,
                 results_postproc_fx=None,
                 results_backend='native',
                 results_fx=None,
                 tmp_prefix='tmpsl',
                 nblocks=None,
                 **kwargs):
        """
        Parameters
        ----------
        datameasure : callable
          Any object that takes a :class:`~mvpa2.datasets.base.Dataset`
          and returns some measure when called.
        nbrhood : str or dict or CosmoQueryEngine
          Defines the neighborhood for each feature. A str indicates the
          filename of a matlab .mat file with a neighborhood generated by
          CoSMoMVPA. A dict indicates the result of such a neighborhood
          when loaded by loadmat. A CosmoQueryEngine indicates such a
          neighborhood as a QueryEngineInterface object
        add_center_fa : bool or str
          If True or a string, each searchlight ROI dataset will have a boolean
          vector as a feature attribute that indicates the feature that is the
          seed (e.g. sphere center) for the respective ROI. If True, the
          attribute is named 'roi_seed', the provided string is used as the name
          otherwise.
        results_postproc_fx : callable
          Called with all the results computed in a block for possible
          post-processing which needs to be done in parallel instead of serial
          aggregation in results_fx.
        results_backend : ('native', 'hdf5'), optional
          Specifies the way results are provided back from a processing block
          in case of nproc > 1. 'native' is pickling/unpickling of results by
          pprocess, while 'hdf5' would use h5save/h5load functionality.
          'hdf5' might be more time and memory efficient in some cases.
        results_fx : callable, optional
          Function to process/combine results of each searchlight
          block run.  By default it would simply append them all into
          the list.  It receives as keyword arguments sl, dataset,
          roi_ids, and results (iterable of lists).  It is the one to take
          care of assigning roi_* ca's
        tmp_prefix : str, optional
          If specified -- serves as a prefix for temporary files storage
          if results_backend == 'hdf5'.  Thus can specify the directory to use
          (trailing file path separator is not added automagically).
        nblocks : None or int
          Into how many blocks to split the computation (could be larger than
          nproc).  If None -- nproc is used.
        **kwargs
          In addition this class supports all keyword arguments of its
          base-class :class:`~mvpa2.measures.searchlight.BaseSearchlight`.
        """

        queryengine = from_any(nbrhood)

        expected_type = CosmoQueryEngine
        if not isinstance(queryengine, expected_type):
            raise TypeError('Queryengine should be a %s, found type %s' %
                                    (expected_type, type(queryengine)))

        super(CosmoSearchlight, self).__init__(datameasure, queryengine,
                                               add_center_fa=add_center_fa,
                                               results_postproc_fx=results_postproc_fx,
                                               results_backend=results_backend,
                                               results_fx=results_fx,
                                               tmp_prefix=tmp_prefix,
                                               nblocks=nblocks,
                                               **kwargs)

    def _call(self, ds):
        '''
        Perform standard searchlight analysis, then update
        feature and dataset attributes using the queryengine's .fa and .a
        '''

        # let the parent class do most of the work
        ds_result = super(CosmoSearchlight, self)._call(ds)

        # set dataset (.a) and feature (.fa) attributes
        queryengine = self._queryengine
        return queryengine.set_output_dataset_attributes(ds_result)
