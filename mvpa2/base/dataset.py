# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Multi-purpose dataset container with support for attributes."""

__docformat__ = 'restructuredtext'

import numpy as np
import copy

from mvpa2.base import externals, cfg, warning
from mvpa2.base.collections import SampleAttributesCollection, \
        FeatureAttributesCollection, DatasetAttributesCollection
from mvpa2.base.types import is_datasetlike
from mvpa2.base.dochelpers import _str, _strid

if __debug__:
    from mvpa2.base import debug

__REPR_STYLE__ = cfg.get('datasets', 'repr', 'full')

if not __REPR_STYLE__ in ('full', 'str'):
    raise ValueError, "Incorrect value %r for option datasets.repr." \
          " Valid are 'full' and 'str'." % __REPR_STYLE__

class AttrDataset(object):
    """Generic storage class for datasets with multiple attributes.

    A dataset consists of four pieces.  The core is a two-dimensional
    array that has variables (so-called `features`) in its columns and
    the associated observations (so-called `samples`) in the rows.  In
    addition a dataset may have any number of attributes for features
    and samples.  Unsurprisingly, these are called 'feature attributes'
    and 'sample attributes'.  Each attribute is a vector of any datatype
    that contains a value per each item (feature or sample). Both types
    of attributes are organized in their respective collections --
    accessible via the `sa` (sample attribute) and `fa` (feature
    attribute) attributes.  Finally, a dataset itself may have any number
    of additional attributes (i.e. a mapper) that are stored in their
    own collection that is accessible via the `a` attribute (see
    examples below).

    Attributes
    ----------
    sa : Collection
      Access to all sample attributes, where each attribute is a named
      vector (1d-array) of an arbitrary datatype, with as many elements
      as rows in the `samples` array of the dataset.
    fa : Collection
      Access to all feature attributes, where each attribute is a named
      vector (1d-array) of an arbitrary datatype, with as many elements
      as columns in the `samples` array of the dataset.
    a : Collection
      Access to all dataset attributes, where each attribute is a named
      element of an arbitrary datatype.

    Notes
    -----
    Any dataset might have a mapper attached that is stored as a dataset
    attribute called `mapper`.

    Examples
    --------

    The simplest way to create a dataset is from a 2D array.

    >>> import numpy as np
    >>> from mvpa2.datasets import *
    >>> samples = np.arange(12).reshape((4,3))
    >>> ds = AttrDataset(samples)
    >>> ds.nsamples
    4
    >>> ds.nfeatures
    3
    >>> ds.samples
    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]])

    The above dataset can only be used for unsupervised machine-learning
    algorithms, since it doesn't have any targets associated with its
    samples. However, creating a labeled dataset is equally simple.

    >>> ds_labeled = dataset_wizard(samples, targets=range(4))

    Both the labeled and the unlabeled dataset share the same samples
    array. No copying is performed.

    >>> ds.samples is ds_labeled.samples
    True

    If the data should not be shared the samples array has to be copied
    beforehand.

    The targets are available from the samples attributes collection, but
    also via the convenience property `targets`.

    >>> ds_labeled.sa.targets is ds_labeled.targets
    True

    If desired, it is possible to add an arbitrary amount of additional
    attributes. Regardless if their original sequence type they will be
    converted into an array.

    >>> ds_labeled.sa['lovesme'] = [0,0,1,0]
    >>> ds_labeled.sa.lovesme
    array([0, 0, 1, 0])

    An alternative method to create datasets with arbitrary attributes
    is to provide the attribute collections to the constructor itself --
    which would also test for an appropriate size of the given
    attributes:

    >>> fancyds = AttrDataset(samples, sa={'targets': range(4),
    ...                                'lovesme': [0,0,1,0]})
    >>> fancyds.sa.lovesme
    array([0, 0, 1, 0])

    Exactly the same logic applies to feature attributes as well.

    Datasets can be sliced (selecting a subset of samples and/or
    features) similar to arrays. Selection is possible using boolean
    selection masks, index sequences or slicing arguments. The following
    calls for samples selection all result in the same dataset:

    >>> sel1 = ds[np.array([False, True, True])]
    >>> sel2 = ds[[1,2]]
    >>> sel3 = ds[1:3]
    >>> np.all(sel1.samples == sel2.samples)
    True
    >>> np.all(sel2.samples == sel3.samples)
    True

    During selection data is only copied if necessary. If the slicing
    syntax is used the resulting dataset will share the samples with the
    original dataset (here and below we compare .base against both ds.samples
    and its .base for compatibility with NumPy < 1.7)

    >>> sel1.samples.base in (ds.samples.base, ds.samples)
    False
    >>> sel2.samples.base in (ds.samples.base, ds.samples)
    False
    >>> sel3.samples.base in (ds.samples.base, ds.samples)
    True

    For feature selection the syntax is very similar they are just
    represented on the second axis of the samples array. Plain feature
    selection is achieved be keeping all samples and select a subset of
    features (all syntax variants for samples selection are also
    supported for feature selection).

    >>> fsel = ds[:, 1:3]
    >>> fsel.samples
    array([[ 1,  2],
           [ 4,  5],
           [ 7,  8],
           [10, 11]])

    It is also possible to simultaneously selection a subset of samples
    *and* features. Using the slicing syntax now copying will be
    performed.

    >>> fsel = ds[:3, 1:3]
    >>> fsel.samples
    array([[1, 2],
           [4, 5],
           [7, 8]])
    >>> fsel.samples.base in (ds.samples.base, ds.samples)
    True

    Please note that simultaneous selection of samples and features is
    *not* always congruent to array slicing.

    >>> ds[[0,1,2], [1,2]].samples
    array([[1, 2],
           [4, 5],
           [7, 8]])

    Whereas the call: 'ds.samples[[0,1,2], [1,2]]' would not be
    possible. In `AttrDatasets` selection of samples and features is always
    applied individually and independently to each axis.
    """
    def __init__(self, samples, sa=None, fa=None, a=None):
        """
        A Dataset might have an arbitrary number of attributes for samples,
        features, or the dataset as a whole. However, only the data samples
        themselves are required.

        Parameters
        ----------
        samples : ndarray
          Data samples.  This has to be a two-dimensional (samples x features)
          array. If the samples are not in that format, please consider one of
          the `AttrDataset.from_*` classmethods.
        sa : SampleAttributesCollection
          Samples attributes collection.
        fa : FeatureAttributesCollection
          Features attributes collection.
        a : DatasetAttributesCollection
          Dataset attributes collection.

        """
        # conversions
        if isinstance(samples, list):
            samples = np.array(samples)
        # Check all conditions we need to have for `samples` dtypes
        if not hasattr(samples, 'dtype'):
            raise ValueError(
                "AttrDataset only supports dtypes as samples that have a "
                "`dtype` attribute that behaves similar to the one of an "
                "array-like.")
        if not hasattr(samples, 'shape'):
            raise ValueError(
                "AttrDataset only supports dtypes as samples that have a "
                "`shape` attribute that behaves similar to the one of an "
                "array-like.")
        if not len(samples.shape):
            raise ValueError("Only `samples` with at least one axis are "
                    "supported (got: %i)" % len(samples.shape))

        # handling of 1D-samples
        # i.e. 1D is treated as multiple samples with a single feature
        if len(samples.shape) == 1:
            samples = np.atleast_2d(samples).T

        # that's all -- accepted
        self.samples = samples

        # Everything in a dataset (except for samples) is organized in
        # collections
        # Number of samples is .shape[0] for sparse matrix support
        self.sa = SampleAttributesCollection(length=len(self))
        if not sa is None:
            self.sa.update(sa)
        self.fa = FeatureAttributesCollection(length=self.nfeatures)
        if not fa is None:
            self.fa.update(fa)
        self.a = DatasetAttributesCollection()
        if not a is None:
            self.a.update(a)


    def init_origids(self, which, attr='origids', mode='new'):
        """Initialize the dataset's 'origids' attribute.

        The purpose of origids is that they allow to track the identity of
        a feature or a sample through the lifetime of a dataset (i.e. subsequent
        feature selections).

        Calling this method will overwrite any potentially existing IDs (of the
        XXX)

        Parameters
        ----------
        which : {'features', 'samples', 'both'}
          An attribute is generated for each feature, sample, or both that
          represents a unique ID.  This ID incorporates the dataset instance ID
          and should allow merging multiple datasets without causing multiple
          identical ID and the resulting dataset.
        attr : str
          Name of the attribute to store the generated IDs in.  By convention
          this should be 'origids' (the default), but might be changed for
          specific purposes.
        mode : {'existing', 'new', 'raise'}, optional
          Action if `attr` is already present in the collection.
          Default behavior is 'new' whenever new ids are generated and
          replace existing values if such are present.  With 'existing' it would
          not alter existing content.  With 'raise' it would raise
          `RuntimeError`.

        Raises
        ------
        `RuntimeError`
          If `mode` == 'raise' and `attr` is already defined
        """
        # now do evil to ensure unique ids across multiple datasets
        # so that they could be merged together
        thisid = str(id(self))
        legal_modes = ('raise', 'existing', 'new')
        if not mode in legal_modes:
            raise ValueError, "Incorrect mode %r. Known are %s." % \
                  (mode, legal_modes)
        if which in ('samples', 'both'):
            if attr in self.sa:
                if mode == 'existing':
                    return
                elif mode == 'raise':
                    raise RuntimeError, \
                          "Attribute %r already known to %s" % (attr, self.sa)
            ids = np.array(['%s-%i' % (thisid, i)
                                for i in xrange(self.samples.shape[0])])
            if self.sa.has_key(attr):
                self.sa[attr].value = ids
            else:
                self.sa[attr] = ids
        if which in ('features', 'both'):
            if attr in self.sa:
                if mode == 'existing':
                    return
                elif mode == 'raise':
                    raise RuntimeError, \
                          "Attribute %r already known to %s" % (attr, self.fa)
            ids = np.array(['%s-%i' % (thisid, i)
                                for i in xrange(self.samples.shape[1])])
            if self.fa.has_key(attr):
                self.fa[attr].value = ids
            else:
                self.fa[attr] = ids


    def __copy__(self):
        return self.copy(deep=False)


    def __deepcopy__(self, memo=None):
        return self.copy(deep=True, memo=memo)


    def __reduce__(self):
        return (self.__class__,
                    (self.samples,
                     dict(self.sa),
                     dict(self.fa),
                     dict(self.a)))


    def copy(self, deep=True, sa=None, fa=None, a=None, memo=None):
        """Create a copy of a dataset.

        By default this is going to return a deep copy of the dataset, hence no
        data would be shared between the original dataset and its copy.

        Parameters
        ----------
        deep : boolean, optional
          If False, a shallow copy of the dataset is return instead.  The copy
          contains only views of the samples, sample attributes and feature
          attributes, as well as shallow copies of all dataset
          attributes.
        sa : list or None
          List of attributes in the sample attributes collection to include in
          the copy of the dataset. If `None` all attributes are considered. If
          an empty list is given, all attributes are stripped from the copy.
        fa : list or None
          List of attributes in the feature attributes collection to include in
          the copy of the dataset. If `None` all attributes are considered If
          an empty list is given, all attributes are stripped from the copy.
        a : list or None
          List of attributes in the dataset attributes collection to include in
          the copy of the dataset. If `None` all attributes are considered If
          an empty list is given, all attributes are stripped from the copy.
        memo : dict
          Developers only: This argument is only useful if copy() is called
          inside the __deepcopy__() method and refers to the dict-argument
          `memo` in the Python documentation.
        """
        if __debug__:
            debug('DS_', "Duplicating samples shaped %s"
                         % str(self.samples.shape))
        if deep:
            samples = copy.deepcopy(self.samples, memo)
        else:
            samples = self.samples.view()

        if __debug__:
            debug('DS_', "Create new dataset instance for copy")
        # call the generic init
        out = self.__class__(samples,
                             sa=self.sa.copy(a=sa, deep=deep, memo=memo),
                             fa=self.fa.copy(a=fa, deep=deep, memo=memo),
                             a=self.a.copy(a=a, deep=deep, memo=memo))
        if __debug__:
            debug('DS_', "Return dataset copy %s of source %s"
                         % (_strid(out), _strid(self)))
        return out


    def append(self, other):
        """This method should not be used and will be removed in the future"""
        warning("AttrDataset.append() is deprecated and will be removed. "
                "Instead of ds.append(x) use: ds = vstack((ds, x), a=0)")

        if not self.nfeatures == other.nfeatures:
            raise DatasetError("Cannot merge datasets, because the number of "
                               "features does not match.")

        if not sorted(self.sa.keys()) == sorted(other.sa.keys()):
            raise DatasetError("Cannot merge dataset. This datasets samples "
                               "attributes %s cannot be mapped into the other "
                               "set %s" % (self.sa.keys(), other.sa.keys()))

        # concat the samples as well
        self.samples = np.concatenate((self.samples, other.samples), axis=0)

        # tell the collection the new desired length of all attributes
        self.sa.set_length_check(len(self.samples))
        # concat all samples attributes
        for k, v in other.sa.iteritems():
            self.sa[k].value = np.concatenate((self.sa[k].value, v.value),
                                             axis=0)


    def __getitem__(self, args):
        """
        """
        # uniformize for checks below; it is not a tuple if just single slicing
        # spec is passed
        if not isinstance(args, tuple):
            args = (args,)

        if len(args) > 2:
            raise ValueError("Too many arguments (%i). At most there can be "
                             "two arguments, one for samples selection and one "
                             "for features selection" % len(args))

        # simplify things below and always have samples and feature slicing
        if len(args) == 1:
            args = [args[0], slice(None)]
        else:
            args = [a for a in args]

        samples = None

        # get the intended subset of the samples array
        #
        # need to deal with some special cases to ensure proper behavior
        #
        # ints need to become lists to prevent silent dimensionality changes
        # of the arrays when slicing
        for i, a in enumerate(args):
            if isinstance(a, int):
                args[i] = [a]

        # for simultaneous slicing of numpy arrays we should
        # distinguish the case when one of the args is a slice, so no
        # ix_ is needed
        if __debug__:
            debug('DS_', "Selecting feature/samples of %s" % str(self.samples.shape))
        if isinstance(self.samples, np.ndarray):
            if np.any([isinstance(a, slice) for a in args]):
                samples = self.samples[args[0], args[1]]
            else:
                # works even with bool masks (although without
                # assurance/checking if mask is of actual length as
                # needed, so would work with bogus shorter
                # masks). TODO check in __debug__? or may be just do
                # enforcing of proper dimensions and order manually?
                samples = self.samples[np.ix_(*args)]
        else:
            # in all other cases we have to do the selection sequentially
            #
            # samples subset: only alter if subset is requested
            samples = self.samples[args[0]]
            # features subset
            if not args[1] is slice(None):
                samples = samples[:, args[1]]
        if __debug__:
            debug('DS_', "Selected feature/samples %s" % str(self.samples.shape))
        # and now for the attributes -- we want to maintain the type of the
        # collections
        sa = self.sa.__class__(length=samples.shape[0])
        fa = self.fa.__class__(length=samples.shape[1])
        a = self.a.__class__()

        # per-sample attributes; always needs to run even if slice(None), since
        # we need fresh SamplesAttributes even if they share the data
        for attr in self.sa.values():
            # preserve attribute type
            newattr = attr.__class__(doc=attr.__doc__)
            # slice
            newattr.value = attr.value[args[0]]
            # assign to target collection
            sa[attr.name] = newattr

        # per-feature attributes; always needs to run even if slice(None),
        # since we need fresh SamplesAttributes even if they share the data
        for attr in self.fa.values():
            # preserve attribute type
            newattr = attr.__class__(doc=attr.__doc__)
            # slice
            newattr.value = attr.value[args[1]]
            # assign to target collection
            fa[attr.name] = newattr

        # and finally dataset attributes: this time copying
        for attr in self.a.values():
            # preserve attribute type
            newattr = attr.__class__(name=attr.name, doc=attr.__doc__)
            # do a shallow copy here
            # XXX every DatasetAttribute should have meaningful __copy__ if
            # necessary -- most likely all mappers need to have one
            newattr.value = copy.copy(attr.value)
            # assign to target collection
            a[attr.name] = newattr

        # and after a long way instantiate the new dataset of the same type
        return self.__class__(samples, sa=sa, fa=fa, a=a)


    def __repr_full__(self):
        return "%s(%s, sa=%s, fa=%s, a=%s)" \
                % (self.__class__.__name__,
                   repr(self.samples),
                   repr(self.sa),
                   repr(self.fa),
                   repr(self.a))


    def __str__(self):
        samplesstr = 'x'.join(["%s" % x for x in self.shape])
        samplesstr += '@%s' % self.samples.dtype
        cols = [str(col).replace(col.__class__.__name__, label)
                    for col, label in [(self.sa, 'sa'),
                                       (self.fa, 'fa'),
                                       (self.a, 'a')] if len(col)]
        # include only collections that have content
        return _str(self, samplesstr, *cols)

    __repr__ = {'full' : __repr_full__,
                'str'  : __str__}[__REPR_STYLE__]

    def __array__(self, *args):
        """Provide an 'array' view or copy over dataset.samples

        Parameters
        ----------
        dtype: type, optional
          If provided, passed to .samples.__array__() call

        *args to mimique numpy.ndarray.__array__ behavior which relies
        on the actual number of arguments
        """
        # another possibility would be converting .todense() for sparse data
        # but that might easily kill the machine ;-)
        if not hasattr(self.samples, '__array__'):
            raise RuntimeError(
                "This AttrDataset instance cannot be used like a Numpy array "
                "since its data-container does not provide an '__array__' "
                "methods. Container type is %s." % type(self.samples))
        return self.samples.__array__(*args)


    def __len__(self):
        return self.shape[0]


    @classmethod
    def from_hdf5(cls, source, name=None):
        """Load a Dataset from HDF5 file

        Parameters
        ----------
        source : string or h5py.highlevel.File
          Filename or HDF5's File to load dataset from
        name : string, optional
          If file contains multiple entries at the 1st level, if
          provided, `name` specifies the group to be loaded as the
          AttrDataset.

        Returns
        -------
        AttrDataset

        Raises
        ------
        ValueError
        """
        if not externals.exists('h5py'):
            raise RuntimeError(
                "Missing 'h5py' package -- saving is not possible.")

        import h5py
        from mvpa2.base.hdf5 import hdf2obj

        # look if we got an hdf file instance already
        if isinstance(source, h5py.highlevel.File):
            own_file = False
            hdf = source
        else:
            own_file = True
            hdf = h5py.File(source, 'r')

        if not name is None:
            # some HDF5 subset is requested
            if not name in hdf:
                raise ValueError("Cannot find '%s' group in HDF file %s.  "
                                 "File contains groups: %s"
                                 % (name, source, hdf.keys()))

            # access the group that should contain the dataset
            dsgrp = hdf[name]
            res = hdf2obj(dsgrp)
            if not isinstance(res, AttrDataset):
                # TODO: unittest before committing
                raise ValueError, "%r in %s contains %s not a dataset.  " \
                      "File contains groups: %s." \
                      % (name, source, type(res), hdf.keys())
        else:
            # just consider the whole file
            res = hdf2obj(hdf)
            if not isinstance(res, AttrDataset):
                # TODO: unittest before committing
                raise ValueError, "Failed to load a dataset from %s.  " \
                      "Loaded %s instead." \
                      % (source, type(res))
        if own_file:
            hdf.close()
        return res


    # shortcut properties
    nsamples = property(fget=len)
    nfeatures = property(fget=lambda self:self.shape[1])
    shape = property(fget=lambda self:self.samples.shape)


def datasetmethod(func):
    """Decorator to easily bind functions to an AttrDataset class
    """
    if __debug__:
        debug("DS_",
              "Binding function %s to AttrDataset class" % func.func_name)

    # Bind the function
    setattr(AttrDataset, func.func_name, func)

    # return the original one
    return func


def vstack(datasets, a=None):
    """Stacks datasets vertically (appending samples).

    Feature attribute collections are merged incrementally, attribute with
    identical keys overwriting previous ones in the stacked dataset. All
    datasets must have an identical set of sample attributes (matching keys,
    not values), otherwise a ValueError will be raised.
    No dataset attributes from any source dataset will be transferred into the
    stacked dataset. If all input dataset have common dataset attributes that
    are also valid for the stacked dataset, they can be moved into the output
    dataset like this::

      ds_merged = vstack((ds1, ds2, ds3))
      ds_merged.a.update(ds1.a)

    Parameters
    ----------
    datasets : tuple
        Sequence of datasets to be stacked.
    a: {'unique','drop_nonunique','uniques','all'} or True or False or None (default: None)
        Indicates which dataset attributes from datasets are stored
        in merged_dataset. If an int k, then the dataset attributes from
        datasets[k] are taken. If 'unique' then it is assumed that any
        attribute common to more than one dataset in datasets is unique;
        if not an exception is raised. If 'drop_nonunique' then as 'unique',
        except that exceptions are not raised. If 'uniques' then, for each
        attribute,  any unique value across the datasets is stored in a tuple
        in merged_datasets. If 'all' then each attribute present in any
        dataset across datasets is stored as a tuple in merged_datasets;
        missing values are replaced by None. If None (the default) then no
        attributes are stored in merged_dataset. True is equivalent to
        'drop_nonunique'. False is equivalent to None.

    Returns
    -------
    AttrDataset (or respective subclass)
    """
    if not len(datasets):
        raise ValueError('concatenation of zero-length sequences is impossible')
    if not len(datasets) > 1:
        # trivial vstack
        return datasets[0]
    # fall back to numpy if it is not a dataset
    if not is_datasetlike(datasets[0]):
        return AttrDataset(np.vstack(datasets))

    if __debug__:
        target = sorted(datasets[0].sa.keys())
        if not np.all([sorted(ds.sa.keys()) == target for ds in datasets]):
            raise ValueError("Sample attributes collections of to be stacked "
                             "datasets have varying attributes.")
    # will puke if not equal number of features
    stacked_samp = np.concatenate([ds.samples for ds in datasets], axis=0)

    stacked_sa = {}
    for attr in datasets[0].sa:
        stacked_sa[attr] = np.concatenate(
            [ds.sa[attr].value for ds in datasets], axis=0)
    # create the dataset
    merged = datasets[0].__class__(stacked_samp, sa=stacked_sa)

    for ds in datasets:
        merged.fa.update(ds.fa)

    _stack_add_equal_dataset_attributes(merged, datasets, a)
    return merged


def hstack(datasets, a=None):
    """Stacks datasets horizontally (appending features).

    Sample attribute collections are merged incrementally, attribute with
    identical keys overwriting previous ones in the stacked dataset. All
    datasets must have an identical set of feature attributes (matching keys,
    not values), otherwise a ValueError will be raised.
    No dataset attributes from any source dataset will be transferred into the
    stacked dataset.

    Parameters
    ----------
    datasets : tuple
        Sequence of datasets to be stacked.
    a: {'unique','drop_nonunique','uniques','all'} or True or False or None (default: None)
        Indicates which dataset attributes from datasets are stored
        in merged_dataset. If an int k, then the dataset attributes from
        datasets[k] are taken. If 'unique' then it is assumed that any
        attribute common to more than one dataset in datasets is unique;
        if not an exception is raised. If 'drop_nonunique' then as 'unique',
        except that exceptions are not raised. If 'uniques' then, for each
        attribute,  any unique value across the datasets is stored in a tuple
        in merged_datasets. If 'all' then each attribute present in any
        dataset across datasets is stored as a tuple in merged_datasets;
        missing values are replaced by None. If None (the default) then no
        attributes are stored in merged_dataset. True is equivalent to
        'drop_nonunique'. False is equivalent to None.

    Returns
    -------
    AttrDataset (or respective subclass)
    """
    #
    # XXX Use CombinedMapper in here whenever it comes back
    #

    if not len(datasets):
        raise ValueError('concatenation of zero-length sequences is impossible')
    if not len(datasets) > 1:
        # trivial hstack
        return datasets[0]
    # fall back to numpy if it is not a dataset
    if not is_datasetlike(datasets[0]):
        # we might get a list of 1Ds that would yield wrong results when
        # turned into a dict (would run along samples-axis)
        return AttrDataset(np.atleast_2d(np.hstack(datasets)))

    if __debug__:
        target = sorted(datasets[0].fa.keys())
        if not np.all([sorted(ds.fa.keys()) == target for ds in datasets]):
            raise ValueError("Feature attributes collections of to be stacked "
                             "datasets have varying attributes.")
    # will puke if not equal number of samples
    stacked_samp = np.concatenate([ds.samples for ds in datasets], axis=1)

    stacked_fa = {}
    for attr in datasets[0].fa:
        stacked_fa[attr] = np.concatenate(
            [ds.fa[attr].value for ds in datasets], axis=0)
    # create the dataset
    merged = datasets[0].__class__(stacked_samp, fa=stacked_fa)

    for ds in datasets:
        merged.sa.update(ds.sa)

    _stack_add_equal_dataset_attributes(merged, datasets, a)

    return merged


def all_equal(x, y):
    '''General function that compares two values. Usually this function
    behaves like x==y and type(x)==type(y), but for numpy arrays it
    behaves like np.array_equal(x==y).

    Parameters
    ----------
    x, y : any type
        Elements to be compared

    Returns
    -------
    eq: bool
        True iff x and y are equal. If in the comparison of x and y
        and exception is thrown then False is returned
        This comparison is performed element-wise, if applicable, and
        in that case True is only returned if all elements are equal
    '''

    # an equality comparison that also works on numpy arrays
    try:
        eq = x == y
    except:
        # we can get here, in case of dictionaries with non-simple values
        # (e.g. arrays)
        try:
            if not set(x.keys()) == set(y.keys()):
                return False
            keys = list(x.keys())
            x = [x[k] for k in keys]
            y = [y[k] for k in keys]
            # this is just for fooling the next test
            eq = [0, 1]
        except:
            # cannot do more than try
            return False

    # eq could be a numpy array or similar. See if it has a length
    try:
        len(eq) # that's fine, so we can zip x and y (below)
                # and compare by elements
    except TypeError:
        # if it's just a bool (or boolean-like, such as numpy.bool_)
        # then see if it is True or not
        if eq == True or eq == False:
            # also consider the case that eq is a numpy boolean array
            # with just a single element - so compare to True
            return eq == True
        else:
            # no idea what to do
            raise

    # because of numpy's broadcasting either x or y may
    # be a scaler yet eq could be an array
    try:
        same_length = len(x) == len(y)
        if not same_length:
            return False
    except:
        return False

    # do a recursive call on all elements
    return all(all_equal(xx, yy) for (xx, yy) in zip(x, y))

def _stack_add_equal_dataset_attributes(merged_dataset, datasets, a=None):
    """Helper function for vstack and hstack to find dataset
    attributes common to a set of datasets, and at them to the output.
    Note:by default this function does nothing because testing for equality
    may be messy for certain types; to override a value should be assigned
    to the add_keys argument.

    Parameters
    ----------
    merged_dataset: Dataset
        the output dataset to which attributes are added
    datasets: tuple of Dataset
        Sequence of datasets to be stacked. Only attributes present
        in all datasets and with identical values are put in
        merged_dataset
    a: {'unique','drop_nonunique','uniques','all'} or True or False or None (default: None).
        Indicates which dataset attributes from datasets are stored
        in merged_dataset. If an int k, then the dataset attributes from
        datasets[k] are taken. If 'unique' then it is assumed that any
        attribute common to more than one dataset in datasets is unique;
        if not an exception is raised. If 'drop_nonunique' then as 'unique',
        except that exceptions are not raised. If 'uniques' then, for each
        attribute,  any unique value across the datasets is stored in a tuple
        in merged_datasets. If 'all' then each attribute present in any
        dataset across datasets is stored as a tuple in merged_datasets;
        missing values are replaced by None. If None (the default) then no
        attributes are stored in merged_dataset. True is equivalent to
        'drop_nonunique'. False is equivalent to None.
    """
    if a is None or a is False:
        # do nothing
        return
    elif a is True:
        a = 'drop_nonunique'

    if not datasets:
        # empty - so nothing to do
        return

    if type(a) is int:
        base_dataset = datasets[a]

        for key in base_dataset.a.keys():
            merged_dataset.a[key] = base_dataset.a[key].value

        return

    allowed_values = ['unique', 'uniques', 'drop_nonunique', 'all']
    if not a in allowed_values:
        raise ValueError("a should be an int or one of "
                        "%r" % allowed_values)

    # consider all keys that are present in at least one dataset
    all_keys = set.union(*[set(dataset.a.keys()) for dataset in datasets])


    def _contains(xs, y, comparator=all_equal):
        for x in xs:
            if comparator(x, y):
                return True
        return False

    for key in all_keys:
        add_key = True
        values = []
        for i, dataset in enumerate(datasets):
            if not key in dataset.a:
                if a == 'all':
                    values.append(None)
                continue

            value = dataset.a[key].value

            if a in ('drop_nonunique', 'unique'):
                if not values:
                    values.append(value)
                elif not _contains(values, value):
                    if a == 'unique':
                        raise DatasetError("Not unique dataset attribute value "
                                         " for %s: %s and %s" %
                                            (key, values[0], value))
                    else:
                        add_key = False
                        break
            elif a == 'uniques':
                if not _contains(values, value):
                    values.append(value)
            elif a == 'all':
                values.append(value)
            else:
                raise ValueError("this should not happen: %s" % a)

        if add_key:
            if a in ('drop_nonunique', 'unique'):
                merged_dataset.a[key] = values[0]
            else:
                merged_dataset.a[key] = tuple(values)


def _expand_attribute(attr, length, attr_name):
    """Helper function to expand attributes to a desired length.

    If e.g. a sample attribute is given as a scalar expand/repeat it to a
    length matching the number of samples in the dataset.
    """
    try:
        # if we are initializing with a single string -- we should
        # treat it as a single label
        if isinstance(attr, basestring):
            raise TypeError
        if len(attr) != length:
            raise ValueError("Length of attribute '%s' [%d] has to be %d."
                             % (attr_name, len(attr), length))
        # sequence as array
        return np.asanyarray(attr)

    except TypeError:
        # make sequence of identical value matching the desired length
        return np.repeat(attr, length)

def stack_by_unique_sample_attribute(dataset, sa_label):
    """Performs hstack based on unique values in sa_label

    Parameters
    ----------
    dataset: Dataset
        input dataset.
    sa_label: str
        sample attribute label according which samples in dataset
        are stacked.

    Returns
    -------
    stacked_dataset: Dataset
        A dataset where matching features are joined (hstacked).
        If the number of matching features differs for values in sa_label
        and exception is raised.
    """

    unq, masks = _get_unique_attribute_masks(dataset.sa[sa_label].value)

    ds = []
    for i, mask in enumerate(masks):
        d = dataset[mask, :]
        d.fa[sa_label] = [unq[i]] * d.nfeatures
        ds.append(d)

    stacked_ds = hstack(ds, True)
    stacked_ds.sa.pop(sa_label)

    return stacked_ds


def stack_by_unique_feature_attribute(dataset, fa_label):
    """Performs vstack based on unique values in fa_label

    Parameters
    ----------
    dataset: Dataset
        input dataset.
    fa_label: str
        feature attribute label according which samples in dataset
        are stacked.

    Returns
    stacked_dataset: Dataset
        A dataset where matching samples are joined. This dataset has
        a sample attribute fa_label added and the feature attribute
        fa_label removed.
        If the number of matching features differs for values in sa_label
        and exception is raised.
    """

    unq, masks = _get_unique_attribute_masks(dataset.fa[fa_label].value)

    ds = []
    for i, mask in enumerate(masks):
        d = dataset[:, mask]
        d.sa[fa_label] = [unq[i]] * d.nsamples
        ds.append(d)

    stacked_ds = vstack(ds, True)
    stacked_ds.fa.pop(fa_label)

    return stacked_ds


def _get_unique_attribute_masks(xs, raise_unequal_count=True):
    '''Helper function to get masks for each unique value'''
    unq = np.unique(xs)
    masks = [x == xs for x in unq]

    if raise_unequal_count:
        hs = [np.sum(mask) for mask in masks]

        for i, h in enumerate(hs):
            if i == 0:
                h0 = h
            elif h != h0:
                raise ValueError('Value mismatch between input 0 and %d:'
                                 ' %s != %s' % (i, h, h0))
    return unq, masks

def split_by_sample_attribute(ds, sa_label, raise_unequal_count=True):
    '''Splits a dataset based on unique values of a sample attribute

    Parameters
    ----------
    d: Dataset
        input dataset
    sa_label: str or list of str
        sample attribute label(s) on which the split is based

    Returns
    -------
    ds: list of Dataset
        List with n datasets, if d.sa[sa_label] has n unique values
    '''
    if type(sa_label) in (list, tuple):
        label0 = sa_label[0]
        sas = split_by_sample_attribute(ds, label0, raise_unequal_count)
        if len(sa_label) == 1:
            return sas
        else:
            return sum([split_by_sample_attribute(sa, sa_label[1:],
                                                  raise_unequal_count)
                                for sa in sas], [])

    _, masks = _get_unique_attribute_masks(ds.sa[sa_label].value,
                                    raise_unequal_count=raise_unequal_count)

    return [ds[mask, :].copy(deep=False) for mask in masks]


def split_by_feature_attribute(ds, fa_label, raise_unequal_count=True):
    '''Splits a dataset based on unique values of a feature attribute

    Parameters
    ----------
    d: Dataset
        input dataset
    sa_label: str or list of str
        sample attribute label(s) on which the split is based

    Returns
    -------
    ds: list of Dataset
        List with n datasets, if d.fa[fa_label] has n unique values
    '''
    if type(fa_label) in (list, tuple):
        label0 = fa_label[0]
        fas = split_by_feature_attribute(ds, label0, raise_unequal_count)
        if len(fa_label) == 1:
            return fas
        else:
            return sum([split_by_feature_attribute(fa, fa_label[1:],
                                                   raise_unequal_count)
                                for fa in fas], [])

    _, masks = _get_unique_attribute_masks(ds.fa[fa_label].value,
                                    raise_unequal_count=raise_unequal_count)

    return [ds[:, mask].copy(deep=False) for mask in masks]



class DatasetError(Exception):
    """Thrown if there is a problem with the internal integrity of a Dataset.
    """
    # A ValueError exception is too generic to be used for any needed case,
    # thus this one is created
    pass


class DatasetAttributeExtractor(object):
    """Helper to extract arbitrary attributes from dataset collections.

    Examples
    --------
    >>> ds = AttrDataset(np.arange(12).reshape((4,3)),
    ...              sa={'targets': range(4)},
    ...              fa={'foo': [0,0,1]})
    >>> ext = DAE('sa', 'targets')
    >>> ext(ds)
    array([0, 1, 2, 3])

    >>> ext = DAE('fa', 'foo')
    >>> ext(ds)
    array([0, 0, 1])
    """
    def __init__(self, col, key):
        """
        Parameters
        ----------
        col : {'sa', 'fa', 'a'}
          The respective collection to extract an attribute from.
        key : arbitrary
          The name/key of the attribute in the collection.
        """
        self._col = col
        self._key = key

    def __call__(self, ds):
        """
        Parameters
        ----------
        ds : AttrDataset
        """
        return ds.__dict__[self._col][self._key].value

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               repr(self._col), repr(self._key))


# shortcut that allows for more finger/screen-friendly specification of
# attribute extraction
DAE = DatasetAttributeExtractor


@datasetmethod
def save(dataset, destination, name=None, compression=None):
    """Save Dataset into HDF5 file

    Parameters
    ----------
    dataset : `Dataset`
    destination : `h5py.highlevel.File` or str
    name : str, optional
    compression : None or int or {'gzip', 'szip', 'lzf'}, optional
      Level of compression for gzip, or another compression strategy.
    """
    if not externals.exists('h5py'):
        raise RuntimeError("Missing 'h5py' package -- saving is not possible.")

    import h5py
    from mvpa2.base.hdf5 import obj2hdf

    # look if we got an hdf file instance already
    if isinstance(destination, h5py.highlevel.File):
        own_file = False
        hdf = destination
    else:
        own_file = True
        hdf = h5py.File(destination, 'w')

    obj2hdf(hdf, dataset, name, compression=compression)

    # if we opened the file ourselves we close it now
    if own_file:
        hdf.close()
    return
