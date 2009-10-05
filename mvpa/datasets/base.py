# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset container"""

__docformat__ = 'restructuredtext'

import numpy as N

import mvpa.support.copy as copy
from mvpa.misc.state import ClassWithCollections, Collection
from mvpa.misc.attributes import SampleAttribute, FeatureAttribute, \
        DatasetAttribute
from mvpa.misc.exceptions import DatasetError
from mvpa.mappers.mask import MaskMapper

if __debug__:
    from mvpa.base import debug

# XXX This is the place to redo the Dataset base class in a more powerful, yet
# simpler way. The basic goal is to allow for all kinds of attributes:
#
# 1) Samples attributes (per-sample full)
# 2) Features attributes (per-feature stuff)
#
# 3) Dataset attributes (per-dataset stuff)
#
# Use cases:
#
#     1) labels and chunks -- goal: it should be possible to have multivariate
#     labels, e.g. to specify targets for a neural network output layer
#
#     2) feature binding/grouping -- goal: easily define ROIs in datasets, or
#     group/mark various types of feature so they could be selected or
#     discarded all together
#
#     3) Mappers, or chains of them (this should be possible already, but could
#     be better integrated to make applyMapper() obsolete).
#
#
# Perform distortion correction on __init__(). The copy contructor
# implementation should move into a separate classmethod.
#
# Think about implementing the current 'clever' attributes in terms of one-time
# properties as suggested by Fernando on nipy-devel.

# ...


# Remaining public interface of Dataset
class Dataset(ClassWithCollections):
    """The successor of Dataset.

    Conventions
    -----------
    Any dataset might have a mapper attached that is stored as a dataset
    attribute called `mapper`.

    """
    # placeholder for all three basic collections of a Dataset
    # put here to be able to check whether the AttributesCollector already
    # instanciated a particular collection
    # XXX maybe it should not do this at all for Dataset
    sa = None
    fa = None
    a = None

    # storage of samples in a plain NumPy array for fast access
    samples = None

    def __init__(self, samples, sa=None, fa=None, a=None):
        """
        A Dataset might have an arbitrary number of attributes for samples,
        features, or the dataset as a whole. However, only the data samples
        themselves are required.

        Parameters
        ----------
        samples : ndarray
          Data samples. This has to be a two-dimensional (samples x features)
          array. If the samples are not in that format, please consider one of
          the `Dataset.from_*` classmethods.
        sa : Collection
          Samples attributes collection.
        fa : Collection
          Features attributes collection.
        a : Collection
          Dataset attributes collection.

        Examples
        --------

        The simplest way to create a dataset is from a 2D array, the so-called
        :term:`samples matrix`:

        >>> import numpy as N
        >>> from mvpa.datasets import Dataset
        >>> samples = N.arange(12).reshape((4,3))
        >>> ds = Dataset(samples)
        >>> ds.nsamples
        4
        >>> ds.nfeatures
        3
        >>> ds.samples
        array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 6,  7,  8],
               [ 9, 10, 11]])
        """
        # init base class
        ClassWithCollections.__init__(self)

        # Everything in a dataset (except for samples) is organized in
        # collections
        # make sure we have the target collection
        # XXX maybe use different classes for the collections
        # but currently no reason to do so
        if self.sa is None:
            self.sa = Collection(owner=self)
        if self.fa is None:
            self.fa = Collection(owner=self)
        if self.a is None:
            self.a = Collection(owner=self)
        # copy attributes from source collections (scol) into target
        # collections (tcol)
        for scol, tcol in ((sa, self.sa),
                           (fa, self.fa),
                           (a, self.a)):
            # transfer the attributes
            if not scol is None:
                for name, attr in scol.items.iteritems():
                    # this will also update the owner of the attribute
                    tcol.add(attr)

        # sanity checks
        if not len(samples.shape) == 2:
            raise DatasetError('The samples array must be 2D or mappable into'
                               ' 2D (current shape is: %s)'
                               % str(samples.shape))
        self.samples = samples

        # XXX should we make them conditional?
        # samples attributes
        # this should rather be self.sa.iteritems(), but there is none yet
        for attr in self.sa.names:
            if not len(self.sa.getvalue(attr)) == self.nsamples:
                raise DatasetError("Length of samples attribute '%s' (%i) "
                                   "doesn't match the number of samples (%i)"
                                   % (attr,
                                      len(self.sa.getvalue(attr)),
                                      self.nsamples))
        # feature attributes
        for attr in self.fa.names:
            if not len(self.fa.getvalue(attr)) == self.nfeatures:
                raise DatasetError("Length of feature attribute '%s' (%i) "
                                   "doesn't match the number of features (%i)"
                                   % (attr,
                                      len(self.fa.getvalue(attr).attr),
                                      self.nfeatures))


    def __copy__(self):
        # first we create new collections of the right type for each of the
        # three essential collections of a dataset
        sa = self.sa.__class__()
        fa = self.fa.__class__()
        a = self.a.__class__()

        # for all the pieces that are known to be arrays
        for tcol, scol in ((sa, self.sa),
                           (fa, self.fa)):
            for attr in scol.items.values():
                # preserve attribute type
                newattr = attr.__class__(name=attr.name)
                # just get a view of the old data!
                newattr.value = attr.value.view()
                # assign to target collection
                tcol.add(newattr)

        # dataset attributes might be anythings, so they are SHALLOW copied
        # individually
        for attr in self.a.items.values():
            # preserve attribute type
            newattr = attr.__class__(name=attr.name)
            # SHALLOW copy!
            newattr.value = copy.copy(attr.value)
            # assign to target collection
            a.add(newattr)

        # and finally the samples
        samples = self.samples.view()

        # call the generic init
        out = self.__class__(samples, sa=sa, fa=fa, a=a)
        return out


    def __deepcopy__(self, memo=None):
        # first we create new collections of the right type for each of the
        # three essential collections of a dataset
        sa = self.sa.__class__()
        fa = self.fa.__class__()
        a = self.a.__class__()

        # now we copy the attributes
        for tcol, scol in ((sa, self.sa),
                           (fa, self.fa),
                           (a, self.a)):
            for name, attr in scol.items.iteritems():
                tcol.add(copy.deepcopy(attr, memo))

        # and finally the samples
        samples = copy.deepcopy(self.samples, memo)

        # call the generic init
        out = self.__class__(samples, sa=sa, fa=fa, a=a)
        return out


    def copy(self, deep=True):
        """Create a copy of a dataset.

        By default this is going to be a deep copy of the dataset, hence no
        data is shared between the original dataset and its copy.

        Parameter
        ---------
        deep : boolean
          If False, a shallow copy of the dataset is return instead. The copy
          contains only views of the samples, sample attributes and feature
          feature attributes, as well as shallow copies of all dataset
          attributes.
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)


    def __iadd__(self, other):
        """Merge the samples of one Dataset object to another (in-place).

        Note
        ----
        No dataset attributes, or feature attributes will be merged! These
        respective properties of the *other* dataset are neither checked for
        compatibility nor copied over to this dataset.
        """
        if not self.nfeatures == other.nfeatures:
            raise DatasetError("Cannot merge datasets, because the number of "
                               "features does not match.")

        if not sorted(self.sa.names) == sorted(other.sa.names):
            raise DatasetError("Cannot merge dataset. This datasets samples "
                               "attributes %s cannot be mapped into the other "
                               "set %s" % (self.sa.names, other.sa.names))
        # concat all samples attributes
        for k, v in other.sa.items.iteritems():
            self.sa[k].value = N.concatenate((self.sa[k].value, v.value), axis=0)

        # concat the samples as well
        self.samples = N.concatenate((self.samples, other.samples), axis=0)

        return self


    def __add__(self, other):
        """Merge the samples two datasets.
        """
        # shallow copies should be sufficient, since __iadd__ will concatenate
        # most pieces anyway
        merged = self.copy(deep=False)
        merged += other
        return merged


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

        # if we get an slicing array for feature selection and it is *not* 1D
        # try feeding it through the mapper (if there is any)
        if isinstance(args[1], N.ndarray) and len(args[1].shape) > 1 and \
                self.a.isKnown('mapper') and self.a.isSet('mapper'):
                    args[1] = self.a.mapper.forward(args[1])

        # simultaneous slicing of numpy arrays only yields intended results
        # if at least one of the slicing args is an actual slice and not
        # and index list are selection mask vector
        if N.any([isinstance(a, slice) for a in args]):
            samples = self.samples[args[0], args[1]]
        else:
            # in all other cases we have to do the selection sequentially
            #
            # samples subset: only alter if subset is requested
            samples = self.samples[args[0]]
            # features subset
            if not args[1] is slice(None):
                samples = samples[:, args[1]]

        # and now for the attributes -- we want to maintain the type of the
        # collections
        sa = self.sa.__class__()
        fa = self.fa.__class__()
        a = self.a.__class__()

        # per-sample attributes; always needs to run even if slice(None), since
        # we need fresh SamplesAttributes even if they share the data
        for attr in self.sa.items.values():
            # preserve attribute type
            newattr = attr.__class__(name=attr.name)
            # slice
            newattr.value = attr.value[args[0]]
            # assign to target collection
            sa.add(newattr)

        # per-feature attributes; always needs to run even if slice(None),
        # since we need fresh SamplesAttributes even if they share the data
        for attr in self.fa.items.values():
            # preserve attribute type
            newattr = attr.__class__(name=attr.name)
            # slice
            newattr.value = attr.value[args[1]]
            # assign to target collection
            fa.add(newattr)

            # and finally dataset attributes: this time copying
        for attr in self.a.items.values():
            # preserve attribute type
            newattr = attr.__class__(name=attr.name)
            # do a shallow copy here
            # XXX every DatasetAttribute should have meaningful __copy__ if
            # necessary -- most likely all mappers need to have one
            newattr.value = copy.copy(attr.value)
            # assign to target collection
            a.add(newattr)

        # and adjusting the mapper (if any)
        if a.isKnown('mapper') and a.isSet('mapper'):
            a['mapper'].value.selectOut(args[1])

        # and after a long way instantiate the new dataset of the same type
        return self.__class__(samples, sa=sa, fa=fa, a=a)


    @classmethod
    def from_basic(cls, samples, labels=None, chunks=None, mapper=None):
        """Create a Dataset from samples and elementary attributes.

        Parameters
        ----------
        samples : ndarray
          The two-dimensional samples matrix.
        labels : ndarray
        chunks : ndarray
        mapper : Mapper instance
          A (potentially trained) mapper instance that is used to forward-map
          the samples upon construction of the dataset. The mapper must
          have a simple feature space (samples x features) as output. Use
          chained mappers to achieve that, if necessary.

        Returns
        -------
        An instance of a dataset

        Notes
        -----
        blah blah

        See Also
        --------
        blah blah

        Examples
        --------
        blah blah
        """
        # put mapper as a dataset attribute in the general attributes collection
        # need to do that first, since we want to know the final
        # #samples/#features
        a = None
        if not mapper is None:
            # forward-map the samples
            samples = mapper.forward(samples)
            # and store the mapper
            mapper_ = DatasetAttribute(name='mapper')
            mapper_.value = mapper
            a = Collection(items={'mapper': mapper_})

       # compile the necessary samples attributes collection
        sa_items = {}

        if not labels is None:
            labels_ = SampleAttribute(name='labels')
            labels_.value = _expand_attribute(labels,
                                              samples.shape[0],
                                              'labels')
            # feels strange that one has to give the name again
            # XXX why does items have to be a dict when each samples
            # attr already knows its name
            sa_items['labels'] = labels_

        if not chunks is None:
            # unlike previous implementation, we do not do magic to do chunks
            # if there are none, there are none
            chunks_ = SampleAttribute(name='chunks')
            chunks_.value = _expand_attribute(chunks,
                                              samples.shape[0],
                                              'chunks')
            sa_items['chunks'] = chunks_

        # the final collection for samples attributes
        # XXX advantages of using SamplesAttributeCollection?
        sa = Collection(items=sa_items)

        # common checks should go into __init__
        return cls(samples, sa=sa, a=a)


    @classmethod
    def from_masked(cls, samples, labels=None, chunks=None, mask=None):
        """
        """
        # need to have arrays
        samples = N.asanyarray(samples)

        # use full mask if none is provided
        if mask is None:
            mask = N.ones(samples.shape[1:], dtype='bool')

        mapper = MaskMapper(mask)
        return cls.from_basic(samples, labels=labels, chunks=chunks,
                              mapper=mapper)


    # shortcut properties
    S = property(fget=lambda self:self.samples)
    nsamples = property(fget=lambda self:self.samples.shape[0])
    nfeatures = property(fget=lambda self:self.samples.shape[1])
    labels = property(fget=lambda self:self.sa.labels)
    L = labels
    UL = property(fget=lambda self:self.sa['labels'].unique)
    chunks = property(fget=lambda self:self.sa.chunks)
    C = chunks
    UC = property(fget=lambda self:self.sa['chunks'].unique)
    mapper = property(fget=lambda self:self.a.mapper)
    O = property(fget=lambda self:self.a.mapper.reverse(self.samples))



# convenience alias
dataset = Dataset.from_basic

def datasetmethod(func):
    """Decorator to easily bind functions to a Dataset class
    """
    if __debug__:
        debug("DS_",  "Binding function %s to Dataset class" % func.func_name)

    # Bind the function
    setattr(Dataset, func.func_name, func)

    # return the original one
    return func


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
            raise DatasetError, \
                  "Length of attribute '%s' [%d] has to be %d." \
                  % (attr_name, len(attr), length) \
        # sequence as array
        return N.asanyarray(attr)

    except TypeError:
        # make sequence of identical value matching the desired length
        return N.repeat(attr, length)


#    def index(self, *args, **kwargs):
#        pass
#
#
#    def select(self, *args, **kwargs):
#        pass
#
#
#    def where(self, *args, **kwargs):
#        pass




#OLD:# import operator
#OLD:# import random
#OLD:# import mvpa.support.copy as copy
#OLD:# 
#OLD:# from sets import Set
#OLD:# 
#OLD:# # Sooner or later Dataset would become ClassWithCollections as well, but for
#OLD:# # now just an object -- thus commenting out tentative changes
#OLD:# #
#OLD:# #XXX from mvpa.misc.state import ClassWithCollections, SampleAttribute
#OLD:# 
#OLD:# from mvpa.misc.exceptions import DatasetError
#OLD:# from mvpa.misc.support import idhash as idhash_
#OLD:# from mvpa.base.dochelpers import enhancedDocString, table2string
#OLD:# 
#OLD:# from mvpa.base import warning
#OLD:# 
#OLD:# if __debug__:
#OLD:#     from mvpa.base import debug
#OLD:# 
#OLD:#     def _validate_indexes_uniq_sorted(seq, fname, item):
#OLD:#         """Helper function to validate that seq contains unique sorted values
#OLD:#         """
#OLD:#         if operator.isSequenceType(seq):
#OLD:#             seq_unique = N.unique(seq)
#OLD:#             if len(seq) != len(seq_unique):
#OLD:#                 warning("%s() operates only with indexes for %s without"
#OLD:#                         " repetitions. Repetitions were removed."
#OLD:#                         % (fname, item))
#OLD:#             if N.any(N.sort(seq) != seq_unique):
#OLD:#                 warning("%s() does not guarantee the original order"
#OLD:#                         " of selected %ss. Use selectSamples() and "
#OLD:#                         " selectFeatures(sort=False) instead" % (fname, item))
#OLD:# 
#OLD:# 
#OLD:# #XXX class Dataset(ClassWithCollections):
#OLD:# class Dataset(object):
#OLD:#     """*The* Dataset.
#OLD:# 
#OLD:#     This class provides a container to store all necessary data to
#OLD:#     perform MVPA analyses. These are the data samples, as well as the
#OLD:#     labels associated with the samples. Additionally, samples can be
#OLD:#     grouped into chunks.
#OLD:# 
#OLD:#     :Groups:
#OLD:#       - `Creators`: `__init__`, `selectFeatures`, `selectSamples`,
#OLD:#         `applyMapper`
#OLD:#       - `Mutators`: `permuteLabels`
#OLD:# 
#OLD:#     Important: labels assumed to be immutable, i.e. no one should modify
#OLD:#     them externally by accessing indexed items, ie something like
#OLD:#     ``dataset.labels[1] += 100`` should not be used. If a label has
#OLD:#     to be modified, full copy of labels should be obtained, operated on,
#OLD:#     and assigned back to the dataset, otherwise dataset.uniquelabels
#OLD:#     would not work.  The same applies to any other attribute which has
#OLD:#     corresponding unique* access property.
#OLD:# 
#OLD:#     """
#OLD:#     # XXX Notes about migration to use Collections to store data and
#OLD:#     # attributes for samples, features, and dataset itself:
#OLD:# 
#OLD:#     # changes:
#OLD:#     #   _data  ->  s_attr collection (samples attributes)
#OLD:#     #   _dsattr -> ds_attr collection
#OLD:#     #              f_attr collection (features attributes)
#OLD:# 
#OLD:#     # static definition to track which unique attributes
#OLD:#     # have to be reset/recomputed whenever anything relevant
#OLD:#     # changes
#OLD:# 
#OLD:#     # unique{labels,chunks} become a part of dsattr
#OLD:#     _uniqueattributes = []
#OLD:#     """Unique attributes associated with the data"""
#OLD:# 
#OLD:#     _registeredattributes = []
#OLD:#     """Registered attributes (stored in _data)"""
#OLD:# 
#OLD:#     _requiredattributes = ['samples', 'labels']
#OLD:#     """Attributes which have to be provided to __init__, or otherwise
#OLD:#     no default values would be assumed and construction of the
#OLD:#     instance would fail"""
#OLD:# 
#OLD:#     #XXX _ATTRIBUTE_COLLECTIONS = [ 's_attr', 'f_attr', 'ds_attr' ]
#OLD:#     #XXX """Assure those 3 collections to be present in all datasets"""
#OLD:#     #XXX
#OLD:#     #XXX samples__ = SampleAttribute(doc="Samples data. 0th index is time", hasunique=False) # XXX
#OLD:#     #XXX labels__ = SampleAttribute(doc="Labels for the samples", hasunique=True)
#OLD:#     #XXX chunks__ = SampleAttribute(doc="Chunk identities for the samples", hasunique=True)
#OLD:#     #XXX # samples ids (already unique by definition)
#OLD:#     #XXX origids__ = SampleAttribute(doc="Chunk identities for the samples", hasunique=False)
#OLD:# 
#OLD:#     def __init__(self,
#OLD:#                  # for copy constructor
#OLD:#                  data=None,
#OLD:#                  dsattr=None,
#OLD:#                  # automatic dtype conversion
#OLD:#                  dtype=None,
#OLD:#                  # new instances
#OLD:#                  samples=None,
#OLD:#                  labels=None,
#OLD:#                  labels_map=None,
#OLD:#                  chunks=None,
#OLD:#                  origids=None,
#OLD:#                  # flags
#OLD:#                  check_data=True,
#OLD:#                  copy_samples=False,
#OLD:#                  copy_data=True,
#OLD:#                  copy_dsattr=True):
#OLD:#         """Initialize dataset instance
#OLD:# 
#OLD:#         There are basically two different way to create a dataset:
#OLD:# 
#OLD:#         1. Create a new dataset from samples and sample attributes.  In
#OLD:#            this mode a two-dimensional `ndarray` has to be passed to the
#OLD:#            `samples` keyword argument and the corresponding samples
#OLD:#            attributes are provided via the `labels` and `chunks`
#OLD:#            arguments.
#OLD:# 
#OLD:#         2. Copy contructor mode
#OLD:#             The second way is used internally to perform quick coyping
#OLD:#             of datasets, e.g. when performing feature selection. In this
#OLD:#             mode and the two dictionaries (`data` and `dsattr`) are
#OLD:#             required. For performance reasons this mode bypasses most of
#OLD:#             the sanity check performed by the previous mode, as for
#OLD:#             internal operations data integrity is assumed.
#OLD:# 
#OLD:# 
#OLD:#         :Parameters:
#OLD:#           data : dict
#OLD:#             Dictionary with an arbitrary number of entries. The value for
#OLD:#             each key in the dict has to be an ndarray with the
#OLD:#             same length as the number of rows in the samples array.
#OLD:#             A special entry in this dictionary is 'samples', a 2d array
#OLD:#             (samples x features). A shallow copy is stored in the object.
#OLD:#           dsattr : dict
#OLD:#             Dictionary of dataset attributes. An arbitrary number of
#OLD:#             arbitrarily named and typed objects can be stored here. A
#OLD:#             shallow copy of the dictionary is stored in the object.
#OLD:#           dtype: type | None
#OLD:#             If None -- do not change data type if samples
#OLD:#             is an ndarray. Otherwise convert samples to dtype.
#OLD:# 
#OLD:# 
#OLD:#         :Keywords:
#OLD:#           samples : ndarray
#OLD:#             2d array (samples x features)
#OLD:#           labels
#OLD:#             An array or scalar value defining labels for each samples.
#OLD:#             Generally `labels` should be numeric, unless `labels_map`
#OLD:#             is used
#OLD:#           labels_map : None or bool or dict
#OLD:#             Map original labels into numeric labels.  If True, the
#OLD:#             mapping is computed if labels are literal.  If is False,
#OLD:#             no mapping is computed. If dict instance -- provided
#OLD:#             mapping is verified and applied.  If you want to have
#OLD:#             labels_map just be present given already numeric labels,
#OLD:#             just assign labels_map dictionary to existing dataset
#OLD:#             instance
#OLD:#           chunks
#OLD:#             An array or scalar value defining chunks for each sample
#OLD:# 
#OLD:#         Each of the Keywords arguments overwrites what is/might be
#OLD:#         already in the `data` container.
#OLD:# 
#OLD:#         """
#OLD:# 
#OLD:#         #XXX ClassWithCollections.__init__(self)
#OLD:# 
#OLD:#         # see if data and dsattr are none, if so, make them empty dicts
#OLD:#         if data is None:
#OLD:#             data = {}
#OLD:#         if dsattr is None:
#OLD:#             dsattr = {}
#OLD:# 
#OLD:#         # initialize containers; default values are empty dicts
#OLD:#         # always make a shallow copy of what comes in, otherwise total chaos
#OLD:#         # is likely to happen soon
#OLD:#         if copy_data:
#OLD:#             # deep copy (cannot use copy.deepcopy, because samples is an
#OLD:#             # exception
#OLD:#             # but shallow copy first to get a shared version of the data in
#OLD:#             # any case
#OLD:#             lcl_data = data.copy()
#OLD:#             for k, v in data.iteritems():
#OLD:#                 # skip copying samples if requested
#OLD:#                 if k == 'samples' and not copy_samples:
#OLD:#                     continue
#OLD:#                 lcl_data[k] = v.copy()
#OLD:#         else:
#OLD:#             # shallow copy
#OLD:#             # XXX? yoh: it might be better speed wise just assign dictionary
#OLD:#             #      without any shallow .copy
#OLD:#             lcl_data = data.copy()
#OLD:# 
#OLD:#         if copy_dsattr and len(dsattr)>0:
#OLD:#             # deep copy
#OLD:#             if __debug__:
#OLD:#                 debug('DS', "Deep copying dsattr %s" % `dsattr`)
#OLD:#             lcl_dsattr = copy.deepcopy(dsattr)
#OLD:# 
#OLD:#         else:
#OLD:#             # shallow copy
#OLD:#             lcl_dsattr = copy.copy(dsattr)
#OLD:# 
#OLD:#         # has to be not private since otherwise derived methods
#OLD:#         # would have problem accessing it and _registerAttribute
#OLD:#         # would fail on lambda getters
#OLD:#         self._data = lcl_data
#OLD:#         """What makes a dataset."""
#OLD:# 
#OLD:#         self._dsattr = lcl_dsattr
#OLD:#         """Dataset attriibutes."""
#OLD:# 
#OLD:#         # store samples (and possibly transform/reshape/retype them)
#OLD:#         if not samples is None:
#OLD:#             if __debug__:
#OLD:#                 if lcl_data.has_key('samples'):
#OLD:#                     debug('DS',
#OLD:#                           "`Data` dict has `samples` (%s) but there is also" \
#OLD:#                           " __init__ parameter `samples` which overrides " \
#OLD:#                           " stored in `data`" % (`lcl_data['samples'].shape`))
#OLD:#             lcl_data['samples'] = self._shapeSamples(samples, dtype,
#OLD:#                                                      copy_samples)
#OLD:# 
#OLD:#         # TODO? we might want to have the same logic for chunks and labels
#OLD:#         #       ie if no labels present -- assign arange
#OLD:#         #   MH: don't think this is necessary -- or is there a use case?
#OLD:#         # labels
#OLD:#         if not labels is None:
#OLD:#             if __debug__:
#OLD:#                 if lcl_data.has_key('labels'):
#OLD:#                     debug('DS',
#OLD:#                           "`Data` dict has `labels` (%s) but there is also" +
#OLD:#                           " __init__ parameter `labels` which overrides " +
#OLD:#                           " stored in `data`" % (`lcl_data['labels']`))
#OLD:#             if lcl_data.has_key('samples'):
#OLD:#                 lcl_data['labels'] = \
#OLD:#                     self._expandSampleAttribute(labels, 'labels')
#OLD:# 
#OLD:#         # check if we got all required attributes
#OLD:#         for attr in self._requiredattributes:
#OLD:#             if not lcl_data.has_key(attr):
#OLD:#                 raise DatasetError, \
#OLD:#                       "Attribute %s is required to initialize dataset" % \
#OLD:#                       attr
#OLD:# 
#OLD:#         nsamples = self.nsamples
#OLD:# 
#OLD:#         # chunks
#OLD:#         if not chunks == None:
#OLD:#             lcl_data['chunks'] = \
#OLD:#                 self._expandSampleAttribute(chunks, 'chunks')
#OLD:#         elif not lcl_data.has_key('chunks'):
#OLD:#             # if no chunk information is given assume that every pattern
#OLD:#             # is its own chunk
#OLD:#             lcl_data['chunks'] = N.arange(nsamples)
#OLD:# 
#OLD:#         # samples origids
#OLD:#         if not origids is None:
#OLD:#             # simply assign if provided
#OLD:#             lcl_data['origids'] = origids
#OLD:#         elif not lcl_data.has_key('origids'):
#OLD:#             # otherwise contruct unqiue ones
#OLD:#             lcl_data['origids'] = N.arange(len(lcl_data['labels']))
#OLD:#         else:
#OLD:#             # assume origids have been specified already (copy constructor
#OLD:#             # mode) leave them as they are, e.g. to make origids survive
#OLD:#             # selectSamples()
#OLD:#             pass
#OLD:# 
#OLD:#         # Initialize attributes which are registered but were not setup
#OLD:#         for attr in self._registeredattributes:
#OLD:#             if not lcl_data.has_key(attr):
#OLD:#                 if __debug__:
#OLD:#                     debug("DS", "Initializing attribute %s" % attr)
#OLD:#                 lcl_data[attr] = N.zeros(nsamples)
#OLD:# 
#OLD:#         # labels_map
#OLD:#         labels_ = N.asarray(lcl_data['labels'])
#OLD:#         labels_map_known = lcl_dsattr.has_key('labels_map')
#OLD:#         if labels_map is True:
#OLD:#             # need to compose labels_map
#OLD:#             if labels_.dtype.char == 'S': # or not labels_map_known:
#OLD:#                 # Create mapping
#OLD:#                 ulabels = list(Set(labels_))
#OLD:#                 ulabels.sort()
#OLD:#                 labels_map = dict([ (x[1], x[0]) for x in enumerate(ulabels) ])
#OLD:#                 if __debug__:
#OLD:#                     debug('DS', 'Mapping for the labels computed to be %s'
#OLD:#                           % labels_map)
#OLD:#             else:
#OLD:#                 if __debug__:
#OLD:#                     debug('DS', 'Mapping of labels was requested but labels '
#OLD:#                           'are not strings. Skipped')
#OLD:#                 labels_map = None
#OLD:#             pass
#OLD:#         elif labels_map is False:
#OLD:#             labels_map = None
#OLD:# 
#OLD:#         if isinstance(labels_map, dict):
#OLD:#             if labels_map_known:
#OLD:#                 if __debug__:
#OLD:#                     debug('DS',
#OLD:#                           "`dsattr` dict has `labels_map` (%s) but there is also" \
#OLD:#                           " __init__ parameter `labels_map` (%s) which overrides " \
#OLD:#                           " stored in `dsattr`" % (lcl_dsattr['labels_map'], labels_map))
#OLD:# 
#OLD:#             lcl_dsattr['labels_map'] = labels_map
#OLD:#             # map labels if needed (if strings or was explicitely requested)
#OLD:#             if labels_.dtype.char == 'S' or not labels_map_known:
#OLD:#                 if __debug__:
#OLD:#                     debug('DS_', "Remapping labels using mapping %s" % labels_map)
#OLD:#                 # need to remap
#OLD:#                 # !!! N.array is important here
#OLD:#                 try:
#OLD:#                     lcl_data['labels'] = N.array(
#OLD:#                         [labels_map[x] for x in lcl_data['labels']])
#OLD:#                 except KeyError, e:
#OLD:#                     raise ValueError, "Provided labels_map %s is insufficient " \
#OLD:#                           "to map all the labels. Mapping for label %s is " \
#OLD:#                           "missing" % (labels_map, e)
#OLD:# 
#OLD:#         elif not lcl_dsattr.has_key('labels_map'):
#OLD:#             lcl_dsattr['labels_map'] = labels_map
#OLD:#         elif __debug__:
#OLD:#             debug('DS_', 'Not overriding labels_map in dsattr since it has one')
#OLD:# 
#OLD:#         if check_data:
#OLD:#             self._checkData()
#OLD:# 
#OLD:#         # lazy computation of unique members
#OLD:#         #self._resetallunique('_dsattr', self._dsattr)
#OLD:# 
#OLD:#         # Michael: we cannot do this conditional here. When selectSamples()
#OLD:#         # removes a whole data chunk the uniquechunks values will be invalid.
#OLD:#         # Same applies to labels of course.
#OLD:#         if not labels is None or not chunks is None:
#OLD:#             # for a speed up to don't go through all uniqueattributes
#OLD:#             # when no need
#OLD:#             lcl_dsattr['__uniquereseted'] = False
#OLD:#             self._resetallunique(force=True)
#OLD:# 
#OLD:# 
#OLD:#     __doc__ = enhancedDocString('Dataset', locals())
#OLD:# 
#OLD:# 
#OLD:#     @property
#OLD:#     def idhash(self):
#OLD:#         """To verify if dataset is in the same state as when smth else was done
#OLD:# 
#OLD:#         Like if classifier was trained on the same dataset as in question"""
#OLD:# 
#OLD:#         _data = self._data
#OLD:#         res = idhash_(_data)
#OLD:# 
#OLD:#         # we cannot count on the order the values in the dict will show up
#OLD:#         # with `self._data.value()` and since idhash will be order-dependent
#OLD:#         # we have to make it deterministic
#OLD:#         keys = _data.keys()
#OLD:#         keys.sort()
#OLD:#         for k in keys:
#OLD:#             res += idhash_(_data[k])
#OLD:#         return res
#OLD:# 
#OLD:# 
#OLD:#     def _resetallunique(self, force=False):
#OLD:#         """Set to None all unique* attributes of corresponding dictionary
#OLD:#         """
#OLD:#         _dsattr = self._dsattr
#OLD:# 
#OLD:#         if not force and _dsattr['__uniquereseted']:
#OLD:#             return
#OLD:# 
#OLD:#         _uniqueattributes = self._uniqueattributes
#OLD:# 
#OLD:#         if __debug__ and "DS_" in debug.active:
#OLD:#             debug("DS_", "Reseting all attributes %s for dataset %s"
#OLD:#                   % (_uniqueattributes,
#OLD:#                      self.summary(uniq=False, idhash=False,
#OLD:#                                   stats=False, lstats=False)))
#OLD:# 
#OLD:#         # I guess we better checked if dictname is known  but...
#OLD:#         for k in _uniqueattributes:
#OLD:#             _dsattr[k] = None
#OLD:#         _dsattr['__uniquereseted'] = True
#OLD:# 
#OLD:# 
#OLD:#     def _getuniqueattr(self, attrib, dict_):
#OLD:#         """Provide common facility to return unique attributes
#OLD:# 
#OLD:#         XXX `dict_` can be simply replaced now with self._dsattr
#OLD:#         """
#OLD:# 
#OLD:#         # local bindings
#OLD:#         _dsattr = self._dsattr
#OLD:# 
#OLD:#         if not _dsattr.has_key(attrib) or _dsattr[attrib] is None:
#OLD:#             if __debug__ and 'DS_' in debug.active:
#OLD:#                 debug("DS_", "Recomputing unique set for attrib %s within %s" %
#OLD:#                       (attrib, self.summary(uniq=False,
#OLD:#                                             stats=False, lstats=False)))
#OLD:#             # uff... might come up with better strategy to keep relevant
#OLD:#             # attribute name
#OLD:#             _dsattr[attrib] = N.unique( N.asanyarray(dict_[attrib[6:]]) )
#OLD:#             assert(not _dsattr[attrib] is None)
#OLD:#             _dsattr['__uniquereseted'] = False
#OLD:# 
#OLD:#         return _dsattr[attrib]
#OLD:# 
#OLD:# 
#OLD:#     def _setdataattr(self, attrib, value):
#OLD:#         """Provide common facility to set attributes
#OLD:# 
#OLD:#         """
#OLD:#         if len(value) != self.nsamples:
#OLD:#             raise ValueError, \
#OLD:#                   "Provided %s have %d entries while there is %d samples" % \
#OLD:#                   (attrib, len(value), self.nsamples)
#OLD:#         self._data[attrib] = N.asarray(value)
#OLD:#         uniqueattr = "unique" + attrib
#OLD:# 
#OLD:#         _dsattr = self._dsattr
#OLD:#         if _dsattr.has_key(uniqueattr):
#OLD:#             _dsattr[uniqueattr] = None
#OLD:# 
#OLD:# 
#OLD:#     def _getNSamplesPerAttr( self, attrib='labels' ):
#OLD:#         """Returns the number of samples per unique label.
#OLD:#         """
#OLD:#         # local bindings
#OLD:#         _data = self._data
#OLD:# 
#OLD:#         # XXX hardcoded dict_=self._data.... might be in self._dsattr
#OLD:#         uniqueattr = self._getuniqueattr(attrib="unique" + attrib,
#OLD:#                                          dict_=_data)
#OLD:# 
#OLD:#         # use dictionary to cope with arbitrary labels
#OLD:#         result = dict(zip(uniqueattr, [ 0 ] * len(uniqueattr)))
#OLD:#         for l in _data[attrib]:
#OLD:#             result[l] += 1
#OLD:# 
#OLD:#         # XXX only return values to mimic the old interface but we might want
#OLD:#         # to return the full dict instead
#OLD:#         # return result
#OLD:#         return result
#OLD:# 
#OLD:# 
#OLD:#     def _getSampleIdsByAttr(self, values, attrib="labels",
#OLD:#                             sort=True):
#OLD:#         """Return indecies of samples given a list of attributes
#OLD:#         """
#OLD:# 
#OLD:#         if not operator.isSequenceType(values) \
#OLD:#                or isinstance(values, basestring):
#OLD:#             values = [ values ]
#OLD:# 
#OLD:#         # TODO: compare to plain for loop through the labels
#OLD:#         #       on a real data example
#OLD:#         sel = N.array([], dtype=N.int16)
#OLD:#         _data = self._data
#OLD:#         for value in values:
#OLD:#             sel = N.concatenate((
#OLD:#                 sel, N.where(_data[attrib]==value)[0]))
#OLD:# 
#OLD:#         if sort:
#OLD:#             # place samples in the right order
#OLD:#             sel.sort()
#OLD:# 
#OLD:#         return sel
#OLD:# 
#OLD:# 
#OLD:#     def idsonboundaries(self, prior=0, post=0,
#OLD:#                         attributes_to_track=['labels', 'chunks'],
#OLD:#                         affected_labels=None,
#OLD:#                         revert=False):
#OLD:#         """Find samples which are on the boundaries of the blocks
#OLD:# 
#OLD:#         Such samples might need to be removed.  By default (with
#OLD:#         prior=0, post=0) ids of the first samples in a 'block' are
#OLD:#         reported
#OLD:# 
#OLD:#         :Parameters:
#OLD:#           prior : int
#OLD:#             how many samples prior to transition sample to include
#OLD:#           post : int
#OLD:#             how many samples post the transition sample to include
#OLD:#           attributes_to_track : list of basestring
#OLD:#             which attributes to track to decide on the boundary condition
#OLD:#           affected_labels : list of basestring
#OLD:#             for which labels to perform selection. If None - for all
#OLD:#           revert : bool
#OLD:#             either to revert the meaning and provide ids of samples which are found
#OLD:#             to not to be boundary samples
#OLD:#         """
#OLD:#         # local bindings
#OLD:#         _data = self._data
#OLD:#         labels = self.labels
#OLD:#         nsamples = self.nsamples
#OLD:# 
#OLD:#         lastseen = none = [None for attr in attributes_to_track]
#OLD:#         transitions = []
#OLD:# 
#OLD:#         for i in xrange(nsamples+1):
#OLD:#             if i < nsamples:
#OLD:#                 current = [_data[attr][i] for attr in attributes_to_track]
#OLD:#             else:
#OLD:#                 current = none
#OLD:#             if lastseen != current:
#OLD:#                 # transition point
#OLD:#                 new_transitions = range(max(0, i-prior),
#OLD:#                                         min(nsamples-1, i+post)+1)
#OLD:#                 if affected_labels is not None:
#OLD:#                     new_transitions = [labels[i] for i in new_transitions
#OLD:#                                        if i in affected_labels]
#OLD:#                 transitions += new_transitions
#OLD:#                 lastseen = current
#OLD:# 
#OLD:#         transitions = Set(transitions)
#OLD:#         if revert:
#OLD:#             transitions = Set(range(nsamples)).difference(transitions)
#OLD:# 
#OLD:#         # postprocess
#OLD:#         transitions = N.array(list(transitions))
#OLD:#         transitions.sort()
#OLD:#         return list(transitions)
#OLD:# 
#OLD:# 
#OLD:#     def _shapeSamples(self, samples, dtype, copy):
#OLD:#         """Adapt different kinds of samples
#OLD:# 
#OLD:#         Handle all possible input value for 'samples' and tranform
#OLD:#         them into a 2d (samples x feature) representation.
#OLD:#         """
#OLD:#         # put samples array into correct shape
#OLD:#         # 1d arrays or simple sequences are assumed to be a single pattern
#OLD:#         if (not isinstance(samples, N.ndarray)):
#OLD:#             # it is safe to provide dtype which defaults to None,
#OLD:#             # when N would choose appropriate dtype automagically
#OLD:#             samples = N.array(samples, ndmin=2, dtype=dtype, copy=copy)
#OLD:#         else:
#OLD:#             if samples.ndim < 2 \
#OLD:#                    or (not dtype is None and dtype != samples.dtype):
#OLD:#                 if dtype is None:
#OLD:#                     dtype = samples.dtype
#OLD:#                 samples = N.array(samples, ndmin=2, dtype=dtype, copy=copy)
#OLD:#             elif copy:
#OLD:#                 samples = samples.copy()
#OLD:# 
#OLD:#         # only samples x features matrices are supported
#OLD:#         if len(samples.shape) > 2:
#OLD:#             raise DatasetError, "Only (samples x features) -> 2d sample " \
#OLD:#                             + "are supported (got %s shape of samples)." \
#OLD:#                             % (`samples.shape`) \
#OLD:#                             +" Consider MappedDataset if applicable."
#OLD:# 
#OLD:#         return samples
#OLD:# 
#OLD:# 
#OLD:#     def _checkData(self):
#OLD:#         """Checks `_data` members to have the same # of samples.
#OLD:#         """
#OLD:#         #
#OLD:#         # XXX: Maybe just run this under __debug__ and remove the `check_data`
#OLD:#         #      from the constructor, which is too complicated anyway?
#OLD:#         #
#OLD:# 
#OLD:#         # local bindings
#OLD:#         nsamples = self.nsamples
#OLD:#         _data = self._data
#OLD:# 
#OLD:#         for k, v in _data.iteritems():
#OLD:#             if not len(v) == nsamples:
#OLD:#                 raise DatasetError, \
#OLD:#                       "Length of sample attribute '%s' [%i] does not " \
#OLD:#                       "match the number of samples in the dataset [%i]." \
#OLD:#                       % (k, len(v), nsamples)
#OLD:# 
#OLD:#         # check for unique origids
#OLD:#         uniques = N.unique(_data['origids'])
#OLD:#         uniques.sort()
#OLD:#         # need to copy to prevent sorting the original array
#OLD:#         sorted_ids = _data['origids'].copy()
#OLD:#         sorted_ids.sort()
#OLD:# 
#OLD:#         if not (uniques == sorted_ids).all():
#OLD:#             raise DatasetError, "Samples IDs are not unique."
#OLD:# 
#OLD:#         # Check if labels as not literal
#OLD:#         if N.asanyarray(_data['labels'].dtype.char == 'S'):
#OLD:#             warning('Labels for dataset %s are literal, should be numeric. '
#OLD:#                     'You might like to use labels_map argument.' % self)
#OLD:# 
#OLD:#     def _expandSampleAttribute(self, attr, attr_name):
#OLD:#         """If a sample attribute is given as a scalar expand/repeat it to a
#OLD:#         length matching the number of samples in the dataset.
#OLD:#         """
#OLD:#         try:
#OLD:#             # if we are initializing with a single string -- we should
#OLD:#             # treat it as a single label
#OLD:#             if isinstance(attr, basestring):
#OLD:#                 raise TypeError
#OLD:#             if len(attr) != self.nsamples:
#OLD:#                 raise DatasetError, \
#OLD:#                       "Length of sample attribute '%s' [%d]" \
#OLD:#                       % (attr_name, len(attr)) \
#OLD:#                       + " has to match the number of samples" \
#OLD:#                       + " [%d]." % self.nsamples
#OLD:#             # store the sequence as array
#OLD:#             return N.array(attr)
#OLD:# 
#OLD:#         except TypeError:
#OLD:#             # make sequence of identical value matching the number of
#OLD:#             # samples
#OLD:#             return N.repeat(attr, self.nsamples)
#OLD:# 
#OLD:# 
#OLD:#     @classmethod
#OLD:#     def _registerAttribute(cls, key, dictname="_data", abbr=None, hasunique=False):
#OLD:#         """Register an attribute for any Dataset class.
#OLD:# 
#OLD:#         Creates property assigning getters/setters depending on the
#OLD:#         availability of corresponding _get, _set functions.
#OLD:#         """
#OLD:#         classdict = cls.__dict__
#OLD:#         if not classdict.has_key(key):
#OLD:#             if __debug__:
#OLD:#                 debug("DS", "Registering new attribute %s" % key)
#OLD:#             # define get function and use corresponding
#OLD:#             # _getATTR if such defined
#OLD:#             getter = '_get%s' % key
#OLD:#             if classdict.has_key(getter):
#OLD:#                 getter =  '%s.%s' % (cls.__name__, getter)
#OLD:#             else:
#OLD:#                 getter = "lambda x: x.%s['%s']" % (dictname, key)
#OLD:# 
#OLD:#             # define set function and use corresponding
#OLD:#             # _setATTR if such defined
#OLD:#             setter = '_set%s' % key
#OLD:#             if classdict.has_key(setter):
#OLD:#                 setter =  '%s.%s' % (cls.__name__, setter)
#OLD:#             elif dictname=="_data":
#OLD:#                 setter = "lambda self,x: self._setdataattr" + \
#OLD:#                          "(attrib='%s', value=x)" % (key)
#OLD:#             else:
#OLD:#                 setter = None
#OLD:# 
#OLD:#             if __debug__:
#OLD:#                 debug("DS", "Registering new property %s.%s" %
#OLD:#                       (cls.__name__, key))
#OLD:#             exec "%s.%s = property(fget=%s, fset=%s)"  % \
#OLD:#                  (cls.__name__, key, getter, setter)
#OLD:# 
#OLD:#             if abbr is not None:
#OLD:#                 exec "%s.%s = property(fget=%s, fset=%s)"  % \
#OLD:#                      (cls.__name__, abbr, getter, setter)
#OLD:# 
#OLD:#             if hasunique:
#OLD:#                 uniquekey = "unique%s" % key
#OLD:#                 getter = '_get%s' % uniquekey
#OLD:#                 if classdict.has_key(getter):
#OLD:#                     getter = '%s.%s' % (cls.__name__, getter)
#OLD:#                 else:
#OLD:#                     getter = "lambda x: x._getuniqueattr" + \
#OLD:#                             "(attrib='%s', dict_=x.%s)" % (uniquekey, dictname)
#OLD:# 
#OLD:#                 if __debug__:
#OLD:#                     debug("DS", "Registering new property %s.%s" %
#OLD:#                           (cls.__name__, uniquekey))
#OLD:# 
#OLD:#                 exec "%s.%s = property(fget=%s)" % \
#OLD:#                      (cls.__name__, uniquekey, getter)
#OLD:#                 if abbr is not None:
#OLD:#                     exec "%s.U%s = property(fget=%s)" % \
#OLD:#                          (cls.__name__, abbr, getter)
#OLD:# 
#OLD:#                 # create samplesper<ATTR> properties
#OLD:#                 sampleskey = "samplesper%s" % key[:-1] # remove ending 's' XXX
#OLD:#                 if __debug__:
#OLD:#                     debug("DS", "Registering new property %s.%s" %
#OLD:#                           (cls.__name__, sampleskey))
#OLD:# 
#OLD:#                 exec "%s.%s = property(fget=%s)" % \
#OLD:#                      (cls.__name__, sampleskey,
#OLD:#                       "lambda x: x._getNSamplesPerAttr(attrib='%s')" % key)
#OLD:# 
#OLD:#                 cls._uniqueattributes.append(uniquekey)
#OLD:# 
#OLD:#                 # create idsby<ATTR> properties
#OLD:#                 sampleskey = "idsby%s" % key # remove ending 's' XXX
#OLD:#                 if __debug__:
#OLD:#                     debug("DS", "Registering new property %s.%s" %
#OLD:#                           (cls.__name__, sampleskey))
#OLD:# 
#OLD:#                 exec "%s.%s = %s" % (cls.__name__, sampleskey,
#OLD:#                       "lambda self, x: " +
#OLD:#                       "self._getSampleIdsByAttr(x,attrib='%s')" % key)
#OLD:# 
#OLD:#                 cls._uniqueattributes.append(uniquekey)
#OLD:# 
#OLD:#             cls._registeredattributes.append(key)
#OLD:#         elif __debug__:
#OLD:#             warning('Trying to reregister attribute `%s`. For now ' % key +
#OLD:#                     'such capability is not present')
#OLD:# 
#OLD:# 
#OLD:#     def __str__(self):
#OLD:#         """String summary over the object
#OLD:#         """
#OLD:#         return self.summary(uniq=True,
#OLD:#                             idhash=__debug__ and ('DS_ID' in debug.active),
#OLD:#                             stats=__debug__ and ('DS_STATS' in debug.active),
#OLD:#                             lstats=__debug__ and ('DS_STATS' in debug.active),
#OLD:#                             )
#OLD:# 
#OLD:# 
#OLD:#     def __repr__(self):
#OLD:#         return "<%s>" % str(self)
#OLD:# 
#OLD:# 
#OLD:#     def summary(self, uniq=True, stats=True, idhash=False, lstats=True,
#OLD:#                 maxc=30, maxl=20):
#OLD:#         """String summary over the object
#OLD:# 
#OLD:#         :Parameters:
#OLD:#           uniq : bool
#OLD:#              Include summary over data attributes which have unique
#OLD:#           idhash : bool
#OLD:#              Include idhash value for dataset and samples
#OLD:#           stats : bool
#OLD:#              Include some basic statistics (mean, std, var) over dataset samples
#OLD:#           lstats : bool
#OLD:#              Include statistics on chunks/labels
#OLD:#           maxc : int
#OLD:#             Maximal number of chunks when provide details on labels/chunks
#OLD:#           maxl : int
#OLD:#             Maximal number of labels when provide details on labels/chunks
#OLD:#         """
#OLD:#         # local bindings
#OLD:#         samples = self.samples
#OLD:#         _data = self._data
#OLD:#         _dsattr = self._dsattr
#OLD:# 
#OLD:#         if idhash:
#OLD:#             idhash_ds = "{%s}" % self.idhash
#OLD:#             idhash_samples = "{%s}" % idhash_(samples)
#OLD:#         else:
#OLD:#             idhash_ds = ""
#OLD:#             idhash_samples = ""
#OLD:# 
#OLD:#         s = """Dataset %s/ %s %d%s x %d""" % \
#OLD:#             (idhash_ds, samples.dtype,
#OLD:#              self.nsamples, idhash_samples, self.nfeatures)
#OLD:# 
#OLD:#         ssep = (' ', '\n')[lstats]
#OLD:#         if uniq:
#OLD:#             s +=  "%suniq:" % ssep
#OLD:#             for uattr in _dsattr.keys():
#OLD:#                 if not uattr.startswith("unique"):
#OLD:#                     continue
#OLD:#                 attr = uattr[6:]
#OLD:#                 try:
#OLD:#                     value = self._getuniqueattr(attrib=uattr,
#OLD:#                                                 dict_=_data)
#OLD:#                     s += " %d %s" % (len(value), attr)
#OLD:#                 except:
#OLD:#                     pass
#OLD:# 
#OLD:#         if isinstance(self.labels_map, dict):
#OLD:#             s += ' labels_mapped'
#OLD:# 
#OLD:#         if stats:
#OLD:#             # TODO -- avg per chunk?
#OLD:#             # XXX We might like to use scipy.stats.describe to get
#OLD:#             # quick summary statistics (mean/range/skewness/kurtosis)
#OLD:#             s += "%sstats: mean=%g std=%g var=%g min=%g max=%g\n" % \
#OLD:#                  (ssep, N.mean(samples), N.std(samples),
#OLD:#                   N.var(samples), N.min(samples), N.max(samples))
#OLD:# 
#OLD:#         if lstats:
#OLD:#             s += self.summary_labels(maxc=maxc, maxl=maxl)
#OLD:# 
#OLD:#         return s
#OLD:# 
#OLD:# 
#OLD:#     def summary_labels(self, maxc=30, maxl=20):
#OLD:#         """Provide summary statistics over the labels and chunks
#OLD:# 
#OLD:#         :Parameters:
#OLD:#           maxc : int
#OLD:#             Maximal number of chunks when provide details
#OLD:#           maxl : int
#OLD:#             Maximal number of labels when provide details
#OLD:#         """
#OLD:#         # We better avoid bound function since if people only
#OLD:#         # imported Dataset without miscfx it would fail
#OLD:#         from mvpa.datasets.miscfx import getSamplesPerChunkLabel
#OLD:#         spcl = getSamplesPerChunkLabel(self)
#OLD:#         # XXX couldn't they be unordered?
#OLD:#         ul = self.uniquelabels.tolist()
#OLD:#         uc = self.uniquechunks.tolist()
#OLD:#         s = ""
#OLD:#         if len(ul) < maxl and len(uc) < maxc:
#OLD:#             s += "\nCounts of labels in each chunk:"
#OLD:#             # only in a resonable case do printing
#OLD:#             table = [['  chunks\labels'] + ul]
#OLD:#             table += [[''] + ['---'] * len(ul)]
#OLD:#             for c, counts in zip(uc, spcl):
#OLD:#                 table.append([ str(c) ] + counts.tolist())
#OLD:#             s += '\n' + table2string(table)
#OLD:#         else:
#OLD:#             s += "No details due to large number of labels or chunks. " \
#OLD:#                  "Increase maxc and maxl if desired"
#OLD:# 
#OLD:#         labels_map = self.labels_map
#OLD:#         if isinstance(labels_map, dict):
#OLD:#             s += "\nOriginal labels were mapped using following mapping:"
#OLD:#             s += '\n\t'+'\n\t'.join([':\t'.join(map(str, x))
#OLD:#                                      for x in labels_map.items()]) + '\n'
#OLD:# 
#OLD:#         def cl_stats(axis, u, name1, name2):
#OLD:#             """ Compute statistics per label
#OLD:#             """
#OLD:#             stats = {'min': N.min(spcl, axis=axis),
#OLD:#                      'max': N.max(spcl, axis=axis),
#OLD:#                      'mean': N.mean(spcl, axis=axis),
#OLD:#                      'std': N.std(spcl, axis=axis),
#OLD:#                      '#%ss' % name2: N.sum(spcl>0, axis=axis)}
#OLD:#             entries = ['  ' + name1, 'mean', 'std', 'min', 'max', '#%ss' % name2]
#OLD:#             table = [ entries ]
#OLD:#             for i, l in enumerate(u):
#OLD:#                 d = {'  ' + name1 : l}
#OLD:#                 d.update(dict([ (k, stats[k][i]) for k in stats.keys()]))
#OLD:#                 table.append( [ ('%.3g', '%s')[isinstance(d[e], basestring)]
#OLD:#                                 % d[e] for e in entries] )
#OLD:#             return '\nSummary per %s across %ss\n' % (name1, name2) \
#OLD:#                    + table2string(table)
#OLD:# 
#OLD:#         if len(ul) < maxl:
#OLD:#             s += cl_stats(0, ul, 'label', 'chunk')
#OLD:#         if len(uc) < maxc:
#OLD:#             s += cl_stats(1, uc, 'chunk', 'label')
#OLD:#         return s
#OLD:# 
#OLD:# 
#OLD:#     def __iadd__(self, other):
#OLD:#         """Merge the samples of one Dataset object to another (in-place).
#OLD:# 
#OLD:#         No dataset attributes, besides labels_map, will be merged!
#OLD:#         Additionally, a new set of unique `origids` will be generated.
#OLD:#         """
#OLD:#         # local bindings
#OLD:#         _data = self._data
#OLD:#         other_data = other._data
#OLD:# 
#OLD:#         if not self.nfeatures == other.nfeatures:
#OLD:#             raise DatasetError, "Cannot add Dataset, because the number of " \
#OLD:#                                 "feature do not match."
#OLD:# 
#OLD:#         # take care about labels_map and labels
#OLD:#         slm = self.labels_map
#OLD:#         olm = other.labels_map
#OLD:#         if N.logical_xor(slm is None, olm is None):
#OLD:#             raise ValueError, "Cannot add datasets where only one of them " \
#OLD:#                   "has labels map assigned. If needed -- implement it"
#OLD:# 
#OLD:#         # concatenate all sample attributes
#OLD:#         for k,v in _data.iteritems():
#OLD:#             if k == 'origids':
#OLD:#                 # special case samples origids: for now just regenerate unique
#OLD:#                 # ones could also check if concatenation is unique, but it
#OLD:#                 # would be costly performance-wise
#OLD:#                 _data[k] = N.arange(len(v) + len(other_data[k]))
#OLD:# 
#OLD:#             elif k == 'labels' and slm is not None:
#OLD:#                 # special care about labels if mapping was in effect,
#OLD:#                 # we need to append 2nd map to the first one and
#OLD:#                 # relabel 2nd dataset
#OLD:#                 nlm = slm.copy()
#OLD:#                 # figure out maximal numerical label used now
#OLD:#                 nextid = N.sort(nlm.values())[-1] + 1
#OLD:#                 olabels = other.labels
#OLD:#                 olabels_remap = {}
#OLD:#                 for ol, olnum in olm.iteritems():
#OLD:#                     if not nlm.has_key(ol):
#OLD:#                         # check if we can preserve old numberic label
#OLD:#                         # if not -- assign some new one not yet present
#OLD:#                         # in any dataset
#OLD:#                         if olnum in nlm.values():
#OLD:#                             nextid = N.sort(nlm.values() + olm.values())[-1] + 1
#OLD:#                         else:
#OLD:#                             nextid = olnum
#OLD:#                         olabels_remap[olnum] = nextid
#OLD:#                         nlm[ol] = nextid
#OLD:#                         nextid += 1
#OLD:#                     else:
#OLD:#                         olabels_remap[olnum] = nlm[ol]
#OLD:#                 olabels = [olabels_remap[x] for x in olabels]
#OLD:#                 # finally compose new labels
#OLD:#                 _data['labels'] = N.concatenate((v, olabels), axis=0)
#OLD:#                 # and reassign new mapping
#OLD:#                 self._dsattr['labels_map'] = nlm
#OLD:# 
#OLD:#                 if __debug__:
#OLD:#                     # check if we are not dealing with colliding
#OLD:#                     # mapping, since it is problematic and might lead
#OLD:#                     # to various complications
#OLD:#                     if (len(Set(slm.keys())) != len(Set(slm.values()))) or \
#OLD:#                        (len(Set(olm.keys())) != len(Set(olm.values()))):
#OLD:#                         warning("Adding datasets where multiple labels "
#OLD:#                                 "mapped to the same ID is not recommended. "
#OLD:#                                 "Please check the outcome. Original mappings "
#OLD:#                                 "were %s and %s. Resultant is %s"
#OLD:#                                 % (slm, olm, nlm))
#OLD:# 
#OLD:#             else:
#OLD:#                 _data[k] = N.concatenate((v, other_data[k]), axis=0)
#OLD:# 
#OLD:#         # might be more sophisticated but for now just reset -- it is safer ;)
#OLD:#         self._resetallunique()
#OLD:# 
#OLD:#         return self
#OLD:# 
#OLD:# 
#OLD:#     def __add__( self, other ):
#OLD:#         """Merge the samples two Dataset objects.
#OLD:# 
#OLD:#         All data of both datasets is copied, concatenated and a new Dataset is
#OLD:#         returned.
#OLD:# 
#OLD:#         NOTE: This can be a costly operation (both memory and time). If
#OLD:#         performance is important consider the '+=' operator.
#OLD:#         """
#OLD:#         # create a new object of the same type it is now and NOT only Dataset
#OLD:#         out = super(Dataset, self).__new__(self.__class__)
#OLD:# 
#OLD:#         # now init it: to make it work all Dataset contructors have to accept
#OLD:#         # Class(data=Dict, dsattr=Dict)
#OLD:#         out.__init__(data=self._data,
#OLD:#                      dsattr=self._dsattr,
#OLD:#                      copy_samples=True,
#OLD:#                      copy_data=True,
#OLD:#                      copy_dsattr=True)
#OLD:# 
#OLD:#         out += other
#OLD:# 
#OLD:#         return out
#OLD:# 
#OLD:# 
#OLD:#     def copy(self, deep=True):
#OLD:#         """Create a copy (clone) of the dataset, by fully copying current one
#OLD:# 
#OLD:#         :Keywords:
#OLD:#           deep : bool
#OLD:#             deep flag is provided to __init__ for
#OLD:#             copy_{samples,data,dsattr}. By default full copy is done.
#OLD:#         """
#OLD:#         # create a new object of the same type it is now and NOT only Dataset
#OLD:#         out = super(Dataset, self).__new__(self.__class__)
#OLD:# 
#OLD:#         # now init it: to make it work all Dataset contructors have to accept
#OLD:#         # Class(data=Dict, dsattr=Dict)
#OLD:#         out.__init__(data=self._data,
#OLD:#                      dsattr=self._dsattr,
#OLD:#                      copy_samples=True,
#OLD:#                      copy_data=True,
#OLD:#                      copy_dsattr=True)
#OLD:# 
#OLD:#         return out
#OLD:# 
#OLD:# 
#OLD:#     def selectFeatures(self, ids=None, sort=True, groups=None):
#OLD:#         """Select a number of features from the current set.
#OLD:# 
#OLD:#         :Parameters:
#OLD:#           ids
#OLD:#             iterable container to select ids
#OLD:#           sort : bool
#OLD:#             if to sort Ids. Order matters and `selectFeatures` assumes
#OLD:#             incremental order. If not such, in non-optimized code
#OLD:#             selectFeatures would verify the order and sort
#OLD:# 
#OLD:#         Returns a new Dataset object with a copy of corresponding features
#OLD:# 		from the original samples array.
#OLD:# 
#OLD:#         WARNING: The order of ids determines the order of features in
#OLD:#         the returned dataset. This might be useful sometimes, but can
#OLD:#         also cause major headaches! Order would is verified when
#OLD:#         running in non-optimized code (if __debug__)
#OLD:#         """
#OLD:#         if ids is None and groups is None:
#OLD:#             raise ValueError, "No feature selection specified."
#OLD:# 
#OLD:#         # start with empty list if no ids where specified (so just groups)
#OLD:#         if ids is None:
#OLD:#             ids = []
#OLD:# 
#OLD:#         if not groups is None:
#OLD:#             if not self._dsattr.has_key('featuregroups'):
#OLD:#                 raise RuntimeError, \
#OLD:#                 "Dataset has no feature grouping information."
#OLD:# 
#OLD:#             for g in groups:
#OLD:#                 ids += (self._dsattr['featuregroups'] == g).nonzero()[0].tolist()
#OLD:# 
#OLD:#         # XXX set sort default to True, now sorting has to be explicitely
#OLD:#         # disabled and warning is not necessary anymore
#OLD:#         if sort:
#OLD:#             ids = copy.deepcopy(ids)
#OLD:#             ids.sort()
#OLD:#         elif __debug__ and 'CHECK_DS_SORTED' in debug.active:
#OLD:#             from mvpa.misc.support import isSorted
#OLD:#             if not isSorted(ids):
#OLD:#                 warning("IDs for selectFeatures must be provided " +
#OLD:#                        "in sorted order, otherwise major headache might occur")
#OLD:# 
#OLD:#         # shallow-copy all stuff from current data dict
#OLD:#         new_data = self._data.copy()
#OLD:# 
#OLD:#         # assign the selected features -- data is still shared with
#OLD:#         # current dataset
#OLD:#         new_data['samples'] = self._data['samples'][:, ids]
#OLD:# 
#OLD:#         # apply selection to feature groups as well
#OLD:#         if self._dsattr.has_key('featuregroups'):
#OLD:#             new_dsattr = self._dsattr.copy()
#OLD:#             new_dsattr['featuregroups'] = self._dsattr['featuregroups'][ids]
#OLD:#         else:
#OLD:#             new_dsattr = self._dsattr
#OLD:# 
#OLD:#         # create a new object of the same type it is now and NOT only Dataset
#OLD:#         dataset = super(Dataset, self).__new__(self.__class__)
#OLD:# 
#OLD:#         # now init it: to make it work all Dataset contructors have to accept
#OLD:#         # Class(data=Dict, dsattr=Dict)
#OLD:#         dataset.__init__(data=new_data,
#OLD:#                          dsattr=new_dsattr,
#OLD:#                          check_data=False,
#OLD:#                          copy_samples=False,
#OLD:#                          copy_data=False,
#OLD:#                          copy_dsattr=False
#OLD:#                          )
#OLD:# 
#OLD:#         return dataset
#OLD:# 
#OLD:# 
#OLD:#     def applyMapper(self, featuresmapper=None, samplesmapper=None,
#OLD:#                     train=True):
#OLD:#         """Obtain new dataset by applying mappers over features and/or samples.
#OLD:# 
#OLD:#         While featuresmappers leave the sample attributes information
#OLD:#         unchanged, as the number of samples in the dataset is invariant,
#OLD:#         samplesmappers are also applied to the samples attributes themselves!
#OLD:# 
#OLD:#         Applying a featuresmapper will destroy any feature grouping information.
#OLD:# 
#OLD:#         :Parameters:
#OLD:#           featuresmapper : Mapper
#OLD:#             `Mapper` to somehow transform each sample's features
#OLD:#           samplesmapper : Mapper
#OLD:#             `Mapper` to transform each feature across samples
#OLD:#           train : bool
#OLD:#             Flag whether to train the mapper with this dataset before applying
#OLD:#             it.
#OLD:# 
#OLD:#         TODO: selectFeatures is pretty much
#OLD:#               applyMapper(featuresmapper=MaskMapper(...))
#OLD:#         """
#OLD:# 
#OLD:#         # shallow-copy all stuff from current data dict
#OLD:#         new_data = self._data.copy()
#OLD:# 
#OLD:#         # apply mappers
#OLD:# 
#OLD:#         if samplesmapper:
#OLD:#             if __debug__:
#OLD:#                 debug("DS", "Training samplesmapper %s" % `samplesmapper`)
#OLD:#             samplesmapper.train(self)
#OLD:# 
#OLD:#             if __debug__:
#OLD:#                 debug("DS", "Applying samplesmapper %s" % `samplesmapper` +
#OLD:#                       " to samples of dataset `%s`" % `self`)
#OLD:# 
#OLD:#             # get rid of existing 'origids' as they are not valid anymore and
#OLD:#             # applying a mapper to them is not really meaningful
#OLD:#             if new_data.has_key('origids'):
#OLD:#                 del(new_data['origids'])
#OLD:# 
#OLD:#             # apply mapper to all sample-wise data in dataset
#OLD:#             for k in new_data.keys():
#OLD:#                 new_data[k] = samplesmapper.forward(self._data[k])
#OLD:# 
#OLD:#         # feature mapping might affect dataset attributes
#OLD:#         # XXX: might be obsolete when proper feature attributes are implemented
#OLD:#         new_dsattr = self._dsattr
#OLD:# 
#OLD:#         if featuresmapper:
#OLD:#             if __debug__:
#OLD:#                 debug("DS", "Training featuresmapper %s" % `featuresmapper`)
#OLD:#             featuresmapper.train(self)
#OLD:# 
#OLD:#             if __debug__:
#OLD:#                 debug("DS", "Applying featuresmapper %s" % `featuresmapper` +
#OLD:#                       " to samples of dataset `%s`" % `self`)
#OLD:#             new_data['samples'] = featuresmapper.forward(self._data['samples'])
#OLD:# 
#OLD:#             # remove feature grouping, who knows what the mapper did to the
#OLD:#             # features
#OLD:#             if self._dsattr.has_key('featuregroups'):
#OLD:#                 new_dsattr = self._dsattr.copy()
#OLD:#                 del(new_dsattr['featuregroups'])
#OLD:#             else:
#OLD:#                 new_dsattr = self._dsattr
#OLD:# 
#OLD:#         # create a new object of the same type it is now and NOT only Dataset
#OLD:#         dataset = super(Dataset, self).__new__(self.__class__)
#OLD:# 
#OLD:#         # now init it: to make it work all Dataset contructors have to accept
#OLD:#         # Class(data=Dict, dsattr=Dict)
#OLD:#         dataset.__init__(data=new_data,
#OLD:#                          dsattr=new_dsattr,
#OLD:#                          check_data=False,
#OLD:#                          copy_samples=False,
#OLD:#                          copy_data=False,
#OLD:#                          copy_dsattr=False
#OLD:#                          )
#OLD:# 
#OLD:#         # samples attributes might have changed after applying samplesmapper
#OLD:#         if samplesmapper:
#OLD:#             dataset._resetallunique(force=True)
#OLD:# 
#OLD:#         return dataset
#OLD:# 
#OLD:# 
#OLD:#     def selectSamples(self, ids):
#OLD:#         """Choose a subset of samples defined by samples IDs.
#OLD:# 
#OLD:#         Returns a new dataset object containing the selected sample
#OLD:#         subset.
#OLD:# 
#OLD:#         TODO: yoh, we might need to sort the mask if the mask is a
#OLD:#         list of ids and is not ordered. Clarify with Michael what is
#OLD:#         our intent here!
#OLD:#         """
#OLD:#         # without having a sequence a index the masked sample array would
#OLD:#         # loose its 2d layout
#OLD:#         if not operator.isSequenceType( ids ):
#OLD:#             ids = [ids]
#OLD:#         # TODO: Reconsider crafting a slice if it can be done to don't copy
#OLD:#         #       the data
#OLD:#         #try:
#OLD:#         #    minmask = min(mask)
#OLD:#         #    maxmask = max(mask)
#OLD:#         #except:
#OLD:#         #    minmask = min(map(int,mask))
#OLD:#         #    maxmask = max(map(int,mask))
#OLD:#         # lets see if we could get it done with cheap view/slice
#OLD:#         #(minmask, maxmask) != (0, 1) and \
#OLD:#         #if len(mask) > 2 and \
#OLD:#         #       N.array([N.arange(minmask, maxmask+1) == N.array(mask)]).all():
#OLD:#         #    slice_ = slice(minmask, maxmask+1)
#OLD:#         #    if __debug__:
#OLD:#         #        debug("DS", "We can and do convert mask %s into splice %s" %
#OLD:#         #              (mask, slice_))
#OLD:#         #    mask = slice_
#OLD:#         # mask all sample attributes
#OLD:#         data = {}
#OLD:#         for k, v in self._data.iteritems():
#OLD:#             data[k] = v[ids, ]
#OLD:# 
#OLD:#         # create a new object of the same type it is now and NOT onyl Dataset
#OLD:#         dataset = super(Dataset, self).__new__(self.__class__)
#OLD:# 
#OLD:#         # now init it: to make it work all Dataset contructors have to accept
#OLD:#         # Class(data=Dict, dsattr=Dict)
#OLD:#         dataset.__init__(data=data,
#OLD:#                          dsattr=self._dsattr,
#OLD:#                          check_data=False,
#OLD:#                          copy_samples=False,
#OLD:#                          copy_data=False,
#OLD:#                          copy_dsattr=False)
#OLD:# 
#OLD:#         dataset._resetallunique(force=True)
#OLD:#         return dataset
#OLD:# 
#OLD:# 
#OLD:# 
#OLD:#     def index(self, *args, **kwargs):
#OLD:#         """Universal indexer to obtain indexes of interesting samples/features.
#OLD:#         See .select() for more information
#OLD:# 
#OLD:#         :Return: tuple of (samples indexes, features indexes). Each
#OLD:#           item could be also None, if no selection on samples or
#OLD:#           features was requested (to discriminate between no selected
#OLD:#           items, and no selections)
#OLD:#         """
#OLD:#         s_indx = []                     # selections for samples
#OLD:#         f_indx = []                     # selections for features
#OLD:#         return_dataset = kwargs.pop('return_dataset', False)
#OLD:#         largs = len(args)
#OLD:# 
#OLD:#         args = list(args)               # so we could override
#OLD:#         # Figure out number of positional
#OLD:#         largs_nonstring = 0
#OLD:#         # need to go with index since we might need to override internally
#OLD:#         for i in xrange(largs):
#OLD:#             l = args[i]
#OLD:#             if isinstance(l, basestring):
#OLD:#                 if l.lower() == 'all':
#OLD:#                     # override with a slice
#OLD:#                     args[i] = slice(None)
#OLD:#                 else:
#OLD:#                     break
#OLD:#             largs_nonstring += 1
#OLD:# 
#OLD:#         if largs_nonstring >= 1:
#OLD:#             s_indx.append(args[0])
#OLD:#             if __debug__ and 'CHECK_DS_SELECT' in debug.active:
#OLD:#                 _validate_indexes_uniq_sorted(args[0], 'select', 'samples')
#OLD:#             if largs_nonstring == 2:
#OLD:#                 f_indx.append(args[1])
#OLD:#                 if __debug__ and 'CHECK_DS_SELECT' in debug.active:
#OLD:#                     _validate_indexes_uniq_sorted(args[1], 'select', 'features')
#OLD:#             elif largs_nonstring > 2:
#OLD:#                 raise ValueError, "Only two positional arguments are allowed" \
#OLD:#                       ". 1st for samples, 2nd for features"
#OLD:# 
#OLD:#         # process left positional arguments which must encode selections like
#OLD:#         # ('labels', [1,2,3])
#OLD:# 
#OLD:#         if (largs - largs_nonstring) % 2 != 0:
#OLD:#             raise ValueError, "Positional selections must come in pairs:" \
#OLD:#                   " e.g. ('labels', [1,2,3])"
#OLD:# 
#OLD:#         for i in xrange(largs_nonstring, largs, 2):
#OLD:#             k, v = args[i:i+2]
#OLD:#             kwargs[k] = v
#OLD:# 
#OLD:#         # process keyword parameters
#OLD:#         data_ = self._data
#OLD:#         for k, v in kwargs.iteritems():
#OLD:#             if k == 'samples':
#OLD:#                 s_indx.append(v)
#OLD:#             elif k == 'features':
#OLD:#                 f_indx.append(v)
#OLD:#             elif data_.has_key(k):
#OLD:#                 # so it is an attribute for samples
#OLD:#                 # XXX may be do it not only if __debug__
#OLD:#                 if __debug__: # and 'CHECK_DS_SELECT' in debug.active:
#OLD:#                     if not N.any([isinstance(v, cls) for cls in
#OLD:#                                   [list, tuple, slice, int]]):
#OLD:#                         raise ValueError, "Trying to specify selection for %s " \
#OLD:#                               "based on unsupported '%s'" % (k, v)
#OLD:#                 s_indx.append(self._getSampleIdsByAttr(v, attrib=k, sort=False))
#OLD:#             else:
#OLD:#                 raise ValueError, 'Keyword "%s" is not known, thus' \
#OLD:#                       'select() failed' % k
#OLD:# 
#OLD:#         def combine_indexes(indx, nelements):
#OLD:#             """Helper function: intersect selections given in indx
#OLD:# 
#OLD:#             :Parameters:
#OLD:#               indxs : list of lists or slices
#OLD:#                 selections of elements
#OLD:#               nelements : int
#OLD:#                 number of elements total for deriving indexes from slices
#OLD:#             """
#OLD:#             indx_sel = None                 # pure list of ids for selection
#OLD:#             for s in indx:
#OLD:#                 if isinstance(s, slice) or \
#OLD:#                    isinstance(s, N.ndarray) and s.dtype==bool:
#OLD:#                     # XXX there might be a better way than reconstructing the full
#OLD:#                     # index list. Also we are loosing ability to do simlpe slicing,
#OLD:#                     # ie w.o making a copy of the selected data
#OLD:#                     all_indexes = N.arange(nelements)
#OLD:#                     s = all_indexes[s]
#OLD:#                 elif not operator.isSequenceType(s):
#OLD:#                     s = [ s ]
#OLD:# 
#OLD:#                 if indx_sel is None:
#OLD:#                     indx_sel = Set(s)
#OLD:#                 else:
#OLD:#                     # To be consistent
#OLD:#                     #if not isinstance(indx_sel, Set):
#OLD:#                     #    indx_sel = Set(indx_sel)
#OLD:#                     indx_sel = indx_sel.intersection(s)
#OLD:# 
#OLD:#             # if we got Set -- convert
#OLD:#             if isinstance(indx_sel, Set):
#OLD:#                 indx_sel = list(indx_sel)
#OLD:# 
#OLD:#             # sort for the sake of sanity
#OLD:#             indx_sel.sort()
#OLD:# 
#OLD:#             return indx_sel
#OLD:# 
#OLD:#         # Select samples
#OLD:#         if len(s_indx) == 1 and isinstance(s_indx[0], slice) \
#OLD:#                and s_indx[0] == slice(None):
#OLD:#             # so no actual selection -- full slice
#OLD:#             s_indx = s_indx[0]
#OLD:#         else:
#OLD:#             # else - get indexes
#OLD:#             if len(s_indx) == 0:
#OLD:#                 s_indx = None
#OLD:#             else:
#OLD:#                 s_indx = combine_indexes(s_indx, self.nsamples)
#OLD:# 
#OLD:#         # Select features
#OLD:#         if len(f_indx):
#OLD:#             f_indx = combine_indexes(f_indx, self.nfeatures)
#OLD:#         else:
#OLD:#             f_indx = None
#OLD:# 
#OLD:#         return s_indx, f_indx
#OLD:# 
#OLD:# 
#OLD:#     def select(self, *args, **kwargs):
#OLD:#         """Universal selector
#OLD:# 
#OLD:#         WARNING: if you need to select duplicate samples
#OLD:#         (e.g. samples=[5,5]) or order of selected samples of features
#OLD:#         is important and has to be not ordered (e.g. samples=[3,2,1]),
#OLD:#         please use selectFeatures or selectSamples functions directly
#OLD:# 
#OLD:#         Examples:
#OLD:#           Mimique plain selectSamples::
#OLD:# 
#OLD:#             dataset.select([1,2,3])
#OLD:#             dataset[[1,2,3]]
#OLD:# 
#OLD:#           Mimique plain selectFeatures::
#OLD:# 
#OLD:#             dataset.select(slice(None), [1,2,3])
#OLD:#             dataset.select('all', [1,2,3])
#OLD:#             dataset[:, [1,2,3]]
#OLD:# 
#OLD:#           Mixed (select features and samples)::
#OLD:# 
#OLD:#             dataset.select([1,2,3], [1, 2])
#OLD:#             dataset[[1,2,3], [1, 2]]
#OLD:# 
#OLD:#           Select samples matching some attributes::
#OLD:# 
#OLD:#             dataset.select(labels=[1,2], chunks=[2,4])
#OLD:#             dataset.select('labels', [1,2], 'chunks', [2,4])
#OLD:#             dataset['labels', [1,2], 'chunks', [2,4]]
#OLD:# 
#OLD:#           Mixed -- out of first 100 samples, select only those with
#OLD:#           labels 1 or 2 and belonging to chunks 2 or 4, and select
#OLD:#           features 2 and 3::
#OLD:# 
#OLD:#             dataset.select(slice(0,100), [2,3], labels=[1,2], chunks=[2,4])
#OLD:#             dataset[:100, [2,3], 'labels', [1,2], 'chunks', [2,4]]
#OLD:# 
#OLD:#         """
#OLD:#         s_indx, f_indx = self.index(*args, **kwargs)
#OLD:# 
#OLD:#         # Select samples
#OLD:#         if s_indx == slice(None):
#OLD:#             # so no actual selection was requested among samples.
#OLD:#             # thus proceed with original dataset
#OLD:#             if __debug__:
#OLD:#                 debug('DS', 'in select() not selecting samples')
#OLD:#             ds = self
#OLD:#         else:
#OLD:#             # else do selection
#OLD:#             if __debug__:
#OLD:#                 debug('DS', 'in select() selecting samples given selections'
#OLD:#                       + str(s_indx))
#OLD:#             ds = self.selectSamples(s_indx)
#OLD:# 
#OLD:#         # Select features
#OLD:#         if f_indx is not None:
#OLD:#             if __debug__:
#OLD:#                 debug('DS', 'in select() selecting features given selections'
#OLD:#                       + str(f_indx))
#OLD:#             ds = ds.selectFeatures(f_indx)
#OLD:# 
#OLD:#         return ds
#OLD:# 
#OLD:# 
#OLD:# 
#OLD:#     def where(self, *args, **kwargs):
#OLD:#         """Obtain indexes of interesting samples/features. See select() for more information
#OLD:# 
#OLD:#         XXX somewhat obsoletes idsby...
#OLD:#         """
#OLD:#         s_indx, f_indx = self.index(*args, **kwargs)
#OLD:#         if s_indx is not None and f_indx is not None:
#OLD:#             return s_indx, f_indx
#OLD:#         elif s_indx is not None:
#OLD:#             return s_indx
#OLD:#         else:
#OLD:#             return f_indx
#OLD:# 
#OLD:# 
#OLD:#     def __getitem__(self, *args):
#OLD:#         """Convinience dataset parts selection
#OLD:# 
#OLD:#         See select for more information
#OLD:#         """
#OLD:#         # for cases like ['labels', 1]
#OLD:#         if len(args) == 1 and isinstance(args[0], tuple):
#OLD:#             args = args[0]
#OLD:# 
#OLD:#         args_, args = args, ()
#OLD:#         for a in args_:
#OLD:#             if isinstance(a, slice) and \
#OLD:#                    isinstance(a.start, basestring):
#OLD:#                     # for the constructs like ['labels':[1,2]]
#OLD:#                     if a.stop is None or a.step is not None:
#OLD:#                         raise ValueError, \
#OLD:#                               "Selection must look like ['chunks':[2,3]]"
#OLD:#                     args += (a.start, a.stop)
#OLD:#             else:
#OLD:#                 args += (a,)
#OLD:#         return self.select(*args)
#OLD:# 
#OLD:# 
#OLD:#     def permuteLabels(self, status, perchunk=True, assure_permute=False):
#OLD:#         """Permute the labels.
#OLD:# 
#OLD:#         TODO: rename status into something closer in semantics.
#OLD:# 
#OLD:#         :Parameters:
#OLD:#           status : bool
#OLD:#             Calling this method with set to True, the labels are
#OLD:#             permuted among all samples. If 'status' is False the
#OLD:#             original labels are restored.
#OLD:#           perchunk : bool
#OLD:#             If True permutation is limited to samples sharing the same
#OLD:#             chunk value. Therefore only the association of a certain
#OLD:#             sample with a label is permuted while keeping the absolute
#OLD:#             number of occurences of each label value within a certain
#OLD:#             chunk constant.
#OLD:#           assure_permute : bool
#OLD:#             If True, assures that labels are permutted, ie any one is
#OLD:#             different from the original one
#OLD:#         """
#OLD:#         # local bindings
#OLD:#         _data = self._data
#OLD:# 
#OLD:#         if len(self.uniquelabels)<2:
#OLD:#             raise RuntimeError, \
#OLD:#                   "Call to permuteLabels is bogus since there is insuficient" \
#OLD:#                   " number of labels: %s" % self.uniquelabels
#OLD:# 
#OLD:#         if not status:
#OLD:#             # restore originals
#OLD:#             if _data.get('origlabels', None) is None:
#OLD:#                 raise RuntimeError, 'Cannot restore labels. ' \
#OLD:#                                     'permuteLabels() has never been ' \
#OLD:#                                     'called with status == True.'
#OLD:#             self.labels = _data['origlabels']
#OLD:#             _data.pop('origlabels')
#OLD:#         else:
#OLD:#             # store orig labels, but only if not yet done, otherwise multiple
#OLD:#             # calls with status == True will destroy the original labels
#OLD:#             if not _data.has_key('origlabels') \
#OLD:#                 or _data['origlabels'] == None:
#OLD:#                 # bind old labels to origlabels
#OLD:#                 _data['origlabels'] = _data['labels']
#OLD:#                 # copy labels
#OLD:#                 _data['labels'] = copy.copy(_data['labels'])
#OLD:# 
#OLD:#             labels = _data['labels']
#OLD:#             # now scramble
#OLD:#             if perchunk:
#OLD:#                 for o in self.uniquechunks:
#OLD:#                     labels[self.chunks == o] = \
#OLD:#                         N.random.permutation(labels[self.chunks == o])
#OLD:#             else:
#OLD:#                 labels = N.random.permutation(labels)
#OLD:# 
#OLD:#             self.labels = labels
#OLD:# 
#OLD:#             if assure_permute:
#OLD:#                 if not (_data['labels'] != _data['origlabels']).any():
#OLD:#                     if not (assure_permute is True):
#OLD:#                         if assure_permute == 1:
#OLD:#                             raise RuntimeError, \
#OLD:#                                   "Cannot assure permutation of labels %s for " \
#OLD:#                                   "some reason with chunks %s and while " \
#OLD:#                                   "perchunk=%s . Should not happen" % \
#OLD:#                                   (self.labels, self.chunks, perchunk)
#OLD:#                     else:
#OLD:#                         assure_permute = 11 # make 10 attempts
#OLD:#                     if __debug__:
#OLD:#                         debug("DS",  "Recalling permute to assure different labels")
#OLD:#                     self.permuteLabels(status, perchunk=perchunk,
#OLD:#                                        assure_permute=assure_permute-1)
#OLD:# 
#OLD:# 
#OLD:#     def getRandomSamples(self, nperlabel):
#OLD:#         """Select a random set of samples.
#OLD:# 
#OLD:#         If 'nperlabel' is an integer value, the specified number of samples is
#OLD:#         randomly choosen from the group of samples sharing a unique label
#OLD:#         value ( total number of selected samples: nperlabel x len(uniquelabels).
#OLD:# 
#OLD:#         If 'nperlabel' is a list which's length has to match the number of
#OLD:#         unique label values. In this case 'nperlabel' specifies the number of
#OLD:#         samples that shall be selected from the samples with the corresponding
#OLD:#         label.
#OLD:# 
#OLD:#         The method returns a Dataset object containing the selected
#OLD:#         samples.
#OLD:#         """
#OLD:#         # if interger is given take this value for all classes
#OLD:#         if isinstance(nperlabel, int):
#OLD:#             nperlabel = [ nperlabel for i in self.uniquelabels ]
#OLD:# 
#OLD:#         sample = []
#OLD:#         # for each available class
#OLD:#         labels = self.labels
#OLD:#         for i, r in enumerate(self.uniquelabels):
#OLD:#             # get the list of pattern ids for this class
#OLD:#             sample += random.sample( (labels == r).nonzero()[0],
#OLD:#                                      nperlabel[i] )
#OLD:# 
#OLD:#         return self.selectSamples( sample )
#OLD:# 
#OLD:# 
#OLD:# #    def _setchunks(self, chunks):
#OLD:# #        """Sets chunks and recomputes uniquechunks
#OLD:# #        """
#OLD:# #        self._data['chunks'] = N.array(chunks)
#OLD:# #        self._dsattr['uniquechunks'] = None # None!since we might not need them
#OLD:# 
#OLD:# 
#OLD:#     def getNSamples( self ):
#OLD:#         """Currently available number of patterns.
#OLD:#         """
#OLD:#         return self._data['samples'].shape[0]
#OLD:# 
#OLD:# 
#OLD:#     def getNFeatures( self ):
#OLD:#         """Number of features per pattern.
#OLD:#         """
#OLD:#         return self._data['samples'].shape[1]
#OLD:# 
#OLD:# 
#OLD:#     def getLabelsMap(self):
#OLD:#         """Stored labels map (if any)
#OLD:#         """
#OLD:#         return self._dsattr.get('labels_map', None)
#OLD:# 
#OLD:# 
#OLD:#     def setLabelsMap(self, lm):
#OLD:#         """Set labels map.
#OLD:# 
#OLD:#         Checks for the validity of the mapping -- values should cover
#OLD:#         all existing labels in the dataset
#OLD:#         """
#OLD:#         values = Set(lm.values())
#OLD:#         labels = Set(self.uniquelabels)
#OLD:#         if not values.issuperset(labels):
#OLD:#             raise ValueError, \
#OLD:#                   "Provided mapping %s has some existing labels (out of %s) " \
#OLD:#                   "missing from mapping" % (list(values), list(labels))
#OLD:#         self._dsattr['labels_map'] = lm
#OLD:# 
#OLD:# 
#OLD:#     def setSamplesDType(self, dtype):
#OLD:#         """Set the data type of the samples array.
#OLD:#         """
#OLD:#         # local bindings
#OLD:#         _data = self._data
#OLD:# 
#OLD:#         if _data['samples'].dtype != dtype:
#OLD:#             _data['samples'] = _data['samples'].astype(dtype)
#OLD:# 
#OLD:# 
#OLD:#     def defineFeatureGroups(self, definition):
#OLD:#         """Assign `definition` to featuregroups
#OLD:# 
#OLD:#         XXX Feature-groups was not finished to be useful
#OLD:#         """
#OLD:#         if not len(definition) == self.nfeatures:
#OLD:#             raise ValueError, \
#OLD:#                   "Length of feature group definition %i " \
#OLD:#                   "does not match the number of features %i " \
#OLD:#                   % (len(definition), self.nfeatures)
#OLD:# 
#OLD:#         self._dsattr['featuregroups'] = N.array(definition)
#OLD:# 
#OLD:# 
#OLD:#     def convertFeatureIds2FeatureMask(self, ids):
#OLD:#         """Returns a boolean mask with all features in `ids` selected.
#OLD:# 
#OLD:#         :Parameters:
#OLD:#             ids: list or 1d array
#OLD:#                 To be selected features ids.
#OLD:# 
#OLD:#         :Returns:
#OLD:#             ndarray: dtype='bool'
#OLD:#                 All selected features are set to True; False otherwise.
#OLD:#         """
#OLD:#         fmask = N.repeat(False, self.nfeatures)
#OLD:#         fmask[ids] = True
#OLD:# 
#OLD:#         return fmask
#OLD:# 
#OLD:# 
#OLD:#     def convertFeatureMask2FeatureIds(self, mask):
#OLD:#         """Returns feature ids corresponding to non-zero elements in the mask.
#OLD:# 
#OLD:#         :Parameters:
#OLD:#             mask: 1d ndarray
#OLD:#                 Feature mask.
#OLD:# 
#OLD:#         :Returns:
#OLD:#             ndarray: integer
#OLD:#                 Ids of non-zero (non-False) mask elements.
#OLD:#         """
#OLD:#         return mask.nonzero()[0]
#OLD:# 
#OLD:# 
#OLD:#     @staticmethod
#OLD:#     def _checkCopyConstructorArgs(**kwargs):
#OLD:#         """Common sanity check for Dataset copy constructor calls."""
#OLD:#         # check if we have samples (somwhere)
#OLD:#         samples = None
#OLD:#         if kwargs.has_key('samples'):
#OLD:#             samples = kwargs['samples']
#OLD:#         if samples is None and kwargs.has_key('data') \
#OLD:#            and kwargs['data'].has_key('samples'):
#OLD:#             samples = kwargs['data']['samples']
#OLD:#         if samples is None:
#OLD:#             raise DatasetError, \
#OLD:#                   "`samples` must be provided to copy constructor call."
#OLD:# 
#OLD:#         if not len(samples.shape) == 2:
#OLD:#             raise DatasetError, \
#OLD:#                   "samples must be in 2D shape in copy constructor call."
#OLD:# 
#OLD:# 
#OLD:#     # read-only class properties
#OLD:#     nsamples        = property( fget=getNSamples )
#OLD:#     nfeatures       = property( fget=getNFeatures )
#OLD:#     labels_map      = property( fget=getLabelsMap, fset=setLabelsMap )
#OLD:# 
#OLD:# def datasetmethod(func):
#OLD:#     """Decorator to easily bind functions to a Dataset class
#OLD:#     """
#OLD:#     if __debug__:
#OLD:#         debug("DS_",  "Binding function %s to Dataset class" % func.func_name)
#OLD:# 
#OLD:#     # Bind the function
#OLD:#     setattr(Dataset, func.func_name, func)
#OLD:# 
#OLD:#     # return the original one
#OLD:#     return func
#OLD:# 
#OLD:# 
#OLD:# # Following attributes adherent to the basic dataset
#OLD:# Dataset._registerAttribute("samples", "_data", abbr='S', hasunique=False)
#OLD:# Dataset._registerAttribute("labels",  "_data", abbr='L', hasunique=True)
#OLD:# Dataset._registerAttribute("chunks",  "_data", abbr='C', hasunique=True)
#OLD:# # samples ids (already unique by definition)
#OLD:# Dataset._registerAttribute("origids",  "_data", abbr='I', hasunique=False)
#OLD:# 


