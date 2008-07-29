#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset container"""

__docformat__ = 'restructuredtext'

import operator
import random
import mvpa.misc.copy as copy
import numpy as N

from sets import Set

from mvpa.misc.exceptions import DatasetError
from mvpa.misc.support import idhash as idhash_
from mvpa.base.dochelpers import enhancedDocString

if __debug__:
    from mvpa.base import debug, warning

class Dataset(object):
    """*The* Dataset.

    This class provides a container to store all necessary data to
    perform MVPA analyses. These are the data samples, as well as the
    labels associated with the samples. Additionally, samples can be
    grouped into chunks.

    :Groups:
      - `Creators`: `__init__`, `selectFeatures`, `selectSamples`,
                    `applyMapper`
      - `Mutators`: `permuteLabels`

    Important: labels assumed to be immutable, i.e. noone should modify
    them externally by accessing indexed items, ie something like
    ``dataset.labels[1] += "_bad"`` should not be used. If a label has
    to be modified, full copy of labels should be obtained, operated on,
    and assigned back to the dataset, otherwise dataset.uniquelabels
    would not work.  The same applies to any other attribute which has
    corresponding unique* access property.

    """

    # static definition to track which unique attributes
    # have to be reset/recomputed whenever anything relevant
    # changes

    # unique{labels,chunks} become a part of dsattr
    _uniqueattributes = []
    """Unique attributes associated with the data"""

    _registeredattributes = []
    """Registered attributes (stored in _data)"""

    _requiredattributes = ['samples', 'labels']
    """Attributes which have to be provided to __init__, or otherwise
    no default values would be assumed and construction of the
    instance would fail"""

    def __init__(self,
                 # for copy constructor
                 data=None,
                 dsattr=None,
                 # automatic dtype conversion
                 dtype=None,
                 # new instances
                 samples=None,
                 labels=None,
                 chunks=None,
                 origids=None,
                 # flags
                 check_data=True,
                 copy_samples=False,
                 copy_data=True,
                 copy_dsattr=True):
        """Initialize dataset instance

        There are basically two different way to create a dataset:

        1. Create a new dataset from samples and sample attributes.  In
           this mode a two-dimensional `ndarray` has to be passed to the
           `samples` keyword argument and the corresponding samples
           attributes are provided via the `labels` and `chunks`
           arguments.

        2. Copy contructor mode
            The second way is used internally to perform quick coyping
            of datasets, e.g. when performing feature selection. In this
            mode and the two dictionaries (`data` and `dsattr`) are
            required. For performance reasons this mode bypasses most of
            the sanity check performed by the previous mode, as for
            internal operations data integrity is assumed.


        :Parameters:
          data : dict
            Dictionary with an arbitrary number of entries. The value for
            each key in the dict has to be an ndarray with the
            same length as the number of rows in the samples array.
            A special entry in this dictionary is 'samples', a 2d array
            (samples x features). A shallow copy is stored in the object.
          dsattr : dict
            Dictionary of dataset attributes. An arbitrary number of
            arbitrarily named and typed objects can be stored here. A
            shallow copy of the dictionary is stored in the object.
          dtype: type | None
            If None -- do not change data type if samples
            is an ndarray. Otherwise convert samples to dtype.


        :Keywords:
          samples : ndarray
            a 2d array (samples x features)
          labels
            array or scalar value defining labels for each samples
          chunks
            array or scalar value defining chunks for each sample

        Each of the Keywords arguments overwrites what is/might be
        already in the `data` container.

        """
        # see if data and dsattr are none, if so, make them empty dicts
        if data is None:
            data = {}
        if dsattr is None:
            dsattr = {}

        # initialize containers; default values are empty dicts
        # always make a shallow copy of what comes in, otherwise total chaos
        # is likely to happen soon
        if copy_data:
            # deep copy (cannot use copy.deepcopy, because samples is an
            # exception
            # but shallow copy first to get a shared version of the data in
            # any case
            lcl_data = data.copy()
            for k, v in data.iteritems():
                # skip copying samples if requested
                if k == 'samples' and not copy_samples:
                    continue
                lcl_data[k] = v.copy()
        else:
            # shallow copy
            # XXX? yoh: it might be better speed wise just assign dictionary
            #      without any shallow .copy
            lcl_data = data.copy()

        if copy_dsattr and len(dsattr)>0:
            # deep copy
            if __debug__:
                debug('DS', "Deep copying dsattr %s" % `dsattr`)
            lcl_dsattr = copy.deepcopy(dsattr)

        else:
            # shallow copy
            lcl_dsattr = copy.copy(dsattr)

        # has to be not private since otherwise derived methods
        # would have problem accessing it and _registerAttribute
        # would fail on lambda getters
        self._data = lcl_data
        """What makes a dataset."""

        self._dsattr = lcl_dsattr
        """Dataset attriibutes."""

        # store samples (and possibly transform/reshape/retype them)
        if not samples == None:
            if __debug__:
                if self._data.has_key('samples'):
                    debug('DS',
                          "`Data` dict has `samples` (%s) but there is also" +
                          " __init__ parameter `samples` which overrides " +
                          " stored in `data`" % (`self._data['samples'].shape`))
            self._data['samples'] = self._shapeSamples(samples, dtype,
                                                        copy_samples)

        # TODO? we might want to have the same logic for chunks and labels
        #       ie if no labels present -- assign arange
        #   MH: don't think this is necessary -- or is there a use case?
        # labels
        if not labels == None:
            if __debug__:
                if self._data.has_key('labels'):
                    debug('DS',
                          "`Data` dict has `labels` (%s) but there is also" +
                          " __init__ parameter `labels` which overrides " +
                          " stored in `data`" % (`self._data['labels']`))
            if self._data.has_key('samples'):
                self._data['labels'] = \
                    self._expandSampleAttribute(labels, 'labels')

        # check if we got all required attributes
        for attr in self._requiredattributes:
            if not self._data.has_key(attr):
                raise DatasetError, \
                      "Attribute %s is required to initialize dataset" % \
                      attr

        # chunks
        if not chunks == None:
            self._data['chunks'] = \
                self._expandSampleAttribute(chunks, 'chunks')
        elif not self._data.has_key('chunks'):
            # if no chunk information is given assume that every pattern
            # is its own chunk
            self._data['chunks'] = N.arange(self.nsamples)

        # samples origids
        if not origids is None:
            # simply assign if provided
            self._data['origids'] = origids
        elif not self._data.has_key('origids'):
            # otherwise contruct unqiue ones
            self._data['origids'] = N.arange(len(self._data['labels']))
        else:
            # assume origids have been specified already (copy constructor
            # mode) leave them as they are, e.g. to make origids survive
            # selectSamples()
            pass

        # Initialize attributes which are registered but were not setup
        for attr in self._registeredattributes:
            if not self._data.has_key(attr):
                if __debug__:
                    debug("DS", "Initializing attribute %s" % attr)
                self._data[attr] = N.zeros(self.nsamples)

        if check_data:
            self._checkData()

        # lazy computation of unique members
        #self._resetallunique('_dsattr', self._dsattr)

        # Michael: we cannot do this conditional here. When selectSamples()
        # removes a whole data chunk the uniquechunks values will be invalid.
        # Same applies to labels of course.
        if not labels is None or not chunks is None:
            # for a speed up to don't go through all uniqueattributes
            # when no need
            self._dsattr['__uniquereseted'] = False
            self._resetallunique(force=True)


    __doc__ = enhancedDocString('Dataset', locals())


    @property
    def idhash(self):
        """To verify if dataset is in the same state as when smth else was done

        Like if classifier was trained on the same dataset as in question"""

        res = idhash_(self._data)

        # we cannot count on the order the values in the dict will show up
        # with `self._data.value()` and since idhash will be order-dependent
        # we have to make it deterministic
        keys = self._data.keys()
        keys.sort()
        for k in keys:
            res += idhash_(self._data[k])
        return res


    def _resetallunique(self, force=False):
        """Set to None all unique* attributes of corresponding dictionary
        """

        if not force and self._dsattr['__uniquereseted']:
            return

        # I guess we better checked if dictname is known  but...
        for k in self._uniqueattributes:
            if __debug__:
                debug("DS_", "Reset attribute %s for dataset %s"
                      % (k,
                         self.summary(uniq=False, idhash=False, stats=False)))
            self._dsattr[k] = None
        self._dsattr['__uniquereseted'] = True


    def _getuniqueattr(self, attrib, dict_):
        """Provide common facility to return unique attributes

        XXX `dict_` can be simply replaced now with self._dsattr
        """
        if not self._dsattr.has_key(attrib) or self._dsattr[attrib] is None:
            if __debug__ and 'DS_' in debug.active:
                debug("DS_", "Recomputing unique set for attrib %s within %s" %
                      (attrib, self.summary(uniq=False, stats=False)))
            # uff... might come up with better strategy to keep relevant
            # attribute name
            self._dsattr[attrib] = N.unique( N.asanyarray(dict_[attrib[6:]]) )
            assert(not self._dsattr[attrib] is None)
            self._dsattr['__uniquereseted'] = False

        return self._dsattr[attrib]


    def _setdataattr(self, attrib, value):
        """Provide common facility to set attributes

        """
        if len(value) != self.nsamples:
            raise ValueError, \
                  "Provided %s have %d entries while there is %d samples" % \
                  (attrib, len(value), self.nsamples)
        self._data[attrib] = N.asarray(value)
        uniqueattr = "unique" + attrib

        if self._dsattr.has_key(uniqueattr):
            self._dsattr[uniqueattr] = None


    def _getNSamplesPerAttr( self, attrib='labels' ):
        """Returns the number of samples per unique label.
        """
        # XXX hardcoded dict_=self._data.... might be in self._dsattr
        uniqueattr = self._getuniqueattr(attrib="unique" + attrib,
                                         dict_=self._data)

        # use dictionary to cope with arbitrary labels
        result = dict(zip(uniqueattr, [ 0 ] * len(uniqueattr)))
        for l in self._data[attrib]:
            result[l] += 1

        # XXX only return values to mimic the old interface but we might want
        # to return the full dict instead
        # return result
        return result


    def _getSampleIdsByAttr(self, values, attrib="labels"):
        """Return indecies of samples given a list of attributes
        """

        if not operator.isSequenceType(values) \
               or isinstance(values, basestring):
            values = [ values ]

        # TODO: compare to plain for loop through the labels
        #       on a real data example
        sel = N.array([], dtype=N.int16)
        for value in values:
            sel = N.concatenate((
                sel, N.where(self._data[attrib]==value)[0]))

        # place samples in the right order
        sel.sort()

        return sel


    def idsonboundaries(self, prior=0, post=0,
                        attributes_to_track=['labels', 'chunks'],
                        affected_labels=None,
                        revert=False):
        """Find samples which are on the boundaries of the blocks

        Such samples might need to be removed.  By default (with
        prior=0, post=0) ids of the first samples in a 'block' are
        reported

        :Parameters:
          prior : int
            how many samples prior to transition sample to include
          post : int
            how many samples post the transition sample to include
          attributes_to_track : list of basestring
            which attributes to track to decide on the boundary condition
          affected_labels : list of basestring
            for which labels to perform selection. If None - for all
          revert : bool
            either to revert the meaning and provide ids of samples which are found
            to not to be boundary samples
        """
        lastseen = [None for attr in attributes_to_track]
        transitions = []
        nsamples = self.nsamples
        for i in xrange(nsamples):
            current = [self._data[attr][i] for attr in attributes_to_track]
            if lastseen != current:
                # transition point
                new_transitions = range(max(0, i-prior),
                                        min(nsamples-1, i+post)+1)
                if affected_labels is not None:
                    new_transitions = filter(lambda i: self.labels[i] in affected_labels,
                                             new_transitions)
                transitions += new_transitions
                lastseen = current

        transitions = Set(transitions)
        if revert:
            transitions = Set(range(nsamples)).difference(transitions)

        # postprocess
        transitions = N.array(list(transitions))
        transitions.sort()
        return list(transitions)


    def _shapeSamples(self, samples, dtype, copy):
        """Adapt different kinds of samples

        Handle all possible input value for 'samples' and tranform
        them into a 2d (samples x feature) representation.
        """
        # put samples array into correct shape
        # 1d arrays or simple sequences are assumed to be a single pattern
        if (not isinstance(samples, N.ndarray)):
            # it is safe to provide dtype which defaults to None,
            # when N would choose appropriate dtype automagically
            samples = N.array(samples, ndmin=2, dtype=dtype, copy=copy)
        else:
            if samples.ndim < 2 \
                   or (not dtype is None and dtype != samples.dtype):
                if dtype is None:
                    dtype = samples.dtype
                samples = N.array(samples, ndmin=2, dtype=dtype, copy=copy)
            elif copy:
                samples = samples.copy()

        # only samples x features matrices are supported
        if len(samples.shape) > 2:
            raise DatasetError, "Only (samples x features) -> 2d sample " \
                            + "are supported (got %s shape of samples)." \
                            % (`samples.shape`) \
                            +" Consider MappedDataset if applicable."

        return samples


    def _checkData(self):
        """Checks `_data` members to have the same # of samples.
        """
        #
        # XXX: Maybe just run this under __debug__ and remove the `check_data`
        #      from the constructor, which is too complicated anyway?
        #
        for k, v in self._data.iteritems():
            if not len(v) == self.nsamples:
                raise DatasetError, \
                      "Length of sample attribute '%s' [%i] does not " \
                      "match the number of samples in the dataset [%i]." \
                      % (k, len(v), self.nsamples)

        # check for unique origids
        uniques = N.unique(self._data['origids'])
        uniques.sort()
        # need to copy to prevent sorting the original array
        sorted_ids = self._data['origids'].copy()
        sorted_ids.sort()

        if not (uniques == sorted_ids).all():
            raise DatasetError, "Samples IDs are not unique."


    def _expandSampleAttribute(self, attr, attr_name):
        """If a sample attribute is given as a scalar expand/repeat it to a
        length matching the number of samples in the dataset.
        """
        try:
            if len(attr) != self.nsamples:
                raise DatasetError, \
                      "Length of sample attribute '%s' [%d]" \
                      % (attr_name, len(attr)) \
                      + " has to match the number of samples" \
                      + " [%d]." % self.nsamples
            # store the sequence as array
            return N.array(attr)

        except TypeError:
            # make sequence of identical value matching the number of
            # samples
            return N.repeat(attr, self.nsamples)


    @classmethod
    def _registerAttribute(cls, key, dictname="_data", hasunique=False):
        """Register an attribute for any Dataset class.

        Creates property assigning getters/setters depending on the
        availability of corresponding _get, _set functions.
        """
        classdict = cls.__dict__
        if not classdict.has_key(key):
            if __debug__:
                debug("DS", "Registering new attribute %s" % key)
            # define get function and use corresponding
            # _getATTR if such defined
            getter = '_get%s' % key
            if classdict.has_key(getter):
                getter =  '%s.%s' % (cls.__name__, getter)
            else:
                getter = "lambda x: x.%s['%s']" % (dictname, key)

            # define set function and use corresponding
            # _setATTR if such defined
            setter = '_set%s' % key
            if classdict.has_key(setter):
                setter =  '%s.%s' % (cls.__name__, setter)
            elif dictname=="_data":
                setter = "lambda self,x: self._setdataattr" + \
                         "(attrib='%s', value=x)" % (key)
            else:
                setter = None

            if __debug__:
                debug("DS", "Registering new property %s.%s" %
                      (cls.__name__, key))
            exec "%s.%s = property(fget=%s, fset=%s)"  % \
                 (cls.__name__, key, getter, setter)

            if hasunique:
                uniquekey = "unique%s" % key
                getter = '_get%s' % uniquekey
                if classdict.has_key(getter):
                    getter = '%s.%s' % (cls.__name__, getter)
                else:
                    getter = "lambda x: x._getuniqueattr" + \
                            "(attrib='%s', dict_=x.%s)" % (uniquekey, dictname)

                if __debug__:
                    debug("DS", "Registering new property %s.%s" %
                          (cls.__name__, uniquekey))

                exec "%s.%s = property(fget=%s)" % \
                     (cls.__name__, uniquekey, getter)

                # create samplesper<ATTR> properties
                sampleskey = "samplesper%s" % key[:-1] # remove ending 's' XXX
                if __debug__:
                    debug("DS", "Registering new property %s.%s" %
                          (cls.__name__, sampleskey))

                exec "%s.%s = property(fget=%s)" % \
                     (cls.__name__, sampleskey,
                      "lambda x: x._getNSamplesPerAttr(attrib='%s')" % key)

                cls._uniqueattributes.append(uniquekey)

                # create idsby<ATTR> properties
                sampleskey = "idsby%s" % key # remove ending 's' XXX
                if __debug__:
                    debug("DS", "Registering new property %s.%s" %
                          (cls.__name__, sampleskey))

                exec "%s.%s = %s" % (cls.__name__, sampleskey,
                      "lambda self, x: " +
                      "self._getSampleIdsByAttr(x,attrib='%s')" % key)

                cls._uniqueattributes.append(uniquekey)

            cls._registeredattributes.append(key)
        elif __debug__:
            warning('Trying to reregister attribute `%s`. For now ' % key +
                    'such capability is not present')


    def __str__(self):
        """String summary over the object
        """
        return self.summary(uniq=True,
                            idhash=__debug__ and ('DS_ID' in debug.active),
                            stats=__debug__ and ('DS_STATS' in debug.active),
                            )


    def __repr__(self):
        return "<%s>" % str(self)


    def summary(self, uniq=True, stats=True, idhash=False):
        """String summary over the object

        :Parameters:
          uniq : bool
             include summary over data attributes which have unique
          idhash : bool
             include idhash value for dataset and samples
          stats : bool
             include some basic statistics (mean, std, var) over dataset samples
        """
        if idhash:
            idhash_ds = "{%s}" % self.idhash
            idhash_samples = "{%s}" % idhash_(self.samples)
        else:
            idhash_ds = ""
            idhash_samples = ""

        s = """Dataset %s/ %s %d%s x %d""" % \
            (idhash_ds, self.samples.dtype,
             self.nsamples, idhash_samples, self.nfeatures)

        if uniq:
            s +=  " uniq:"
            for uattr in self._dsattr.keys():
                if not uattr.startswith("unique"):
                    continue
                attr = uattr[6:]
                try:
                    value = self._getuniqueattr(attrib=uattr,
                                                dict_=self._data)
                    s += " %d %s" % (len(value), attr)
                except:
                    pass

        if stats:
            # TODO -- avg per chunk?
            s += " stats: mean=%g std=%g var=%g min=%g max=%g" % \
                 (N.mean(self.samples), N.std(self.samples),
                  N.var(self.samples), N.min(self.samples), N.max(self.samples))
        return s


    def __iadd__( self, other ):
        """Merge the samples of one Dataset object to another (in-place).

        No dataset attributes will be merged! Additionally, a new set of
        unique `origids` will be generated.
        """
        if not self.nfeatures == other.nfeatures:
            raise DatasetError, "Cannot add Dataset, because the number of " \
                                "feature do not match."

        # concatenate all sample attributes
        for k, v in self._data.iteritems():
            if k == 'origids':
                # special case samples origids: for now just regenerate unique
                # ones could also check if concatenation is unique, but it
                # would be costly performance-wise
                self._data[k] = N.arange(len(v) + len(other._data[k]))
            else:
                self._data[k] = N.concatenate((v, other._data[k]), axis=0)

        # might be more sophisticated but for now just reset -- it is safer ;)
        self._resetallunique()

        return self


    def __add__( self, other ):
        """Merge the samples two Dataset objects.

        All data of both datasets is copied, concatenated and a new Dataset is
        returned.

        NOTE: This can be a costly operation (both memory and time). If
        performance is important consider the '+=' operator.
        """
        # create a new object of the same type it is now and NOT onyl Dataset
        out = super(Dataset, self).__new__(self.__class__)

        # now init it: to make it work all Dataset contructors have to accept
        # Class(data=Dict, dsattr=Dict)
        out.__init__(data=self._data,
                     dsattr=self._dsattr,
                     copy_samples=True,
                     copy_data=True,
                     copy_dsattr=True)

        out += other

        return out


    def selectFeatures(self, ids, sort=True):
        """Select a number of features from the current set.

        :Parameters:
          ids
            iterable container to select ids
          sort : bool
            if to sort Ids. Order matters and `selectFeatures` assumes
            incremental order. If not such, in non-optimized code
            selectFeatures would verify the order and sort

        Returns a new Dataset object with a view of the original
        samples array (no copying is performed).

        WARNING: The order of ids determines the order of features in
        the returned dataset. This might be useful sometimes, but can
        also cause major headaches! Order would is verified when
        running in non-optimized code (if __debug__)
        """
        # XXX set sort default to True, now sorting has to be explicitely
        # disabled and warning is not necessary anymore
        if sort:
            ids.sort()
        elif __debug__ and 'CHECK_DS_SORTED' in debug.active:
            from mvpa.misc.support import isSorted
            if not isSorted(ids):
                warning("IDs for selectFeatures must be provided " +
                       "in sorted order, otherwise major headache might occur")

        # shallow-copy all stuff from current data dict
        new_data = self._data.copy()

        # assign the selected features -- data is still shared with
        # current dataset
        new_data['samples'] = self._data['samples'][:, ids]

        # create a new object of the same type it is now and NOT onyl Dataset
        dataset = super(Dataset, self).__new__(self.__class__)

        # now init it: to make it work all Dataset contructors have to accept
        # Class(data=Dict, dsattr=Dict)
        dataset.__init__(data=new_data,
                         dsattr=self._dsattr,
                         check_data=False,
                         copy_samples=False,
                         copy_data=False,
                         copy_dsattr=False
                         )

        return dataset


    def applyMapper(self, featuresmapper=None, samplesmapper=None):
        """Obtain new dataset by applying mappers over features and/or samples.

        :Parameters:
          featuresmapper : Mapper
            `Mapper` to somehow transform each sample's features
          samplesmapper : Mapper
            `Mapper` to transform each feature across samples

        WARNING: At the moment, handling of samplesmapper is not yet
        implemented since there were no real use case.

        TODO: selectFeatures is pretty much applyMapper(featuresmapper=MaskMapper(...))
        """

        # shallow-copy all stuff from current data dict
        new_data = self._data.copy()

        # apply mappers

        if samplesmapper:
            raise NotImplementedError

        if featuresmapper:
            if __debug__:
                debug("DS", "Applying featuresmapper %s" % `featuresmapper` +
                      " to samples of dataset `%s`" % `self`)
            new_data['samples'] = featuresmapper.forward(self._data['samples'])

        # create a new object of the same type it is now and NOT only Dataset
        dataset = super(Dataset, self).__new__(self.__class__)

        # now init it: to make it work all Dataset contructors have to accept
        # Class(data=Dict, dsattr=Dict)
        dataset.__init__(data=new_data,
                         dsattr=self._dsattr,
                         check_data=False,
                         copy_samples=False,
                         copy_data=False,
                         copy_dsattr=False
                         )

        return dataset


    def selectSamples(self, ids):
        """Choose a subset of samples defined by samples IDs.

        Returns a new dataset object containing the selected sample
        subset.

        TODO: yoh, we might need to sort the mask if the mask is a
        list of ids and is not ordered. Clarify with Michael what is
        our intent here!
        """
        # without having a sequence a index the masked sample array would
        # loose its 2d layout
        if not operator.isSequenceType( ids ):
            ids = [ids]
        # TODO: Reconsider crafting a slice if it can be done to don't copy
        #       the data
        #try:
        #    minmask = min(mask)
        #    maxmask = max(mask)
        #except:
        #    minmask = min(map(int,mask))
        #    maxmask = max(map(int,mask))
        # lets see if we could get it done with cheap view/slice
        #(minmask, maxmask) != (0, 1) and \
        #if len(mask) > 2 and \
        #       N.array([N.arange(minmask, maxmask+1) == N.array(mask)]).all():
        #    slice_ = slice(minmask, maxmask+1)
        #    if __debug__:
        #        debug("DS", "We can and do convert mask %s into splice %s" %
        #              (mask, slice_))
        #    mask = slice_
        # mask all sample attributes
        data = {}
        for k, v in self._data.iteritems():
            data[k] = v[ids, ]

        # create a new object of the same type it is now and NOT onyl Dataset
        dataset = super(Dataset, self).__new__(self.__class__)

        # now init it: to make it work all Dataset contructors have to accept
        # Class(data=Dict, dsattr=Dict)
        dataset.__init__(data=data,
                         dsattr=self._dsattr,
                         check_data=False,
                         copy_samples=False,
                         copy_data=False,
                         copy_dsattr=False)

        dataset._resetallunique(force=True)
        return dataset


    def permuteLabels(self, status, perchunk=True, assure_permute=False):
        """Permute the labels.

        TODO: rename status into something closer in semantics.

        :Parameters:
          status : bool
            Calling this method with set to True, the labels are
            permuted among all samples. If 'status' is False the
            original labels are restored.
          perchunk : bool
            If True permutation is limited to samples sharing the same
            chunk value. Therefore only the association of a certain
            sample with a label is permuted while keeping the absolute
            number of occurences of each label value within a certain
            chunk constant.
          assure_permute : bool
            If True, assures that labels are permutted, ie any one is
            different from the original one
        """
        if len(self.uniquelabels)<2:
            raise RuntimeError, \
                  "Call to permuteLabels is bogus since there is insuficient" \
                  " number of labels: %s" % self.uniquelabels

        if not status:
            # restore originals
            if self._data.get('origlabels', None) is None:
                raise RuntimeError, 'Cannot restore labels. ' \
                                    'permuteLabels() has never been ' \
                                    'called with status == True.'
            self.labels = self._data['origlabels']
            self._data.pop('origlabels')
        else:
            # store orig labels, but only if not yet done, otherwise multiple
            # calls with status == True will destroy the original labels
            if not self._data.has_key('origlabels') \
                or self._data['origlabels'] == None:
                # bind old labels to origlabels
                self._data['origlabels'] = self._data['labels']
                # copy labels
                self._data['labels'] = copy.copy(self._data['labels'])

            labels = self._data['labels']
            # now scramble
            if perchunk:
                for o in self.uniquechunks:
                    labels[self.chunks == o] = \
                        N.random.permutation(labels[self.chunks == o])
            else:
                labels = N.random.permutation(labels)

            self.labels = labels

            if assure_permute:
                if not (self._data['labels'] != self._data['origlabels']).any():
                    if not (assure_permute is True):
                        if assure_permute == 1:
                            raise RuntimeError, \
                                  "Cannot assure permutation of labels %s for " \
                                  "some reason with chunks %s and while " \
                                  "perchunk=%s . Should not happen" % \
                                  (self.labels, self.chunks, perchunk)
                    else:
                        assure_permute = 11 # make 10 attempts
                    if __debug__:
                        debug("DS",  "Recalling permute to assure different labels")
                    self.permuteLabels(status, perchunk=perchunk,
                                       assure_permute=assure_permute-1)


    def getRandomSamples(self, nperlabel):
        """Select a random set of samples.

        If 'nperlabel' is an integer value, the specified number of samples is
        randomly choosen from the group of samples sharing a unique label
        value ( total number of selected samples: nperlabel x len(uniquelabels).

        If 'nperlabel' is a list which's length has to match the number of
        unique label values. In this case 'nperlabel' specifies the number of
        samples that shall be selected from the samples with the corresponding
        label.

        The method returns a Dataset object containing the selected
        samples.
        """
        # if interger is given take this value for all classes
        if isinstance(nperlabel, int):
            nperlabel = [ nperlabel for i in self.uniquelabels ]

        sample = []
        # for each available class
        for i, r in enumerate(self.uniquelabels):
            # get the list of pattern ids for this class
            sample += random.sample( (self.labels == r).nonzero()[0],
                                     nperlabel[i] )

        return self.selectSamples( sample )


#    def _setchunks(self, chunks):
#        """Sets chunks and recomputes uniquechunks
#        """
#        self._data['chunks'] = N.array(chunks)
#        self._dsattr['uniquechunks'] = None # None!since we might not need them


    def getNSamples( self ):
        """Currently available number of patterns.
        """
        return self._data['samples'].shape[0]


    def getNFeatures( self ):
        """Number of features per pattern.
        """
        return self._data['samples'].shape[1]


    def setSamplesDType(self, dtype):
        """Set the data type of the samples array.
        """
        if self._data['samples'].dtype != dtype:
            self._data['samples'] = self._data['samples'].astype(dtype)


    def convertFeatureIds2FeatureMask(self, ids):
        """Returns a boolean mask with all features in `ids` selected.

        :Parameters:
            ids: list or 1d array
                To be selected features ids.

        :Returns:
            ndarray: dtype='bool'
                All selected features are set to True; False otherwise.
        """
        fmask = N.repeat(False, self.nfeatures)
        fmask[ids] = True

        return fmask


    def convertFeatureMask2FeatureIds(self, mask):
        """Returns feature ids corresponding to non-zero elements in the mask.

        :Parameters:
            mask: 1d ndarray
                Feature mask.

        :Returns:
            ndarray: integer
                Ids of non-zero (non-False) mask elements.
        """
        return mask.nonzero()[0]


    # read-only class properties
    nsamples        = property( fget=getNSamples )
    nfeatures       = property( fget=getNFeatures )



# Following attributes adherent to the basic dataset
Dataset._registerAttribute("samples", "_data", hasunique=False)
Dataset._registerAttribute("labels",  "_data", hasunique=True)
Dataset._registerAttribute("chunks",  "_data", hasunique=True)
# samples ids (already unique by definition)
Dataset._registerAttribute("origids",  "_data", hasunique=False)
