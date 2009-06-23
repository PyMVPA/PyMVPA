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

import operator
import random
import mvpa.support.copy as copy
import numpy as N

from sets import Set

# Sooner or later Dataset would become ClassWithCollections as well, but for
# now just an object -- thus commenting out tentative changes
#
#XXX from mvpa.misc.state import ClassWithCollections, SampleAttribute

from mvpa.misc.exceptions import DatasetError
from mvpa.misc.support import idhash as idhash_
from mvpa.base.dochelpers import enhancedDocString, table2string

if __debug__:
    from mvpa.base import debug, warning

    def _validate_indexes_uniq_sorted(seq, fname, item):
        """Helper function to validate that seq contains unique sorted values
        """
        if operator.isSequenceType(seq):
            seq_unique = N.unique(seq)
            if len(seq) != len(seq_unique):
                warning("%s() operates only with indexes for %s without"
                        " repetitions. Repetitions were removed."
                        % (fname, item))
            if N.any(N.sort(seq) != seq_unique):
                warning("%s() does not guarantee the original order"
                        " of selected %ss. Use selectSamples() and "
                        " selectFeatures(sort=False) instead" % (fname, item))


#XXX class Dataset(ClassWithCollections):
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
    # XXX Notes about migration to use Collections to store data and
    # attributes for samples, features, and dataset itself:

    # changes:
    #   _data  ->  s_attr collection (samples attributes)
    #   _dsattr -> ds_attr collection
    #              f_attr collection (features attributes)

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

    #XXX _ATTRIBUTE_COLLECTIONS = [ 's_attr', 'f_attr', 'ds_attr' ]
    #XXX """Assure those 3 collections to be present in all datasets"""
    #XXX
    #XXX samples__ = SampleAttribute(doc="Samples data. 0th index is time", hasunique=False) # XXX
    #XXX labels__ = SampleAttribute(doc="Labels for the samples", hasunique=True)
    #XXX chunks__ = SampleAttribute(doc="Chunk identities for the samples", hasunique=True)
    #XXX # samples ids (already unique by definition)
    #XXX origids__ = SampleAttribute(doc="Chunk identities for the samples", hasunique=False)

    def __init__(self,
                 # for copy constructor
                 data=None,
                 dsattr=None,
                 # automatic dtype conversion
                 dtype=None,
                 # new instances
                 samples=None,
                 labels=None,
                 labels_map=None,
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
            2d array (samples x features)
          labels
            An array or scalar value defining labels for each samples
          labels_map : None or bool or dict
            Map from labels into literal names. If is None or True,
            the mapping is computed, from labels which must be literal.
            If is False, no mapping is computed. If dict -- mapping is
            verified and taken, labels get remapped. Dict must map
            literal -> number
          chunks
            An array or scalar value defining chunks for each sample

        Each of the Keywords arguments overwrites what is/might be
        already in the `data` container.

        """

        #XXX ClassWithCollections.__init__(self)

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
                if lcl_data.has_key('samples'):
                    debug('DS',
                          "`Data` dict has `samples` (%s) but there is also" \
                          " __init__ parameter `samples` which overrides " \
                          " stored in `data`" % (`lcl_data['samples'].shape`))
            lcl_data['samples'] = self._shapeSamples(samples, dtype,
                                                     copy_samples)

        # TODO? we might want to have the same logic for chunks and labels
        #       ie if no labels present -- assign arange
        #   MH: don't think this is necessary -- or is there a use case?
        # labels
        if not labels == None:
            if __debug__:
                if lcl_data.has_key('labels'):
                    debug('DS',
                          "`Data` dict has `labels` (%s) but there is also" +
                          " __init__ parameter `labels` which overrides " +
                          " stored in `data`" % (`lcl_data['labels']`))
            if lcl_data.has_key('samples'):
                lcl_data['labels'] = \
                    self._expandSampleAttribute(labels, 'labels')

        # check if we got all required attributes
        for attr in self._requiredattributes:
            if not lcl_data.has_key(attr):
                raise DatasetError, \
                      "Attribute %s is required to initialize dataset" % \
                      attr

        nsamples = self.nsamples

        # chunks
        if not chunks == None:
            lcl_data['chunks'] = \
                self._expandSampleAttribute(chunks, 'chunks')
        elif not lcl_data.has_key('chunks'):
            # if no chunk information is given assume that every pattern
            # is its own chunk
            lcl_data['chunks'] = N.arange(nsamples)

        # samples origids
        if not origids is None:
            # simply assign if provided
            lcl_data['origids'] = origids
        elif not lcl_data.has_key('origids'):
            # otherwise contruct unqiue ones
            lcl_data['origids'] = N.arange(len(lcl_data['labels']))
        else:
            # assume origids have been specified already (copy constructor
            # mode) leave them as they are, e.g. to make origids survive
            # selectSamples()
            pass

        # Initialize attributes which are registered but were not setup
        for attr in self._registeredattributes:
            if not lcl_data.has_key(attr):
                if __debug__:
                    debug("DS", "Initializing attribute %s" % attr)
                lcl_data[attr] = N.zeros(nsamples)

        # labels_map
        labels_ = N.asarray(lcl_data['labels'])
        labels_map_known = lcl_dsattr.has_key('labels_map')
        if labels_map is True:
            # need to composte labels_map
            if labels_.dtype.char == 'S' or not labels_map_known:
                # Create mapping
                ulabels = list(Set(labels_))
                ulabels.sort()
                labels_map = dict([ (x[1], x[0]) for x in enumerate(ulabels) ])
                if __debug__:
                    debug('DS', 'Mapping for the labels computed to be %s'
                          % labels_map)
            else:
                if __debug__:
                    debug('DS', 'Mapping of labels was requested but labels '
                          'are not strings. Skipped')
                labels_map = None
            pass
        elif labels_map is False:
            labels_map = None

        if isinstance(labels_map, dict):
            if labels_map_known:
                if __debug__:
                    debug('DS',
                          "`dsattr` dict has `labels_map` (%s) but there is also" \
                          " __init__ parameter `labels_map` (%s) which overrides " \
                          " stored in `dsattr`" % (lcl_dsattr['labels_map'], labels_map))

            lcl_dsattr['labels_map'] = labels_map
            # map labels if needed (if strings or was explicitely requested)
            if labels_.dtype.char == 'S' or not labels_map_known:
                if __debug__:
                    debug('DS_', "Remapping labels using mapping %s" % labels_map)
                # need to remap
                # !!! N.array is important here
                try:
                    lcl_data['labels'] = N.array(
                        [labels_map[x] for x in lcl_data['labels']])
                except KeyError, e:
                    raise ValueError, "Provided labels_map %s is insufficient " \
                          "to map all the labels. Mapping for label %s is " \
                          "missing" % (labels_map, e)

        elif not lcl_dsattr.has_key('labels_map'):
            lcl_dsattr['labels_map'] = labels_map
        elif __debug__:
            debug('DS_', 'Not overriding labels_map in dsattr since it has one')

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
            lcl_dsattr['__uniquereseted'] = False
            self._resetallunique(force=True)


    __doc__ = enhancedDocString('Dataset', locals())


    @property
    def idhash(self):
        """To verify if dataset is in the same state as when smth else was done

        Like if classifier was trained on the same dataset as in question"""

        _data = self._data
        res = idhash_(_data)

        # we cannot count on the order the values in the dict will show up
        # with `self._data.value()` and since idhash will be order-dependent
        # we have to make it deterministic
        keys = _data.keys()
        keys.sort()
        for k in keys:
            res += idhash_(_data[k])
        return res


    def _resetallunique(self, force=False):
        """Set to None all unique* attributes of corresponding dictionary
        """
        _dsattr = self._dsattr

        if not force and _dsattr['__uniquereseted']:
            return

        _uniqueattributes = self._uniqueattributes

        if __debug__ and "DS_" in debug.active:
            debug("DS_", "Reseting all attributes %s for dataset %s"
                  % (_uniqueattributes,
                     self.summary(uniq=False, idhash=False,
                                  stats=False, lstats=False)))

        # I guess we better checked if dictname is known  but...
        for k in _uniqueattributes:
            _dsattr[k] = None
        _dsattr['__uniquereseted'] = True


    def _getuniqueattr(self, attrib, dict_):
        """Provide common facility to return unique attributes

        XXX `dict_` can be simply replaced now with self._dsattr
        """

        # local bindings
        _dsattr = self._dsattr

        if not _dsattr.has_key(attrib) or _dsattr[attrib] is None:
            if __debug__ and 'DS_' in debug.active:
                debug("DS_", "Recomputing unique set for attrib %s within %s" %
                      (attrib, self.summary(uniq=False,
                                            stats=False, lstats=False)))
            # uff... might come up with better strategy to keep relevant
            # attribute name
            _dsattr[attrib] = N.unique( N.asanyarray(dict_[attrib[6:]]) )
            assert(not _dsattr[attrib] is None)
            _dsattr['__uniquereseted'] = False

        return _dsattr[attrib]


    def _setdataattr(self, attrib, value):
        """Provide common facility to set attributes

        """
        if len(value) != self.nsamples:
            raise ValueError, \
                  "Provided %s have %d entries while there is %d samples" % \
                  (attrib, len(value), self.nsamples)
        self._data[attrib] = N.asarray(value)
        uniqueattr = "unique" + attrib

        _dsattr = self._dsattr
        if _dsattr.has_key(uniqueattr):
            _dsattr[uniqueattr] = None


    def _getNSamplesPerAttr( self, attrib='labels' ):
        """Returns the number of samples per unique label.
        """
        # local bindings
        _data = self._data

        # XXX hardcoded dict_=self._data.... might be in self._dsattr
        uniqueattr = self._getuniqueattr(attrib="unique" + attrib,
                                         dict_=_data)

        # use dictionary to cope with arbitrary labels
        result = dict(zip(uniqueattr, [ 0 ] * len(uniqueattr)))
        for l in _data[attrib]:
            result[l] += 1

        # XXX only return values to mimic the old interface but we might want
        # to return the full dict instead
        # return result
        return result


    def _getSampleIdsByAttr(self, values, attrib="labels",
                            sort=True):
        """Return indecies of samples given a list of attributes
        """

        if not operator.isSequenceType(values) \
               or isinstance(values, basestring):
            values = [ values ]

        # TODO: compare to plain for loop through the labels
        #       on a real data example
        sel = N.array([], dtype=N.int16)
        _data = self._data
        for value in values:
            sel = N.concatenate((
                sel, N.where(_data[attrib]==value)[0]))

        if sort:
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
        # local bindings
        _data = self._data
        labels = self.labels
        nsamples = self.nsamples

        lastseen = none = [None for attr in attributes_to_track]
        transitions = []

        for i in xrange(nsamples+1):
            if i < nsamples:
                current = [_data[attr][i] for attr in attributes_to_track]
            else:
                current = none
            if lastseen != current:
                # transition point
                new_transitions = range(max(0, i-prior),
                                        min(nsamples-1, i+post)+1)
                if affected_labels is not None:
                    new_transitions = [labels[i] for i in new_transitions
                                       if i in affected_labels]
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

        # local bindings
        nsamples = self.nsamples
        _data = self._data

        for k, v in _data.iteritems():
            if not len(v) == nsamples:
                raise DatasetError, \
                      "Length of sample attribute '%s' [%i] does not " \
                      "match the number of samples in the dataset [%i]." \
                      % (k, len(v), nsamples)

        # check for unique origids
        uniques = N.unique(_data['origids'])
        uniques.sort()
        # need to copy to prevent sorting the original array
        sorted_ids = _data['origids'].copy()
        sorted_ids.sort()

        if not (uniques == sorted_ids).all():
            raise DatasetError, "Samples IDs are not unique."


    def _expandSampleAttribute(self, attr, attr_name):
        """If a sample attribute is given as a scalar expand/repeat it to a
        length matching the number of samples in the dataset.
        """
        try:
            # if we are initializing with a single string -- we should
            # treat it as a single label
            if isinstance(attr, basestring):
                raise TypeError
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
    def _registerAttribute(cls, key, dictname="_data", abbr=None, hasunique=False):
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

            if abbr is not None:
                exec "%s.%s = property(fget=%s, fset=%s)"  % \
                     (cls.__name__, abbr, getter, setter)

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
                if abbr is not None:
                    exec "%s.U%s = property(fget=%s)" % \
                         (cls.__name__, abbr, getter)

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
                            lstats=__debug__ and ('DS_STATS' in debug.active),
                            )


    def __repr__(self):
        return "<%s>" % str(self)


    def summary(self, uniq=True, stats=True, idhash=False, lstats=True,
                maxc=30, maxl=20):
        """String summary over the object

        :Parameters:
          uniq : bool
             Include summary over data attributes which have unique
          idhash : bool
             Include idhash value for dataset and samples
          stats : bool
             Include some basic statistics (mean, std, var) over dataset samples
          lstats : bool
             Include statistics on chunks/labels
          maxc : int
            Maximal number of chunks when provide details on labels/chunks
          maxl : int
            Maximal number of labels when provide details on labels/chunks
        """
        # local bindings
        samples = self.samples
        _data = self._data
        _dsattr = self._dsattr

        if idhash:
            idhash_ds = "{%s}" % self.idhash
            idhash_samples = "{%s}" % idhash_(samples)
        else:
            idhash_ds = ""
            idhash_samples = ""

        s = """Dataset %s/ %s %d%s x %d""" % \
            (idhash_ds, samples.dtype,
             self.nsamples, idhash_samples, self.nfeatures)

        ssep = (' ', '\n')[lstats]
        if uniq:
            s +=  "%suniq:" % ssep
            for uattr in _dsattr.keys():
                if not uattr.startswith("unique"):
                    continue
                attr = uattr[6:]
                try:
                    value = self._getuniqueattr(attrib=uattr,
                                                dict_=_data)
                    s += " %d %s" % (len(value), attr)
                except:
                    pass

        if isinstance(self.labels_map, dict):
            s += ' labels_mapped'

        if stats:
            # TODO -- avg per chunk?
            # XXX We might like to use scipy.stats.describe to get
            # quick summary statistics (mean/range/skewness/kurtosis)
            s += "%sstats: mean=%g std=%g var=%g min=%g max=%g\n" % \
                 (ssep, N.mean(samples), N.std(samples),
                  N.var(samples), N.min(samples), N.max(samples))

        if lstats:
            s += self.summary_labels(maxc=maxc, maxl=maxl)

        return s


    def summary_labels(self, maxc=30, maxl=20):
        """Provide summary statistics over the labels and chunks

        :Parameters:
          maxc : int
            Maximal number of chunks when provide details
          maxl : int
            Maximal number of labels when provide details
        """
        # We better avoid bound function since if people only
        # imported Dataset without miscfx it would fail
        from mvpa.datasets.miscfx import getSamplesPerChunkLabel
        spcl = getSamplesPerChunkLabel(self)
        # XXX couldn't they be unordered?
        ul = self.uniquelabels.tolist()
        uc = self.uniquechunks.tolist()
        s = ""
        if len(ul) < maxl and len(uc) < maxc:
            s += "\nCounts of labels in each chunk:"
            # only in a resonable case do printing
            table = [['  chunks\labels'] + ul]
            table += [[''] + ['---'] * len(ul)]
            for c, counts in zip(uc, spcl):
                table.append([ str(c) ] + counts.tolist())
            s += '\n' + table2string(table)
        else:
            s += "No details due to large number of labels or chunks. " \
                 "Increase maxc and maxl if desired"

        labels_map = self.labels_map
        if isinstance(labels_map, dict):
            s += "\nOriginal labels were mapped using following mapping:"
            s += '\n\t'+'\n\t'.join([':\t'.join(map(str, x))
                                     for x in labels_map.items()]) + '\n'

        def cl_stats(axis, u, name1, name2):
            """ Compute statistics per label
            """
            stats = {'min': N.min(spcl, axis=axis),
                     'max': N.max(spcl, axis=axis),
                     'mean': N.mean(spcl, axis=axis),
                     'std': N.std(spcl, axis=axis),
                     '#%ss' % name2: N.sum(spcl>0, axis=axis)}
            entries = ['  ' + name1, 'mean', 'std', 'min', 'max', '#%ss' % name2]
            table = [ entries ]
            for i, l in enumerate(u):
                d = {'  ' + name1 : l}
                d.update(dict([ (k, stats[k][i]) for k in stats.keys()]))
                table.append( [ ('%.3g', '%s')[isinstance(d[e], basestring)]
                                % d[e] for e in entries] )
            return '\nSummary per %s across %ss\n' % (name1, name2) \
                   + table2string(table)

        if len(ul) < maxl:
            s += cl_stats(0, ul, 'label', 'chunk')
        if len(uc) < maxc:
            s += cl_stats(1, uc, 'chunk', 'label')
        return s


    def __iadd__(self, other):
        """Merge the samples of one Dataset object to another (in-place).

        No dataset attributes, besides labels_map, will be merged!
        Additionally, a new set of unique `origids` will be generated.
        """
        # local bindings
        _data = self._data
        other_data = other._data

        if not self.nfeatures == other.nfeatures:
            raise DatasetError, "Cannot add Dataset, because the number of " \
                                "feature do not match."

        # take care about labels_map and labels
        slm = self.labels_map
        olm = other.labels_map
        if N.logical_xor(slm is None, olm is None):
            raise ValueError, "Cannot add datasets where only one of them " \
                  "has labels map assigned. If needed -- implement it"

        # concatenate all sample attributes
        for k,v in _data.iteritems():
            if k == 'origids':
                # special case samples origids: for now just regenerate unique
                # ones could also check if concatenation is unique, but it
                # would be costly performance-wise
                _data[k] = N.arange(len(v) + len(other_data[k]))

            elif k == 'labels' and slm is not None:
                # special care about labels if mapping was in effect,
                # we need to append 2nd map to the first one and
                # relabel 2nd dataset
                nlm = slm.copy()
                # figure out maximal numerical label used now
                nextid = N.sort(nlm.values())[-1] + 1
                olabels = other.labels
                olabels_remap = {}
                for ol, olnum in olm.iteritems():
                    if not nlm.has_key(ol):
                        # check if we can preserve old numberic label
                        # if not -- assign some new one not yet present
                        # in any dataset
                        if olnum in nlm.values():
                            nextid = N.sort(nlm.values() + olm.values())[-1] + 1
                        else:
                            nextid = olnum
                        olabels_remap[olnum] = nextid
                        nlm[ol] = nextid
                        nextid += 1
                    else:
                        olabels_remap[olnum] = nlm[ol]
                olabels = [olabels_remap[x] for x in olabels]
                # finally compose new labels
                _data['labels'] = N.concatenate((v, olabels), axis=0)
                # and reassign new mapping
                self._dsattr['labels_map'] = nlm

                if __debug__:
                    # check if we are not dealing with colliding
                    # mapping, since it is problematic and might lead
                    # to various complications
                    if (len(Set(slm.keys())) != len(Set(slm.values()))) or \
                       (len(Set(olm.keys())) != len(Set(olm.values()))):
                        warning("Adding datasets where multiple labels "
                                "mapped to the same ID is not recommended. "
                                "Please check the outcome. Original mappings "
                                "were %s and %s. Resultant is %s"
                                % (slm, olm, nlm))

            else:
                _data[k] = N.concatenate((v, other_data[k]), axis=0)

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
        # create a new object of the same type it is now and NOT only Dataset
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


    def copy(self):
        """Create a copy (clone) of the dataset, by fully copying current one

        """
        # create a new object of the same type it is now and NOT only Dataset
        out = super(Dataset, self).__new__(self.__class__)

        # now init it: to make it work all Dataset contructors have to accept
        # Class(data=Dict, dsattr=Dict)
        out.__init__(data=self._data,
                     dsattr=self._dsattr,
                     copy_samples=True,
                     copy_data=True,
                     copy_dsattr=True)

        return out


    def selectFeatures(self, ids=None, sort=True, groups=None):
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
        ids = copy.deepcopy(ids)
        
        if ids is None and groups is None:
            raise ValueError, "No feature selection specified."

        # start with empty list if no ids where specified (so just groups)
        if ids is None:
            ids = []

        if not groups is None:
            if not self._dsattr.has_key('featuregroups'):
                raise RuntimeError, \
                "Dataset has no feature grouping information."

            for g in groups:
                ids += (self._dsattr['featuregroups'] == g).nonzero()[0].tolist()

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

        # apply selection to feature groups as well
        if self._dsattr.has_key('featuregroups'):
            new_dsattr = self._dsattr.copy()
            new_dsattr['featuregroups'] = self._dsattr['featuregroups'][ids]
        else:
            new_dsattr = self._dsattr

        # create a new object of the same type it is now and NOT onyl Dataset
        dataset = super(Dataset, self).__new__(self.__class__)

        # now init it: to make it work all Dataset contructors have to accept
        # Class(data=Dict, dsattr=Dict)
        dataset.__init__(data=new_data,
                         dsattr=new_dsattr,
                         check_data=False,
                         copy_samples=False,
                         copy_data=False,
                         copy_dsattr=False
                         )

        return dataset


    def applyMapper(self, featuresmapper=None, samplesmapper=None,
                    train=True):
        """Obtain new dataset by applying mappers over features and/or samples.

        While featuresmappers leave the sample attributes information
        unchanged, as the number of samples in the dataset is invariant,
        samplesmappers are also applied to the samples attributes themselves!

        Applying a featuresmapper will destroy any feature grouping information.

        :Parameters:
          featuresmapper : Mapper
            `Mapper` to somehow transform each sample's features
          samplesmapper : Mapper
            `Mapper` to transform each feature across samples
          train : bool
            Flag whether to train the mapper with this dataset before applying
            it.

        TODO: selectFeatures is pretty much
              applyMapper(featuresmapper=MaskMapper(...))
        """

        # shallow-copy all stuff from current data dict
        new_data = self._data.copy()

        # apply mappers

        if samplesmapper:
            if __debug__:
                debug("DS", "Training samplesmapper %s" % `samplesmapper`)
            samplesmapper.train(self)

            if __debug__:
                debug("DS", "Applying samplesmapper %s" % `samplesmapper` +
                      " to samples of dataset `%s`" % `self`)

            # get rid of existing 'origids' as they are not valid anymore and
            # applying a mapper to them is not really meaningful
            if new_data.has_key('origids'):
                del(new_data['origids'])

            # apply mapper to all sample-wise data in dataset
            for k in new_data.keys():
                new_data[k] = samplesmapper.forward(self._data[k])

        # feature mapping might affect dataset attributes
        # XXX: might be obsolete when proper feature attributes are implemented
        new_dsattr = self._dsattr

        if featuresmapper:
            if __debug__:
                debug("DS", "Training featuresmapper %s" % `featuresmapper`)
            featuresmapper.train(self)

            if __debug__:
                debug("DS", "Applying featuresmapper %s" % `featuresmapper` +
                      " to samples of dataset `%s`" % `self`)
            new_data['samples'] = featuresmapper.forward(self._data['samples'])

            # remove feature grouping, who knows what the mapper did to the
            # features
            if self._dsattr.has_key('featuregroups'):
                new_dsattr = self._dsattr.copy()
                del(new_dsattr['featuregroups'])
            else:
                new_dsattr = self._dsattr

        # create a new object of the same type it is now and NOT only Dataset
        dataset = super(Dataset, self).__new__(self.__class__)

        # now init it: to make it work all Dataset contructors have to accept
        # Class(data=Dict, dsattr=Dict)
        dataset.__init__(data=new_data,
                         dsattr=new_dsattr,
                         check_data=False,
                         copy_samples=False,
                         copy_data=False,
                         copy_dsattr=False
                         )

        # samples attributes might have changed after applying samplesmapper
        if samplesmapper:
            dataset._resetallunique(force=True)

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



    def index(self, *args, **kwargs):
        """Universal indexer to obtain indexes of interesting samples/features.
        See .select() for more information

        :Return: tuple of (samples indexes, features indexes). Each
          item could be also None, if no selection on samples or
          features was requested (to discriminate between no selected
          items, and no selections)
        """
        s_indx = []                     # selections for samples
        f_indx = []                     # selections for features
        return_dataset = kwargs.pop('return_dataset', False)
        largs = len(args)

        args = list(args)               # so we could override
        # Figure out number of positional
        largs_nonstring = 0
        # need to go with index since we might need to override internally
        for i in xrange(largs):
            l = args[i]
            if isinstance(l, basestring):
                if l.lower() == 'all':
                    # override with a slice
                    args[i] = slice(None)
                else:
                    break
            largs_nonstring += 1

        if largs_nonstring >= 1:
            s_indx.append(args[0])
            if __debug__ and 'CHECK_DS_SELECT' in debug.active:
                _validate_indexes_uniq_sorted(args[0], 'select', 'samples')
            if largs_nonstring == 2:
                f_indx.append(args[1])
                if __debug__ and 'CHECK_DS_SELECT' in debug.active:
                    _validate_indexes_uniq_sorted(args[1], 'select', 'features')
            elif largs_nonstring > 2:
                raise ValueError, "Only two positional arguments are allowed" \
                      ". 1st for samples, 2nd for features"

        # process left positional arguments which must encode selections like
        # ('labels', [1,2,3])

        if (largs - largs_nonstring) % 2 != 0:
            raise ValueError, "Positional selections must come in pairs:" \
                  " e.g. ('labels', [1,2,3])"

        for i in xrange(largs_nonstring, largs, 2):
            k, v = args[i:i+2]
            kwargs[k] = v

        # process keyword parameters
        data_ = self._data
        for k, v in kwargs.iteritems():
            if k == 'samples':
                s_indx.append(v)
            elif k == 'features':
                f_indx.append(v)
            elif data_.has_key(k):
                # so it is an attribute for samples
                # XXX may be do it not only if __debug__
                if __debug__: # and 'CHECK_DS_SELECT' in debug.active:
                    if not N.any([isinstance(v, cls) for cls in
                                  [list, tuple, slice, int]]):
                        raise ValueError, "Trying to specify selection for %s " \
                              "based on unsupported '%s'" % (k, v)
                s_indx.append(self._getSampleIdsByAttr(v, attrib=k, sort=False))
            else:
                raise ValueError, 'Keyword "%s" is not known, thus' \
                      'select() failed' % k

        def combine_indexes(indx, nelements):
            """Helper function: intersect selections given in indx

            :Parameters:
              indxs : list of lists or slices
                selections of elements
              nelements : int
                number of elements total for deriving indexes from slices
            """
            indx_sel = None                 # pure list of ids for selection
            for s in indx:
                if isinstance(s, slice) or \
                   isinstance(s, N.ndarray) and s.dtype==bool:
                    # XXX there might be a better way than reconstructing the full
                    # index list. Also we are loosing ability to do simlpe slicing,
                    # ie w.o making a copy of the selected data
                    all_indexes = N.arange(nelements)
                    s = all_indexes[s]
                elif not operator.isSequenceType(s):
                    s = [ s ]

                if indx_sel is None:
                    indx_sel = Set(s)
                else:
                    # To be consistent
                    #if not isinstance(indx_sel, Set):
                    #    indx_sel = Set(indx_sel)
                    indx_sel = indx_sel.intersection(s)

            # if we got Set -- convert
            if isinstance(indx_sel, Set):
                indx_sel = list(indx_sel)

            # sort for the sake of sanity
            indx_sel.sort()

            return indx_sel

        # Select samples
        if len(s_indx) == 1 and isinstance(s_indx[0], slice) \
               and s_indx[0] == slice(None):
            # so no actual selection -- full slice
            s_indx = s_indx[0]
        else:
            # else - get indexes
            if len(s_indx) == 0:
                s_indx = None
            else:
                s_indx = combine_indexes(s_indx, self.nsamples)

        # Select features
        if len(f_indx):
            f_indx = combine_indexes(f_indx, self.nfeatures)
        else:
            f_indx = None

        return s_indx, f_indx


    def select(self, *args, **kwargs):
        """Universal selector

        WARNING: if you need to select duplicate samples
        (e.g. samples=[5,5]) or order of selected samples of features
        is important and has to be not ordered (e.g. samples=[3,2,1]),
        please use selectFeatures or selectSamples functions directly

        Examples:
          Mimique plain selectSamples::

            dataset.select([1,2,3])
            dataset[[1,2,3]]

          Mimique plain selectFeatures::

            dataset.select(slice(None), [1,2,3])
            dataset.select('all', [1,2,3])
            dataset[:, [1,2,3]]

          Mixed (select features and samples)::

            dataset.select([1,2,3], [1, 2])
            dataset[[1,2,3], [1, 2]]

          Select samples matching some attributes::

            dataset.select(labels=[1,2], chunks=[2,4])
            dataset.select('labels', [1,2], 'chunks', [2,4])
            dataset['labels', [1,2], 'chunks', [2,4]]

          Mixed -- out of first 100 samples, select only those with
          labels 1 or 2 and belonging to chunks 2 or 4, and select
          features 2 and 3::

            dataset.select(slice(0,100), [2,3], labels=[1,2], chunks=[2,4])
            dataset[:100, [2,3], 'labels', [1,2], 'chunks', [2,4]]

        """
        s_indx, f_indx = self.index(*args, **kwargs)

        # Select samples
        if s_indx == slice(None):
            # so no actual selection was requested among samples.
            # thus proceed with original dataset
            if __debug__:
                debug('DS', 'in select() not selecting samples')
            ds = self
        else:
            # else do selection
            if __debug__:
                debug('DS', 'in select() selecting samples given selections'
                      + str(s_indx))
            ds = self.selectSamples(s_indx)

        # Select features
        if f_indx is not None:
            if __debug__:
                debug('DS', 'in select() selecting features given selections'
                      + str(f_indx))
            ds = ds.selectFeatures(f_indx)

        return ds



    def where(self, *args, **kwargs):
        """Obtain indexes of interesting samples/features. See select() for more information

        XXX somewhat obsoletes idsby...
        """
        s_indx, f_indx = self.index(*args, **kwargs)
        if s_indx is not None and f_indx is not None:
            return s_indx, f_indx
        elif s_indx is not None:
            return s_indx
        else:
            return f_indx


    def __getitem__(self, *args):
        """Convinience dataset parts selection

        See select for more information
        """
        # for cases like ['labels', 1]
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]

        args_, args = args, ()
        for a in args_:
            if isinstance(a, slice) and \
                   isinstance(a.start, basestring):
                    # for the constructs like ['labels':[1,2]]
                    if a.stop is None or a.step is not None:
                        raise ValueError, \
                              "Selection must look like ['chunks':[2,3]]"
                    args += (a.start, a.stop)
            else:
                args += (a,)
        return self.select(*args)


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
        # local bindings
        _data = self._data

        if len(self.uniquelabels)<2:
            raise RuntimeError, \
                  "Call to permuteLabels is bogus since there is insuficient" \
                  " number of labels: %s" % self.uniquelabels

        if not status:
            # restore originals
            if _data.get('origlabels', None) is None:
                raise RuntimeError, 'Cannot restore labels. ' \
                                    'permuteLabels() has never been ' \
                                    'called with status == True.'
            self.labels = _data['origlabels']
            _data.pop('origlabels')
        else:
            # store orig labels, but only if not yet done, otherwise multiple
            # calls with status == True will destroy the original labels
            if not _data.has_key('origlabels') \
                or _data['origlabels'] == None:
                # bind old labels to origlabels
                _data['origlabels'] = _data['labels']
                # copy labels
                _data['labels'] = copy.copy(_data['labels'])

            labels = _data['labels']
            # now scramble
            if perchunk:
                for o in self.uniquechunks:
                    labels[self.chunks == o] = \
                        N.random.permutation(labels[self.chunks == o])
            else:
                labels = N.random.permutation(labels)

            self.labels = labels

            if assure_permute:
                if not (_data['labels'] != _data['origlabels']).any():
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
        labels = self.labels
        for i, r in enumerate(self.uniquelabels):
            # get the list of pattern ids for this class
            sample += random.sample( (labels == r).nonzero()[0],
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


    def getLabelsMap(self):
        """Stored labels map (if any)
        """
        return self._dsattr.get('labels_map', None)


    def setLabelsMap(self, lm):
        """Set labels map.

        Checks for the validity of the mapping -- values should cover
        all existing labels in the dataset
        """
        values = Set(lm.values())
        labels = Set(self.uniquelabels)
        if not values.issuperset(labels):
            raise ValueError, \
                  "Provided mapping %s has some existing labels (out of %s) " \
                  "missing from mapping" % (list(values), list(labels))
        self._dsattr['labels_map'] = lm


    def setSamplesDType(self, dtype):
        """Set the data type of the samples array.
        """
        # local bindings
        _data = self._data

        if _data['samples'].dtype != dtype:
            _data['samples'] = _data['samples'].astype(dtype)


    def defineFeatureGroups(self, definition):
        """Assign `definition` to featuregroups

        XXX Feature-groups was not finished to be useful
        """
        if not len(definition) == self.nfeatures:
            raise ValueError, \
                  "Length of feature group definition %i " \
                  "does not match the number of features %i " \
                  % (len(definition), self.nfeatures)

        self._dsattr['featuregroups'] = N.array(definition)


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


    @staticmethod
    def _checkCopyConstructorArgs(**kwargs):
        """Common sanity check for Dataset copy constructor calls."""
        # check if we have samples (somwhere)
        samples = None
        if kwargs.has_key('samples'):
            samples = kwargs['samples']
        if samples is None and kwargs.has_key('data') \
           and kwargs['data'].has_key('samples'):
            samples = kwargs['data']['samples']
        if samples is None:
            raise DatasetError, \
                  "`samples` must be provided to copy constructor call."

        if not len(samples.shape) == 2:
            raise DatasetError, \
                  "samples must be in 2D shape in copy constructor call."


    # read-only class properties
    nsamples        = property( fget=getNSamples )
    nfeatures       = property( fget=getNFeatures )
    labels_map      = property( fget=getLabelsMap, fset=setLabelsMap )

def datasetmethod(func):
    """Decorator to easily bind functions to a Dataset class
    """
    if __debug__:
        debug("DS_",  "Binding function %s to Dataset class" % func.func_name)

    # Bind the function
    setattr(Dataset, func.func_name, func)

    # return the original one
    return func


# Following attributes adherent to the basic dataset
Dataset._registerAttribute("samples", "_data", abbr='S', hasunique=False)
Dataset._registerAttribute("labels",  "_data", abbr='L', hasunique=True)
Dataset._registerAttribute("chunks",  "_data", abbr='C', hasunique=True)
# samples ids (already unique by definition)
Dataset._registerAttribute("origids",  "_data", abbr='I', hasunique=False)



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

from mvpa.misc.state import ClassWithCollections, Collection
from mvpa.misc.attributes import SampleAttribute, FeatureAttribute, \
        DatasetAttribute

# Remaining public interface of Dataset
class _Dataset(ClassWithCollections):
    """The successor of Dataset.
    """
    # placeholder for all three basic collections of a Dataset
    # put here to be able to check whether the AttributesCollector already
    # instanciated a particular collection
    # XXX maybe it should not do this at all for Dataset
    sa = None
    fa = None
    dsa = None

    # storage of samples in a plain NumPy array for fast access
    samples = None

    def __init__(self, samples, sa=None, fa=None, dsa=None):
        """
        This is the generic internal constructor. Its main task is to allow
        for a maximum level of customization during dataset construction,
        including fast copy construction.

        Parameters
        ----------
        samples : ndarray
          Data samples.
        sa : Collection
          Samples attributes collection.
        fa : Collection
          Features attributes collection.
        dsa : Collection
          Dataset attributes collection.
        """
        # init base class
        ClassWithCollections.__init__(self)

        # Internal constructor -- users focus on init* Methods

        # Every dataset needs data (aka samples), completely data-driven
        # analyses might not even need labels, so this is the only mandatory
        # argument
        # XXX add checks
        self.samples = samples

        # Everything else in a dataset (except for samples) is organized in
        # collections
        # copy attributes from source collections (scol) into target
        # collections (tcol)
        for scol, tcol in ((sa, self.sa),
                           (fa, self.fa),
                           (dsa, self.dsa)):
            # make sure we have the target collection
            if tcol is None:
                # XXX maybe use different classes for the collections
                # but currently no reason to do so
                tcol = Collection(owner=self)

            # transfer the attributes
            if not scol is None:
                for name, attr in scol.items.iteritems():
                    # this will also update the owner of the attribute
                    # XXX discuss the implications of always copying
                    tcol.add(copy.copy(attr))


    @classmethod
    def initSimple(klass, samples, labels, chunks):
        # use Numpy convention
        """
        One line summary.

        Long description.

        Parameters
        ----------
        samples : ndarray
          The two-dimensional samples matrix.
        labels : ndarray
        chunks : ndarray

        Returns
        -------
        blah blah

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
        # Demo user contructor

        # compile the necessary samples attributes collection
        labels_ = SampleAttribute(name='labels')
        labels_.value = labels
        chunks_ = SampleAttribute(name='chunks')
        chunks_.value = chunks

        # feels strange that one has to give the name again
        # XXX why does items have to be a dict when each samples
        # attr already knows its name
        sa = Collection(items={'labels': labels_, 'chunks': chunks_})

        # common checks should go into __init__
        return klass(samples, sa=sa)


    def getNSamples( self ):
        """Currently available number of patterns.
        """
        return self.samples.shape[0]


    def getNFeatures( self ):
        """Number of features per pattern.
        """
        return self.samples.shape[1]


#
#    @property
#    def idhash(self):
#        pass
#
#
#    def idsonboundaries(self, prior=0, post=0,
#                        attributes_to_track=['labels', 'chunks'],
#                        affected_labels=None,
#                        revert=False):
#        pass
#
#
#    def summary(self, uniq=True, stats=True, idhash=False, lstats=True,
#                maxc=30, maxl=20):
#        pass
#
#
#    def summary_labels(self, maxc=30, maxl=20):
#        pass
#
#
#    def __iadd__(self, other):
#        pass
#
#
#    def __add__( self, other ):
#        pass
#
#
#    def copy(self):
#        pass
#
#
#    def selectFeatures(self, ids=None, sort=True, groups=None):
#        pass
#
#
#    def applyMapper(self, featuresmapper=None, samplesmapper=None,
#                    train=True):
#        pass
#
#
#    def selectSamples(self, ids):
#        pass
#
#
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
#
#
#    def __getitem__(self, *args):
#        pass
#
#
#    def permuteLabels(self, status, perchunk=True, assure_permute=False):
#        pass
#
#
#    def getRandomSamples(self, nperlabel):
#        pass
#
#
#    def getLabelsMap(self):
#        pass
#
#
#    def setLabelsMap(self, lm):
#        pass
#
#
#    def setSamplesDType(self, dtype):
#        pass
#
#
#    def defineFeatureGroups(self, definition):
#        pass
#
#
#    def convertFeatureIds2FeatureMask(self, ids):
#        pass
#
#
#    def convertFeatureMask2FeatureIds(self, mask):
#        pass
