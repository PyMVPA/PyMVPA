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
import copy

import numpy as N

from mvpa.misc import debug

class Dataset(object):
    """ This class provides a container to store all necessary data to perform
    MVPA analyses. These are the data samples, as well as the labels
    associated with these patterns. Additionally samples can be grouped into
    chunks.
    """

    # static definition to track which unique attributes
    # have to be reset/recomputed whenever anything relevant
    # changes

    # unique{labels,chunks} become a part of dsattr
    _uniqueattributes = []


    def __init__(self, data={}, dsattr={}, dtype=None, \
                 samples=None, labels=None, chunks=None, check_data=True,
                 copy_samples=False, copy_data=True, copy_dsattr=True):
        """
        - `data`: Dict with an arbitrary number of entries. The value for
                  each key in the dict has to be an ndarray with the
                  same length as the number of rows in the samples array.
                  A special entry in theis dictionary is 'samples', a 2d array
                  (samples x features). A shallow copy is stored in the object.
        - `dsattr`: Dictionary of dataset attributes. An arbitrary number of
                    arbitrarily named and typed objects can be stored here. A
                    shallow copy of the dictionary is stored in the object.
        - `dtype`: If None -- do not change data type if samples
                   is an ndarray. Otherwise convert samples to dtype.

        Each of the following arguments overwrites with is/might be already in
        the `data` container.
        - `samples`: a 2d array (samples x features)
        - `labels`: array or scalar value
        - `chunks`: array or scalar value
        """
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

        if copy_dsattr:
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
        # labels
        if not labels == None:
            if __debug__:
                if self._data.has_key('labels'):
                    debug('DS',
                          "`Data` dict has `labels` (%s) but there is also" +
                          " __init__ parameter `labels` which overrides " +
                          " stored in `data`" % (`self._data['labels']`))
            self._data['labels'] = \
                self._expandSampleAttribute(labels, 'labels')

        # chunks
        if not chunks == None:
            self._data['chunks'] = \
                self._expandSampleAttribute(chunks, 'chunks')
        elif not self._data.has_key('chunks'):
            # if no chunk information is given assume that every pattern
            # is its own chunk
            self._data['chunks'] = N.arange(self.nsamples)

        if check_data:
            self._checkData()

        # lazy computation of unique members
        #self._resetallunique('_dsattr', self._dsattr)
        if not labels is None or not chunks is None:
            # for a speed up to don't go through all uniqueattributes
            # when no need
            self._dsattr['__uniquereseted'] = False
            self._resetallunique()


    def _resetallunique(self):
        """Set to None all unique* attributes of corresponding dictionary
        """

        if self._dsattr['__uniquereseted']:
            return

        # I guess we better checked if dictname is known  but...
        for k in self._uniqueattributes:
            if __debug__:
                debug("DS", "Reset attribute %s" % k)
            self._dsattr[k] = None
        self._dsattr['__uniquereseted'] = True


    def _getuniqueattr(self, attrib, dict_):
        """
        Provide common facility to return unique attributes

        XXX dict_ can be simply replaced now with self._dsattr
        """
        if not self._dsattr.has_key(attrib) or self._dsattr[attrib] is None:
            if __debug__:
                debug("DS", "Recomputing unique set for attrib %s within %s" %
                      (attrib, self.__repr__(False)))
            # uff... might come up with better strategy to keep relevant
            # attribute name
            self._dsattr[attrib] = N.unique( dict_[attrib[6:]] )
            assert(not self._dsattr[attrib] is None)
            self._dsattr['__uniquereseted'] = False

        return self._dsattr[attrib]


    def _getNSamplesPerAttr( self, attrib='labels' ):
        """ Returns the number of samples per unique label.
        """
        # XXX hardcoded dict_=self._data.... might be in self._dsattr
        uniqueattr = self._getuniqueattr(attrib="unique" + attrib,
                                         dict_=self._data)

        # TODO what if attribute is not a number???
        result = [ 0 ] * len(uniqueattr)
        for l in self._data[attrib]:
            result[l] += 1
        return result


    def _getSampleIdsByAttr(self, values, attrib="labels"):
        """ Return indecies of samples given a list of attributes
        """

        if not operator.isSequenceType(values):
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
            raise ValueError, "Only (samples x features) -> 2d sample " \
                            + "are supported. Consider MappedDataset if " \
                            + "applicable."

        return samples


    def _checkData(self):
        """Checks `_data` members to have the same # of samples.
        """
        for k, v in self._data.iteritems():
            if not len(v) == self.nsamples:
                raise ValueError, \
                      "Length of sample attribute '%s' [%i] does not " \
                      "match the number of samples in the dataset [%i]." \
                      % (k, len(v), self.nsamples)


    def _expandSampleAttribute(self, attr, attr_name):
        """If a sample attribute is given as a scalar expand/repeat it to a
        length matching the number of samples in the dataset.
        """
        try:
            if len(attr) != self.nsamples:
                raise ValueError, \
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
        """Register an attribute for *Dataset class.

        Creates property assigning getters/setters depending on the
        availability of corresponding _get, _set functions.
        """
        #import pydb
        #pydb.debugger()
        classdict = cls.__dict__
        if not classdict.has_key(key):
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
            else:
                setter = None

            if __debug__:
                debug("DS", "Registering new property %s.%s" %
                      (cls.__name__, key))
            exec "%s.%s = property(fget=%s,fset=%s)"  % \
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
                      "lambda self, x: self._getSampleIdsByAttr(x,attrib='%s')" % key)

                cls._uniqueattributes.append(uniquekey)



        elif __debug__:
            debug('DS', 'Trying to reregister attribute `%s`. For now ' +
                  'such facility is not active')


    def __repr__(self, full=True):
        """ String summary over the object
        """
        if full:
            return """Dataset / %s %d x %d, uniq: %d labels, %d chunks""" % \
                   (self.samples.dtype, self.nsamples, self.nfeatures,
                    len(self.uniquelabels), len(self.uniquechunks))
        else:
            return """Dataset / %s %d x %d""" % \
                   (self.samples.dtype, self.nsamples, self.nfeatures)


    def __iadd__( self, other ):
        """ Merge the samples of one Dataset object to another (in-place).

        No dataset attributes will be merged!
        """
        if not self.nfeatures == other.nfeatures:
            raise ValueError, "Cannot add Dataset, because the number of " \
                              "feature do not match."

        # concatenate all sample attributes
        for k, v in self._data.iteritems():
            self._data[k] = N.concatenate((v, other._data[k]), axis=0)

        # might be more sophisticated but for now just reset -- it is safer ;)
        self._resetallunique()

        return self


    def __add__( self, other ):
        """ Merge the samples two Dataset objects.

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


    def selectFeatures(self, ids):
        """ Select a number of features from the current set.

        `ids` is a list of feature IDs

        Returns a new Dataset object with a view of the original samples
        array (no copying is performed).

        ATTENTION: The order of ids determines the order of features in the
        returned dataset. This might be useful sometimes, but can also cause
        major headaches!
        """
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
                         dsattr=self._dsattr)

        return dataset


    def selectSamples(self, mask):
        """ Choose a subset of samples.

        Returns a new dataset object containing the selected sample
        subset.
        """
        # without having a sequence a index the masked sample array would
        # loose its 2d layout
        if not operator.isSequenceType( mask ):
            mask = [mask]

        # mask all sample attributes
        data = {}
        for k, v in self._data.iteritems():
            data[k] = v[mask, ]

        # create a new object of the same type it is now and NOT onyl Dataset
        dataset = super(Dataset, self).__new__(self.__class__)

        # now init it: to make it work all Dataset contructors have to accept
        # Class(data=Dict, dsattr=Dict)
        dataset.__init__(data=data,
                         dsattr=self._dsattr)

        return dataset



    def permutedRegressors( self, status, perchunk = True ):
        """ Permute the labels.

        Calling this method with 'status' set to True, the labels are
        permuted among all samples.

        If 'perorigin' is True permutation is limited to samples sharing the
        same chunk value. Therefore only the association of a certain sample
        with a label is permuted while keeping the absolute number of
        occurences of each label value within a certain chunk constant.

        If 'status' is False the original labels are restored.
        """
        if not status:
            # restore originals
            if self._data['origlabels'] == None:
                raise RuntimeError, 'Cannot restore labels. ' \
                                    'randomizedRegressors() has never been ' \
                                    'called with status == True.'
            self._setLabels(self._data['origlabels'])
            self._data['origlabels'] = None
        else:
            # permute labels per origin

            # make a backup of the original labels
            self._data['origlabels'] = self._data['labels'].copy()

            # now scramble the rest
            if perchunk:
                for o in self.uniquechunks:
                    self._data['labels'][self.chunks == o ] = \
                        N.random.permutation( self.labels[ self.chunks == o ] )
                # to recompute uniquelabels
                self._setLabels(self._data['labels'])
            else:
                self._setLabels(N.random.permutation(self._data['labels']))


    def getRandomSamples( self, nperlabel ):
        """ Select a random set of samples.

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


    # TODO? Following 2 setters might be gone as well after appropriate
    # modification of _registerAttribute
    def _setLabels(self, labels):
        """ Sets labels and recomputes uniquelabels
        """
        self._data['labels'] = labels
        self._data['uniquelabels'] = None # None!since we might not need them


    def _setChunks(self, chunks):
        """ Sets chunks and recomputes uniquechunks
        """
        self._data['chunks'] = chunks
        self._data['uniquechunks'] = None # None!since we might not need them


    def getNSamples( self ):
        """ Currently available number of patterns.
        """
        return self._data['samples'].shape[0]


    def getNFeatures( self ):
        """ Number of features per pattern.
        """
        return self._data['samples'].shape[1]



    def setSamplesDType(self, dtype):
        """Set the data type of the samples array.
        """
        if self._data['samples'].dtype != dtype:
            self._data['samples'] = self._data['samples'].astype(dtype)


    # read-only class properties
    nsamples        = property( fget=getNSamples )
    nfeatures       = property( fget=getNFeatures )


# Following attributes adherent to the basic dataset
Dataset._registerAttribute("samples", "_data", hasunique=False)
Dataset._registerAttribute("labels",  "_data", hasunique=True)
Dataset._registerAttribute("chunks",  "_data", hasunique=True)
