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


class Dataset(object):
    """ This class provides a container to store all necessary data to perform
    MVPA analyses. These are the data samples, as well as the labels
    associated with these patterns. Additionally samples can be grouped into
    chunks.
    """
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
            lcl_dsattr = copy.deepcopy(dsattr)
        else:
            # shallow copy
            # XXX? same here
            lcl_data = copy.copy(dsattr)

        self.__data = lcl_data
        """What make a dataset."""
        self.__dsattr = lcl_dsattr
        """Dataset attriibutes."""

        # store samples (and possibly transform/reshape/retype them)
        if not samples == None:
            if __debug__:
                if self.__data.has_key('samples'):
                    debug('DS',
                          "`Data` dict has `samples` (%s) but there is also" +
                          " __init__ parameter `samples` which overrides " +
                          " stored in `data`" % (`self.__data['samples'].shape`))
            self.__data['samples'] = self._shapeSamples(samples, dtype,
                                                        copy_samples)

        if not labels == None:
            self.__data['labels'] = \
                self._expandSampleAttribute(labels, 'labels')
        if chunks == None and not self.__data.has_key('chunks'):
            # if no chunk information is given assume that every pattern
            # is its own chunk
            self.__data['chunks'] = N.arange(self.nsamples)
        if not chunks == None:
            self.__data['chunks'] = \
                self._expandSampleAttribute(chunks, 'chunks')

        if check_data:
            self._checkData()

        # XXX make those two go away
        self.__uniqueLabels = None
        self.__uniqueChunks = None


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
        """Checks all elements in the data dictionary whether
        their length is consistent with the number of samples in the dataset.
        """
        for k, v in self.__data.iteritems():
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


    def __repr__(self):
        """ String summary over the object
        """
        return """Dataset / %s %d x %d, %d uniq labels, %d uniq chunks""" % \
               (self.samples.dtype, self.nsamples, self.nfeatures,
                len(self.uniquelabels), len(self.uniquechunks))


    def __iadd__( self, other ):
        """ Merge the samples of one Dataset object to another (in-place).

        No dataset attributes will be merged!
        """
        if not self.nfeatures == other.nfeatures:
            raise ValueError, "Cannot add Dataset, because the number of " \
                              "feature do not match."

        # concatenate all sample attributes
        for k, v in self.__data.iteritems():
            self.__data[k] = N.concatenate((v, other.__data[k]), axis=0)

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
        out.__init__(data=self.__data,
                     dsattr=self.__dsattr,
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
        new_data = self.__data.copy()

        # assign the selected features -- data is still shared with
        # current dataset
        new_data['samples'] = self.__data['samples'][:, ids]

        # create a new object of the same type it is now and NOT onyl Dataset
        dataset = super(Dataset, self).__new__(self.__class__)

        # now init it: to make it work all Dataset contructors have to accept
        # Class(data=Dict, dsattr=Dict)
        dataset.__init__(data=new_data,
                         dsattr=self.__dsattr)

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
        for k, v in self.__data.iteritems():
            data[k] = v[mask,]

        # create a new object of the same type it is now and NOT onyl Dataset
        dataset = super(Dataset, self).__new__(self.__class__)

        # now init it: to make it work all Dataset contructors have to accept
        # Class(data=Dict, dsattr=Dict)
        dataset.__init__(data=data,
                         dsattr=self.__dsattr)

        return dataset


    def getSampleIdsByLabels(self, labels):
        """ Return indecies of samples given a list of labels
        """

        if not operator.isSequenceType(labels):
            labels = [ labels ]

        # TODO: compare to plain for loop through the labels
        #       on a real data example
        sel = N.array([], dtype=N.int16)
        for label in labels:
            sel = N.concatenate((
                        sel, N.where(self.__data['labels']==label)[0]))

        # place samples in the right order
        sel.sort()

        return sel


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
            if self.__data['origlabels'] == None:
                raise RuntimeError, 'Cannot restore labels. ' \
                                    'randomizedRegressors() has never been ' \
                                    'called with status == True.'
            self._setLabels(self.__data['origlabels'])
            self.__data['origlabels'] = None
        else:
            # permute labels per origin

            # make a backup of the original labels
            self.__data['origlabels'] = self.__data['labels'].copy()

            # now scramble the rest
            if perchunk:
                for o in self.uniquechunks:
                    self.__data['labels'][self.chunks == o ] = \
                        N.random.permutation( self.labels[ self.chunks == o ] )
                # to recompute uniquelabels
                self._setLabels(self.__data['labels'])
            else:
                self._setLabels(N.random.permutation(self.__data['labels']))


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


    def _setLabels(self, labels):
        """ Sets labels and recomputes uniquelabels
        """
        self.__data['labels'] = labels
        self.__uniqueLabels = None # None!since we might not need them


    def _setChunks(self, chunks):
        """ Sets chunks and recomputes uniquechunks
        """
        self.__data['chunks'] = chunks
        self.__uniqueChunks = None # None!since we might not need them


    def getNSamples( self ):
        """ Currently available number of patterns.
        """
        return self.__data['samples'].shape[0]


    def getNFeatures( self ):
        """ Number of features per pattern.
        """
        return self.__data['samples'].shape[1]


    def getSamples( self ):
        """ Returns the sample matrix.
        """
        return self.__data['samples']


    def getLabels( self ):
        """ Returns the label vector.
        """
        return self.__data['labels']


    def getChunks( self ):
        """ Returns the sample chunking vector.

        Each unique value in this vector defines a group of samples.
        """
        return self.__data['chunks']


    def getUniqueLabels(self):
        """ Returns an array with all unique class labels in the labels vector.

        Late evaluation for speedup in cases when uniquelabels is not needed
        """
        if self.__uniqueLabels is None:
            self.__uniqueLabels = N.unique( self.labels )
            assert(not self.__uniqueLabels is None)
        return self.__uniqueLabels


    def getUniqueChunks( self ):
        """ Returns an array with all unique labels in the chunk vector.

        Late evaluation for speedup in cases when uniquechunks is not needed
        """
        if self.__uniqueChunks is None:
            self.__uniqueChunks = N.unique( self.chunks )
            assert(not self.__uniqueChunks is None)
        return self.__uniqueChunks


    def getNSamplesPerLabel( self ):
        """ Returns the number of samples per unique label.
        """
        return [ len(self.samples[self.labels == l]) \
                    for l in self.uniquelabels ]


    def getNSamplesPerChunk( self ):
        """ Returns the number of samples per unique chunk value.
        """
        return [ len(self.samples[self.chunks == c]) \
                    for c in self.uniquechunks ]


    def setSamplesDType(self, dtype):
        """Set the data type of the samples array.
        """
        if self.__data['samples'].dtype != dtype:
            self.__data['samples'] = self.__data['samples'].astype(dtype)


    # read-only class properties
    samples         = property( fget=getSamples )
    labels          = property( fget=getLabels )
    chunks          = property( fget=getChunks )
    nsamples        = property( fget=getNSamples )
    nfeatures       = property( fget=getNFeatures )
    uniquelabels    = property( fget=getUniqueLabels )
    uniquechunks    = property( fget=getUniqueChunks )
    samplesperlabel = property( fget=getNSamplesPerLabel )
    samplesperchunk = property( fget=getNSamplesPerChunk )
