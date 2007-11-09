#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset container"""

import numpy as N
import operator
import random

# TODO? yoh: There is too much in common between chunks and labels....
#   michael: they might be two instances of some label that is attached to
#            a data sample. In some cases 'chunks' is not necessary. There also
#            might be cases where more than labels+chunks is necessary. Maybe
#            we should move towards something like 'sample_properties', where
#            you can have any number of them in each dataset. And maybe we
#            should put them in a dict so we can access them by name and just
#            make a policy that labels should be 'labels' and chunks should be
#            'chunks'. But addtionally there might be more of them and 3rd party
#            algorithms might use them. To summarize:
#
#            sample_properties = \
#               {'name': <1d container with len() == samples.shape[0], ... }
#
#            This would also allow for non-numerical properties. But we might
#            want to enforce ndarray for 'labels', not sure though.
#            Thinking again, selectSamples() really relies on the slicing
#            capabilities of ndarray....

class Dataset(object):
    """ This class provides a container to store all necessary data to perform
    MVPA analyses. These are the data samples, as well as the labels
    associated with these patterns. Additionally samples can be grouped into
    chunks.
    """

    # Common parameters for all subclasses. To don't replicate, __init__.__doc__
    # has to be extended with them after it is defined
    # TODO: discard such way or accept and introduce to derived methods...
    __initparams__ = \
        """
        `samples` -
        `labels`  -
        `chunks`  -
        `dtype`   - if None -- do not change data type if samples
                  is an ndarray. Otherwise convert samples to dtype"""

    def __init__(self, samples, sattr=None, dsattr=None, dtype=None, \
                 labels=None, chunks=None, check_sattr=True):
        """
        - `samples`: 2d array (samples x features).
        - `sattr`: Dict with an arbitrary number of entries. To value for
                   each key in the dict has to be a 1d ndarray with the
                   same length as the number of rows in the samples array.
                   Each value in those 1d arrays is assigned to the
                   corresponding sample.
        - `ds_attr`: Dictionary of dataset attributes. An arbitrary number of
                     arbitrarily named and typed objects can be stored here.
        - `dtype`: If None -- do not change data type if samples
                   is an ndarray. Otherwise convert samples to dtype.
        - `labels`: array or scalar value
        - `chunks`: array or scalar value
                   """
        # initialize containers
        self.__samples = None
        """Samples array."""
        self.__sattr = {}
        """Sample attributes."""
        self.__dsattr = {}
        """Dataset attriibutes."""

        if not dsattr == None:
            self.__dsattr = dsattr

        # put samples array into correct shape
        # 1d arrays or simple sequences are assumed to be a single pattern
        if (not isinstance(samples, N.ndarray)):
            samples = N.array(samples, ndmin=2)
        else:
            if samples.ndim < 2 \
                   or (not dtype is None and dtype != samples.dtype):
                if dtype is None:
                    dtype = samples.dtype
                samples = N.array(samples, ndmin=2, dtype=dtype)

        # only samples x features matrices are supported
        if len(samples.shape) > 2:
            raise ValueError, "Only (samples x features) -> 2d sample " \
                            + "are supported. Consider MappedDataset if " \
                            + "applicable."

        # done -> store
        self.__samples = samples

        # if there is no ready sample attributes dict try using some keyword
        # arguments to initialize one
        if sattr == None:
            if not labels == None:
                self.__sattr['labels'] = \
                    self._expandSampleAttribute(labels, 'labels')
            if chunks == None:
                # if no chunk information is given assume that every pattern
                # is its own chunk
                self.__sattr['chunks'] = N.arange(len(self.__samples))
            else:
                self.__sattr['chunks'] = \
                    self._expandSampleAttribute(chunks, 'chunks')
        elif isinstance(sattr, dict):
            # if there is one, use provided attributes dict
            self.__sattr = sattr
        else:
            raise ValueError, "Don't mess with 'sattr'!!!"

        if check_sattr:
            self._checkSampleAttributes

        # XXX make those two go away
        self.__uniqueLabels = None
        self.__uniqueChunks = None


    def _checkSampleAttributes(self):
        """Checks all elements in the sample attributes dictionary whether
        their length matches the number of samples in the dataset.
        """
        for k, v in self.__sattr.iteritems():
            if not len(v) == len(self.__samples):
                raise ValueError, \
                      "Length of sample attribute '%s' does not " \
                      "match the number of samples in the dataset." % k


    def _expandSampleAttribute(self, attr, attr_name):
        """If a sample attribute is given as a scalar expand/repeat it to a
        length matching the number of samples in the dataset.
        """
        try:
            if len(attr) != len(self.__samples):
                raise ValueError, \
                      "Length of sample attribute '%s' [%d]" \
                      % (attr_name, len(attr)) \
                      + " has to match the number of samples" \
                      + " [%d]." % len(self.__samples)
            # store the sequence as array
            return N.array(attr)

        except TypeError:
            # make sequence of identical value matching the number of
            # samples
            return N.repeat( attr, len( self.samples ) )


    def __repr__(self):
        """ String summary over the object
        """
        return """Dataset / %s %d x %d, %d uniq labels, %d uniq chunks""" % \
               (self.samples.dtype, self.nsamples, self.nfeatures,
                len(self.uniquelabels), len(self.uniquechunks))


    def __iadd__( self, other ):
        """ Merge the samples of one Dataset object to another (in-place).

        Please note that the samples, labels and chunks are simply
        concatenated to create a Dataset object that contains the patterns of
        both objects. No further processing is done. In particular the chunk
        values are not modified: Samples with the same origin from both
        Datasets will still share the same chunk.
        """
        if not self.nfeatures == other.nfeatures:
            raise ValueError, "Cannot add Dataset, because the number of " \
                              "feature do not match."

        self.__samples = \
            N.concatenate( ( self.samples, other.samples ), axis=0)

        # concatenate all sample attributes
        for k, v in self.__sattr.iteritems():
            self.__sattr[k] = N.concatenate((v, other.__sattr[k]), axis=0)

        return self


    def __add__( self, other ):
        """ Merge the samples two Dataset objects.

        Please note that the samples, labels and chunks are simply
        concatenated to create a Dataset object that contains the patterns of
        both objects. No further processing is done. In particular the chunk
        values are not modified: Samples with the same origin from both
        Datasets will still share the same chunk.
        """
        # create a new object of the same type it is now and NOT onyl Dataset
        out = super(Dataset, self).__new__(self.__class__)

        # XXX need to make copy of sample attributes otherwise
        # it will result in modified attributes in 'self', because of the
        # behaviour of __iadd__
        # maybe reimplment this whole thing!!
        sattr = {}
        for k, v in self.__sattr.iteritems():
            sattr[k] = v.copy()

        # now init it: to make it work all Dataset contructors have to accept
        # Class(ndarray, sattr=Dict, dsattr=Dict)
        out.__init__(self.__samples,
                     sattr=sattr,
                     dsattr=self.__dsattr)

        out += other

        return out


    def selectFeatures( self, ids ):
        """ Select a number of features from the current set.

        'ids' is a list of feature IDs

        Returns a new Dataset object with a view of the original samples
        array (no copying is performed).
        """
        # create a new object of the same type it is now and NOT onyl Dataset
        dataset = super(Dataset, self).__new__(self.__class__)

        # now init it: to make it work all Dataset contructors have to accept
        # Class(ndarray, sattr=Dict, dsattr=Dict)
        dataset.__init__(self.__samples[:, ids],
                         sattr=self.__sattr,
                         dsattr=self.__dsattr)

        return dataset


    def selectSamples( self, mask ):
        """ Choose a subset of samples.

        Returns a new dataset object containing the selected sample
        subset.
        """
        # without having a sequence a index the masked sample array would
        # loose its 2d layout
        if not operator.isSequenceType( mask ):
            mask = [mask]

        # mask all sample attributes
        sattr = {}
        for k, v in self.__sattr.iteritems():
            sattr[k] = v[mask,]

        # create a new object of the same type it is now and NOT onyl Dataset
        dataset = super(Dataset, self).__new__(self.__class__)

        # now init it: to make it work all Dataset contructors have to accept
        # Class(ndarray, sattr=Dict, dsattr=Dict)
        dataset.__init__(self.__samples[mask,],
                         sattr=sattr,
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
                        sel, N.where(self.__sattr['labels']==label)[0]))

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
            if self.__sattr['origlabels'] == None:
                raise RuntimeError, 'Cannot restore labels. ' \
                                    'randomizedRegressors() has never been ' \
                                    'called with status == True.'
            self._setLabels(self.__sattr['origlabels'])
            self.__sattr['origlabels'] = None
        else:
            # permute labels per origin

            # make a backup of the original labels
            self.__sattr['origlabels'] = self.__sattr['labels'].copy()

            # now scramble the rest
            if perchunk:
                for o in self.uniquechunks:
                    self.__sattr['labels'][self.chunks == o ] = \
                        N.random.permutation( self.labels[ self.chunks == o ] )
                # to recompute uniquelabels
                self._setLabels(self.__sattr['labels'])
            else:
                self._setLabels(N.random.permutation(self.__sattr['labels']))


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
        self.__sattr['labels'] = labels
        self.__uniqueLabels = None # None!since we might not need them


    def _setChunks(self, chunks):
        """ Sets chunks and recomputes uniquechunks
        """
        self.__sattr['chunks'] = chunks
        self.__uniqueChunks = None # None!since we might not need them


    def getNSamples( self ):
        """ Currently available number of patterns.
        """
        return self.samples.shape[0]


    def getNFeatures( self ):
        """ Number of features per pattern.
        """
        return self.samples.shape[1]


    def getSamples( self ):
        """ Returns the sample matrix.
        """
        return self.__samples


    def getLabels( self ):
        """ Returns the label vector.
        """
        return self.__sattr['labels']


    def getChunks( self ):
        """ Returns the sample chunking vector.

        Each unique value in this vector defines a group of samples.
        """
        return self.__sattr['chunks']


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
        if self.__samples.dtype != dtype:
            self.__samples = self.__samples.astype(dtype)


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
