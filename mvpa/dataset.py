### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Dataset container
#
#    Copyright (C) 2006-2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2 of the License, or (at your option) any later version.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as N
import operator
import random


class Dataset(object):
    """ This class provides a container to store all necessary data to perform
    MVPA analyses. These are the data samples, as well as the regressors
    associated with these patterns. Additionally samples can be grouped into
    chunks.
    """
    def __init__(self, samples, regs, chunks ):
        """ Initialize the Dataset.

        Parameters:
          samples -
          regs    -
          chunks  -
        """
        # initialize containers
        self.__samples = None
        self.__regs = None
        self.__chunks = None
        self.__origregs = None

        # 1d arrays or simple sequences are assumed to be a single pattern
        if (not isinstance(samples, N.ndarray)) or samples.ndim < 2:
            samples = N.array( samples, ndmin=2 )

        # only samples x features matrices are supported
        if len(samples.shape) > 2:
            raise ValueError, "Only (samples x features) -> 2d sample " \
                            + "are supported. Consider MappedDataset if " \
                            + "applicable."

        # done -> store
        self.__samples = samples

        # check if regs is supplied as a sequence
        try:
            if len( regs ) != len( self.samples ):
                raise ValueError, "Length of 'reg' has to match the number" \
                                  " of patterns."
            # store the sequence as array
            regs = N.array( regs )

        except TypeError:
            # make sequence of identical value matching the number of patterns
            regs = N.repeat( regs, len( self.samples ) )

        # done -> store
        self.__regs = regs

        # if no chunk information is given assume that every pattern is its
        # own chunk
        if chunks == None:
            chunks = N.arange( len( self.samples ) )
        else:
            try:
                if len( chunks ) != len( self.samples ):
                    raise ValueError, "Length of 'chunks' has to match the" \
                                      " number of samples."
                # store the sequence as array
                chunks = N.array( chunks )

            except TypeError:
                # make sequence of identical value matching the number of
                # patterns
                chunks = N.repeat( chunks, len( self.samples ) )

        # done -> store
        self.__chunks = chunks


    def __iadd__( self, other ):
        """ Merge the samples of one Dataset object to another (in-place).

        Please note that the samples, regressors and chunks are simply
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
        self.__regs = \
            N.concatenate( ( self.regs, other.regs ), axis=0)
        self.__chunks = \
            N.concatenate( ( self.chunks, other.chunks ), axis=0)

        return self


    def __add__( self, other ):
        """ Merge the samples two Dataset objects.

        Please note that the samples, regressors and chunks are simply
        concatenated to create a Dataset object that contains the patterns of
        both objects. No further processing is done. In particular the chunk
        values are not modified: Samples with the same origin from both
        Datasets will still share the same chunk.
        """
        out = Dataset( self.__samples,
                       self.__regs,
                       self.__chunks )

        out += other

        return out


    def selectFeatures( self, ids ):
        """ Select a number of features from the current set.

        'ids' is a list of feature IDs

        Returns a new Dataset object with a view of the original samples
        array (no copying is performed).
        """
        return Dataset( self.__samples[:, ids],
                        self.__regs,
                        self.__chunks )


    def selectSamples( self, mask ):
        """ Choose a subset of samples.

        Returns a new Dataset object containing the selected sample
        subset.
        """
        # without having a sequence a index the masked sample array would
        # loose its 2d layout
        if not operator.isSequenceType( mask ):
            mask = [mask]

        return Dataset( self.samples[mask,],
                        self.regs[mask,],
                        self.chunks[mask,] )


    def permutatedRegressors( self, status, perchunk = True ):
        """ Permutate the regressors.

        Calling this method with 'status' set to True, the regressors are
        permutated among all samples.

        If 'perorigin' is True permutation is limited to samples sharing the
        same chunk value. Therefore only the association of a certain sample
        with a regressor is permutated while keeping the absolute number of
        occurences of each regressor value within a certain chunk constant.

        If 'status' is False the original regressors are restored.
        """
        if not status:
            # restore originals
            if self.__origregs == None:
                raise RuntimeError, 'Cannot restore regressors. ' \
                                    'randomizedRegressors() has never been ' \
                                    'called with status == True.'
            self.__regs = self.__origregs
            self.__origregs = None
        else:
            # permutate regs per origin

            # make a backup of the original regressors
            self.__origregs = self.__regs.copy()

            # now scramble the rest
            if perchunk:
                for o in self.chunklabels:
                    self.__regs[self.chunks == o ] = \
                        N.random.permutation( self.regs[ self.chunks == o ] )
            else:
                self.__regs = N.random.permutation( self.__regs )


    def getRandomSamples( self, nperreg ):
        """ Select a random set of samples.

        If 'nperreg' is an integer value, the specified number of samples is
        randomly choosen from the group of samples sharing a unique regressor
        value ( total number of selected samples: nperreg x len(reglabels).

        If 'nperreg' is a list which's length has to match the number of unique
        regressor labels. In this case 'nperreg' specifies the number of
        samples that shall be selected from the samples with the corresponding
        regressor label.

        The method returns a Dataset object containing the selected
        samples.
        """
        # if interger is given take this value for all classes
        if isinstance(nperreg, int):
            nperreg = [ nperreg for i in self.reglabels ]

        sample = []
        # for each available class
        for i,r in enumerate(self.reglabels):
            # get the list of pattern ids for this class
            sample += random.sample( (self.regs == r).nonzero()[0],
                                     nperreg[i] )

        return self.selectSamples( sample )


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


    def getRegs( self ):
        """ Returns the regressors vector.
        """
        return self.__regs


    def getChunks( self ):
        """ Returns the sample chunking vector.

        Each unique value in this vector defines a group of samples.
        """
        return self.__chunks


    def getRegLabels( self ):
        """ Returns an array with all unique class labels in the regressors
        vector.
        """

        return N.unique( self.regs )


    def getChunkLabels( self ):
        """ Returns an array with all unique labels in the chunk vector.
        """

        return N.unique( self.chunks )


    def getNSamplesPerReg( self ):
        """ Returns the number of patterns per regressor label.
        """
        return [ len(self.samples[self.regs == r]) \
                    for r in self.reglabels ]


    def getNSamplesPerChunk( self ):
        """ Returns the number of patterns per regressor label.
        """
        return [ len(self.samples[self.regs == r]) \
                    for r in self.reglabels ]


    # read-only class properties
    samples         = property( fget=getSamples )
    regs            = property( fget=getRegs )
    chunks          = property( fget=getChunks )
    nsamples        = property( fget=getNSamples )
    nfeatures       = property( fget=getNFeatures )
    reglabels       = property( fget=getRegLabels )
    chunklabels     = property( fget=getChunkLabels )
    samplesperreg   = property( fget=getNSamplesPerReg )
    samplesperchunk = property( fget=getNSamplesPerChunk )
