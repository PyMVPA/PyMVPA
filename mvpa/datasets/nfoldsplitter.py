#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Generic N-M fold combination splitter"""

import numpy as N

import mvpa.misc.support as support
from mvpa.datasets.splitter import Splitter

class NFoldSplitter(Splitter):
    """ Generic N-fold data splitter.

    Terminology:
        working set
        spare set
    """
    def __init__( self,
                  cvtype = 1,
                  nworkingsamples = None,
                  nsparesamples = None,
                  nrunsperfold = 1,
                  permute = False ):
        """
        Initialize the N-fold splitter.

          cvtype:     Type of cross-validation: N-(cvtype)
          nworkingsamples:
                      Number of working set samples to be included in each
                      fold. Please see the setNWorkingSetSamples() method for
                      special arguments.
          nsparesamples:
                      Number of spare set samples to be included in each
                      fold. Please see the setNSpareSetSamples() method for
                      special arguments.
          nrunsperfold:
                      Number of times samples for each fold are chosen. This
                      is mostly useful if a subset of the available samples
                      is used in each fold and the subset is randomly
                      selected for each run (see the nworkingsamples
                      and nsparesamples arguments).
          permute:  If set to True, the labels of each generated dataset
                      will be permuted on a per-chunk basis.
        """
        Splitter.__init__(self)

        # pylint happyness block
        self.__cvtype = None
        self.__working_samplesize = None
        self.__runsperfold = None
        self.__spare_samplesize = None

        # pattern sampling status vars
        self.setNWorkingSetSamples( nworkingsamples )
        self.setNSpareSetSamples( nsparesamples )
        self.setNRunsPerFold( nrunsperfold )
        self.setCVType( cvtype )

        self.__permute = permute


    def __repr__(self):
        """ String summary over the object
        """
        return \
          "%d-fold splitter / work:%s runs-per-fold:%d spare:%s permute:%s "%\
               (self.__cvtype, self.__working_samplesize,
                self.__runsperfold, self.__spare_samplesize,
                self.__permute)


    def setNWorkingSetSamples( self, samplesize ):
        """ None is off, 'auto' sets sample size to highest possible number
        of patterns that can be provided by each class.
        """
        # check if automization is requested
        if isinstance(samplesize, str):
            if not samplesize == 'auto':
                raise ValueError, "Only 'auto' is a valid string argument."

        self.__working_samplesize = samplesize


    def setNSpareSetSamples( self, samplesize ):
        """ None is off, 'auto' sets sample size to highest possible number
        of patterns that can be provided by each class.
        """
        # check if automization is requested
        if isinstance(samplesize, str):
            if not samplesize == 'auto':
                raise ValueError, "Only 'auto' is a valid string argument."

        self.__spare_samplesize = samplesize


    def setNRunsPerFold( self, runs ):
        """ Set the number of runs that are performed per fold.
        """
        self.__runsperfold = runs


    def setCVType( self, cvtype ):
        """ Set the cross-validation type.

        (N-'type')-fold cross-validation.
        """
        self.__cvtype = cvtype


    def getNSamplesPerFold( self, dataset ):
        """ Returns a tuple of two arrays with the number of samples per
        unique label value and fold. The first array lists the available
        working set samples and the second array the spare samples.

        Array rows: folds, columns: labels
        """
        # get the list of all combinations of to be excluded folds
        cv_list = support.getUniqueLengthNCombinations(
                        dataset.uniquechunks, self.__cvtype)

        nwsamples = N.zeros( (len(cv_list), len(dataset.uniquelabels) ) )
        nssamples = N.zeros( nwsamples.shape )

        for fold, exclude in enumerate(cv_list):
            # build a boolean selector vector to choose training and
            # test data for this CV fold
            exclude_filter =  \
                N.array([ i in exclude for i in dataset.chunks ])

            # split data into working and spare set
            wset = \
                dataset.selectSamples(
                    N.logical_not(exclude_filter) )
            sset = dataset.selectSamples( exclude_filter )

            nwsamples[fold] = wset.getNSamplesPerReg()
            nssamples[fold] = sset.getNSamplesPerReg()

        return nwsamples, nssamples


    @staticmethod
    def splitWorkingSpareDataset( dataset, sparechunks ):
        """ Split a dataset into a working and a spare set separating
        the samples of some chunks.

        Parameter:
            dataset      - source Dataset
            spare_chunks - sequence with chunk values of the dataset that
                           shall form the spare set.

        Returns:
            Tuple of Datasets (working, spare).
        """
        # build a boolean selector vector to choose training and
        # test data
        spare_filter =  \
            N.array([ i in sparechunks for i in dataset.chunks ])

        # split data into working and spare set
        wset = dataset.selectSamples( N.logical_not( spare_filter) )
        sset = dataset.selectSamples( spare_filter )

        return wset, sset


    @staticmethod
    def selectSampleSubset( dataset, samplesize ):
        """ Select a number of patterns for each label value.

        Parameter:
            dataset    - Dataset object with the source samples
            samplesize - number of to be selected samples. Two special values
                         are recognized. None is off (all samples are
                         selected), 'auto' sets sample size to highest
                         possible number of samples that can be provided by
                         each label class.

        Returns:
            - Dataset object with the selected samples
        """
        if not samplesize == None:
            # determine number number of patterns per class
            if samplesize == 'auto':
                samplesize = \
                   N.array( dataset.getNSamplesPerReg() ).min()

            # finally select the patterns
            samples = dataset.getRandomSamples( samplesize )
        else:
            # take all training patterns in the sampling run
            samples = dataset

        return samples


    def __call__( self, dataset ):
        """ Splits the dataset.

        This method behaves like a generator.
        """
        # get the list of all combinations of to be excluded chunks
        cv_list = \
            support.getUniqueLengthNCombinations( dataset.uniquechunks,
                                                  self.__cvtype )

        # do cross-validation
        for exclude in cv_list:
            # split into working and spare set for this fold
            wset, sset = \
                NFoldSplitter.splitWorkingSpareDataset( dataset, exclude )

            # do the sampling for this CV fold
            for run in xrange( self.__runsperfold ):
                # permute the labels in training and test dataset
                if self.__permute:
                    wset.permutedRegressors( True, perchunk=True )
                    sset.permutedRegressors( True, perchunk=True )

                # choose a training pattern sample
                wset_samples = NFoldSplitter.selectSampleSubset(
                                wset,
                                self.__working_samplesize )

                # choose a test pattern sample
                sset_samples = NFoldSplitter.selectSampleSubset(
                                sset,
                                self.__spare_samplesize )

                yield wset_samples, sset_samples
