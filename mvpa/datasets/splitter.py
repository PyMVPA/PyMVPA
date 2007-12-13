#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Base class of all dataset splitter"""

__docformat__ = 'restructuredtext'

import numpy as N

import mvpa.misc.support as support


class Splitter(object):
    """Base class of a data splitter.

    Each splitter should be initialized with all its necessary parameters. The
    final splitting is done running the splitter object on a certain Dataset
    via __call__(). This method has to be implemented like a generator, i.e. it
    has to return every possible split with a yield() call.

    Each split has to be returned as a tuple of Dataset(s). The properties
    of the splitted dataset may vary between implementations. It is possible
    to declare tuple element as 'None'. 

    Please note, that even if there is only one Dataset returned it has to be
    an element in a tuple and not just the Dataset object!
    """
    def __init__(self):
        """Does nothing.
        """
        pass


    def __call__(self, dataset):
        """
        """
        raise NotImplementedError


    @staticmethod
    def splitWorkingSpareDataset(dataset, sparechunks):
        """Split a dataset into a working and a spare set separating
        the samples of some chunks.

        :Parameters:
          - `dataset`: source `Dataset`
          - `sparechunks`: sequence with chunk values of the dataset that
            shall form the spare set.

        :Returns: Tuple of Datasets (working, spare).
        """
        # build a boolean selector vector to choose training and
        # test data
        spare_filter =  \
            N.array([ i in sparechunks for i in dataset.chunks ])

        # split data into working and spare set
        wset = dataset.selectSamples(N.logical_not(spare_filter))
        sset = dataset.selectSamples(spare_filter)

        return wset, sset


    @staticmethod
    def selectSampleSubset(dataset, samplesize):
        """Select a number of patterns for each label value.

        :Parameters:
          - `dataset`: `Dataset` object with the source samples
          - `samplesize`: number of to be selected samples. Two special values
            are recognized. None is off (all samples are
            selected), 'auto' sets sample size to highest
            possible number of samples that can be provided by
            each label class.

        :Returns: `Dataset` object with the selected samples
        """
        if not samplesize == None:
            # determine number number of patterns per class
            if samplesize == 'auto':
                samplesize = \
                   N.array(dataset.samplesperlabel).min()

            # finally select the patterns
            samples = dataset.getRandomSamples(samplesize)
        else:
            # take all training patterns in the sampling run
            samples = dataset

        return samples



class NoneSplitter(Splitter):
    """This is a dataset splitter that does NOT split. It simply returns the
    full dataset that it is called with.
    """
    def __call__(self, dataset):
        """This splitter returns the passed dataset as the second element of
        a 2-tuple. The first element of that tuple will always be 'None'.
        """
        return (None, dataset)



class OddEvenSplitter(Splitter):
    """Split a dataset into odd and even chunks.

    The splitter yields to splits: first (odd, even) and second (even, odd).
    """
    def __init__(self):
        """Cheap init -- nothing special
        """
        Splitter.__init__(self)


    def __call__(self, dataset):
        """Splits the dataset.

        This method behaves like a generator and returns two iterations: first
        (odd,even) then (even,odd).
        """
        odd_chunks = dataset.uniquechunks[(dataset.uniquechunks % 2) == True]
        even_chunks = dataset.uniquechunks[(dataset.uniquechunks % 2) == False]

        yield Splitter.splitWorkingSpareDataset(dataset, even_chunks)
        yield Splitter.splitWorkingSpareDataset(dataset, odd_chunks)



class NFoldSplitter(Splitter):
    """Generic N-fold data splitter.

    Terminology:
      - working set
      - spare set
    """
    def __init__(self,
                 cvtype = 1,
                 nworkingsamples = None,
                 nsparesamples = None,
                 nrunsperfold = 1,
                 permute = False):
        """Initialize the N-fold splitter.

        :Parameters:
          cvtype: Int
            Type of cross-validation: N-(cvtype)
          nworkingsamples
            Number of working set samples to be included in each
            fold. Please see the setNWorkingSetSamples() method for
            special arguments.
          nsparesamples
            Number of spare set samples to be included in each
            fold. Please see the setNSpareSetSamples() method for
            special arguments.
          nrunsperfold
            Number of times samples for each fold are chosen. This
            is mostly useful if a subset of the available samples
            is used in each fold and the subset is randomly
            selected for each run (see the nworkingsamples
            and nsparesamples arguments).
          permute : bool
            If set to `True`, the labels of each generated dataset
            will be permuted on a per-chunk basis.

        """
        Splitter.__init__(self)

        # pylint happyness block
        self.__cvtype = cvtype
        self.__working_samplesize = None
        self.__runsperfold = nrunsperfold
        self.__spare_samplesize = None
        self.__permute = permute

        # pattern sampling status vars
        self.setNWorkingSetSamples(nworkingsamples)
        self.setNSpareSetSamples(nsparesamples)



    def __repr__(self):
        """String summary over the object
        """
        return \
          "%d-fold splitter / work:%s runs-per-fold:%d spare:%s permute:%s " \
          % (self.__cvtype, self.__working_samplesize,
            self.__runsperfold, self.__spare_samplesize,
            self.__permute)


    def setNWorkingSetSamples(self, samplesize):
        """None is off, 'auto' sets sample size to highest possible number
        of patterns that can be provided by each class.
        """
        # check if automization is requested
        if isinstance(samplesize, str):
            if not samplesize == 'auto':
                raise ValueError, "Only 'auto' is a valid string argument."

        self.__working_samplesize = samplesize


    def setNSpareSetSamples(self, samplesize):
        """None is off, 'auto' sets sample size to highest possible number
        of patterns that can be provided by each class.
        """
        # check if automization is requested
        if isinstance(samplesize, str):
            if not samplesize == 'auto':
                raise ValueError, "Only 'auto' is a valid string argument."

        self.__spare_samplesize = samplesize


    def __call__(self, dataset):
        """Splits the dataset.

        This method behaves like a generator.
        """
        # get the list of all combinations of to be excluded chunks
        cv_list = \
            support.getUniqueLengthNCombinations(dataset.uniquechunks,
                                                  self.__cvtype)

        # do cross-validation
        for exclude in cv_list:
            # split into working and spare set for this fold
            wset, sset = \
                NFoldSplitter.splitWorkingSpareDataset(dataset, exclude)

            # do the sampling for this CV fold
            for run in xrange(self.__runsperfold):
                # permute the labels in training and test dataset
                if self.__permute:
                    wset.permuteLabels(True, perchunk=True)
                    sset.permuteLabels(True, perchunk=True)

                # choose a training pattern sample
                wset_samples = NFoldSplitter.selectSampleSubset(
                                wset,
                                self.__working_samplesize)

                # choose a test pattern sample
                sset_samples = NFoldSplitter.selectSampleSubset(
                                sset,
                                self.__spare_samplesize)

                yield wset_samples, sset_samples
