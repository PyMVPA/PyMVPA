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
    to declare a tuple element as 'None'. 

    Please note, that even if there is only one Dataset returned it has to be
    an element in a tuple and not just the Dataset object!
    """
    def __init__(self,
                 nworkingsamples=None,
                 nvalidationsamples=None,
                 nrunspersplit=1,
                 permute=False):
        """Initialize base splitter.

        :Parameters:
          nworkingsamples : int, str or None
            Number of working set samples to be included in each
            split. Please see the setNWorkingSetSamples() method for
            special arguments.
          nvalidationsamples : int, str or None
            Number of validation set samples to be included in each
            split. Please see the setNSpareSetSamples() method for
            special arguments.
          nrunspersplit: int
            Number of times samples for each split are chosen. This
            is mostly useful if a subset of the available samples
            is used in each split and the subset is randomly
            selected for each run (see the `nworkingsamples`
            and `nvalidationsamples` arguments).
          permute : bool
            If set to `True`, the labels of each generated dataset
            will be permuted on a per-chunk basis.

        """
        # pylint happyness block
        self.__working_samplesize = None
        self.__runspersplit = nrunspersplit
        self.__validation_samplesize = None
        self.__permute = permute

        # pattern sampling status vars
        self.setNWorkingSetSamples(nworkingsamples)
        self.setNValidationSetSamples(nvalidationsamples)


    def setNWorkingSetSamples(self, samplesize):
        """None is off, 'auto' sets sample size to highest possible number
        of patterns that can be provided by each class.
        """
        # check if automization is requested
        if isinstance(samplesize, str):
            if not samplesize == 'auto':
                raise ValueError, "Only 'auto' is a valid string argument."

        self.__working_samplesize = samplesize


    def setNValidationSetSamples(self, samplesize):
        """None is off, 'auto' sets sample size to highest possible number
        of patterns that can be provided by each class.
        """
        # check if automization is requested
        if isinstance(samplesize, str):
            if not samplesize == 'auto':
                raise ValueError, "Only 'auto' is a valid string argument."

        self.__validation_samplesize = samplesize


    def _getSplitConfig(self, uniquechunks):
        """Each subclass has to implement this method. It gets a sequence with
        the unique chunk ids of a dataset and has to return a list of lists
        containing chunk ids to split into the validation set.
        """
        raise NotImplementedError


    def __call__(self, dataset):
        """Splits the dataset.

        This method behaves like a generator.
        """
        splitcfg = self._getSplitConfig(dataset.uniquechunks)

        # do cross-validation
        for split in splitcfg:
            wset, vset = \
                Splitter.splitDataset(dataset, split)

            # do the sampling for this split
            for run in xrange(self.__runspersplit):
                # permute the labels
                if self.__permute:
                    wset.permuteLabels(True, perchunk=True)
                    vset.permuteLabels(True, perchunk=True)

                # choose samples
                wset_samples = \
                    Splitter.selectSamples(wset, self.__working_samplesize)
                vset_samples = \
                    Splitter.selectSamples(vset,  self.__validation_samplesize)

                yield wset_samples, vset_samples


    @staticmethod
    def splitDataset(dataset, splitchunks):
        """Split a dataset by separating the samples of some chunks.

        :Parameters:
          dataset : Dataset
            This is this source dataset.
          splitchunks : list or other sequence
            Contains ids of chunks that shall be split into the validation
            dataset.

        :Returns: Tuple of Datasets (working, validation).
        """
        # build a boolean selector vector to choose select samples
        split_filter =  \
            N.array([ i in splitchunks for i in dataset.chunks ])
        wset_filter = N.logical_not(split_filter)

        # split data: return None if no samples are left
        # XXX: Maybe it should simply return an empty dataset instead, but
        #      keeping it this way for now, to maintain current behavior
        if (wset_filter == False).all():
            wset = None
        else:
            wset = dataset.selectSamples(wset_filter)
        if (split_filter == False).all():
            vset = None
        else:
            vset = dataset.selectSamples(split_filter)

        return wset, vset


    @staticmethod
    def selectSamples(dataset, samplesize):
        """Select a number of samples for each label value.

        :Parameters:
          dataset : Dataset
            Source samples.
          samplesize : int, str, None
            number of to be selected samples. Two special values are
            recognized. None is off (all samples are selected), 'auto' sets
            sample size to highest possible number of samples that can be
            provided by each label class.

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


    def __str__(self):
        """String summary over the object
        """
        return \
          "SplitterConfig: work:%s runs-per-split:%d validate:%s permute:%s" \
          % (self.__working_samplesize, self.__runspersplit,
             self.__validation_samplesize, self.__permute)



class NoneSplitter(Splitter):
    """This is a dataset splitter that does **not** split. It simply returns
    the full dataset that it is called with.

    The passed dataset is returned as the second element of the 2-tuple.
    The first element of that tuple will always be 'None'.
    """
    def __init__(self, **kwargs):
        """Cheap init -- nothing special
        """
        Splitter.__init__(self, **(kwargs))


    def _getSplitConfig(self, uniquechunks):
        """Return just one full split: no working set.
        """
        return [uniquechunks]


    def __str__(self):
        """String summary over the object
        """
        return \
          "NoneSplitter / " + Splitter.__str__(self)



class OddEvenSplitter(Splitter):
    """Split a dataset into odd and even chunks.

    The splitter yields to splits: first (odd, even) and second (even, odd).
    """
    def __init__(self, **kwargs):
        """Cheap init -- nothing special
        """
        Splitter.__init__(self, **(kwargs))


    def _getSplitConfig(self, uniquechunks):
        """Huka chaka!
        """
        return [uniquechunks[(uniquechunks % 2) == True],
                uniquechunks[(uniquechunks % 2) == False]]


    def __str__(self):
        """String summary over the object
        """
        return \
          "OddEvenSplitter / " + Splitter.__str__(self)



class NFoldSplitter(Splitter):
    """Generic N-fold data splitter.

    Terminology:
      - working set
      - spare set
    """
    def __init__(self,
                 cvtype = 1,
                 **kwargs):
        """Initialize the N-fold splitter.

        :Parameter:
          cvtype: Int
            Type of cross-validation: N-(cvtype)
          kwargs
            Addtional parameters are passed to the `Splitter` base class.
        """
        Splitter.__init__(self, **(kwargs))

        # pylint happyness block
        self.__cvtype = cvtype


    def __str__(self):
        """String summary over the object
        """
        return \
          "N-%d-FoldSplitter / " % self.__cvtype + Splitter.__str__(self)


    def _getSplitConfig(self, uniquechunks):
        """Returns proper split configuration for N-M fold split.
        """
        return support.getUniqueLengthNCombinations(uniquechunks,
                                                    self.__cvtype)
