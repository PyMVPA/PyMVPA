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
                 nfirstsamples=None,
                 nsecondsamples=None,
                 nrunspersplit=1,
                 permute=False,
                 attr='chunks'):
        """Initialize base splitter.

        :Parameters:
          nfirstsamples : int, str or None
            Number of first dataset samples to be included in each
            split. Please see the setNFirstSetSamples() method for
            special arguments.
          nsecondsamples : int, str or None
            Number of second dataset samples to be included in each
            split. Please see the setNSecondSetSamples() method for
            special arguments.
          nrunspersplit: int
            Number of times samples for each split are chosen. This
            is mostly useful if a subset of the available samples
            is used in each split and the subset is randomly
            selected for each run (see the `nfirstsamples`
            and `nsecondsamples` arguments).
          permute : bool
            If set to `True`, the labels of each generated dataset
            will be permuted on a per-chunk basis.
          attr : str
            Sample attribute used to determine splits.
        """
        # pylint happyness block
        self.__first_samplesize = None
        self.__second_samplesize = None
        self.__runspersplit = nrunspersplit
        self.__permute = permute
        self.__splitattr = attr

        # pattern sampling status vars
        self.setNFirstSetSamples(nfirstsamples)
        self.setNSecondSetSamples(nsecondsamples)


    def setNFirstSetSamples(self, samplesize):
        """None is off, 'auto' sets sample size to highest possible number
        of patterns that can be provided by each class.
        """
        # check if automization is requested
        if isinstance(samplesize, str):
            if not samplesize == 'auto':
                raise ValueError, "Only 'auto' is a valid string argument."

        self.__first_samplesize = samplesize


    def setNSecondSetSamples(self, samplesize):
        """None is off, 'auto' sets sample size to highest possible number
        of patterns that can be provided by each class.
        """
        # check if automization is requested
        if isinstance(samplesize, str):
            if not samplesize == 'auto':
                raise ValueError, "Only 'auto' is a valid string argument."

        self.__second_samplesize = samplesize


    def _getSplitConfig(self, uniqueattr):
        """Each subclass has to implement this method. It gets a sequence with
        the unique attribte ids of a dataset and has to return a list of lists
        containing attribute ids to split into the second dataset.
        """
        raise NotImplementedError


    def __call__(self, dataset):
        """Splits the dataset.

        This method behaves like a generator.
        """

        # do cross-validation
        for split in self.splitcfg(dataset):
            wset, vset = self.splitDataset(dataset, split)

            # do the sampling for this split
            for run in xrange(self.__runspersplit):
                # permute the labels
                if self.__permute:
                    wset.permuteLabels(True, perchunk=True)
                    vset.permuteLabels(True, perchunk=True)

                # choose samples
                wset_samples = \
                    Splitter.selectSamples(wset, self.__first_samplesize)
                vset_samples = \
                    Splitter.selectSamples(vset,  self.__second_samplesize)

                yield wset_samples, vset_samples


    def splitDataset(self, dataset, splitids):
        """Split a dataset by separating the samples where the configured
        sample attribute matches an element of `splitids`.

        :Parameters:
          dataset : Dataset
            This is this source dataset.
          splitids : list or other sequence
            Contains ids of a sample attribute that shall be split into the
            another dataset.

        :Returns: Tuple of splitted datasets.
        """
        # build a boolean selector vector to choose select samples
        split_filter =  \
            N.array([ i in splitids \
                for i in eval('dataset.' + self.__splitattr)])
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
          % (self.__first_samplesize, self.__runspersplit,
             self.__second_samplesize, self.__permute)


    def splitcfg(self, dataset):
        """Return splitcfg for a given dataset"""
        return self._getSplitConfig(eval('dataset.unique' + self.__splitattr))



class NoneSplitter(Splitter):
    """This is a dataset splitter that does **not** split. It simply returns
    the full dataset that it is called with.

    The passed dataset is returned as the second element of the 2-tuple.
    The first element of that tuple will always be 'None'.
    """

    _known_modes = ['first', 'second']
    
    def __init__(self, mode='second', **kwargs):
        """Cheap init -- nothing special

        :Parameters:
          mode
            Either 'first' or 'second' (default) -- which output dataset
            would actually contain the samples
        """
        Splitter.__init__(self, **(kwargs))

        if not mode in NoneSplitter._known_modes:
            raise ValueError, "Unknown mode %s for NoneSplitter" % mode
        self.__mode = mode


    def _getSplitConfig(self, uniqueattrs):
        """Return just one full split: no first or second dataset.
        """
        if self.__mode == 'second':
            return [uniqueattrs]
        else:
            return [[]]


    def __str__(self):
        """String summary over the object
        """
        return \
          "NoneSplitter / " + Splitter.__str__(self)



class OddEvenSplitter(Splitter):
    """Split a dataset into odd and even values of the sample attribute.

    The splitter yields to splits: first (odd, even) and second (even, odd).
    """
    def __init__(self, usevalues=False, **kwargs):
        """Cheap init.

        :Parameters:
            usevalues: Boolean
                If True the values of the attribute used for splitting will be
                used to determine odd and even samples. If False odd and even
                chunks are defined by the order of attribute values, i.e. first
                unique attribute is odd, second is even, despite the
                corresponding values might indicate the opposite (e.g. in case
                of [2,3].
        """
        Splitter.__init__(self, **(kwargs))

        self.__usevalues = usevalues


    def _getSplitConfig(self, uniqueattrs):
        """Huka chaka!
        """
        if self.__usevalues:
            return [uniqueattrs[(uniqueattrs % 2) == True],
                    uniqueattrs[(uniqueattrs % 2) == False]]
        else:
            return [uniqueattrs[N.arange(len(uniqueattrs)) %2 == True],
                    uniqueattrs[N.arange(len(uniqueattrs)) %2 == False]]


    def __str__(self):
        """String summary over the object
        """
        return \
          "OddEvenSplitter / " + Splitter.__str__(self)



class HalfSplitter(Splitter):
    """Split a dataset into two halves of the sample attribute.

    The splitter yields to splits: first (1st half, 2nd half) and second
    (2nd half, 1st half).
    """
    def __init__(self, **kwargs):
        """Cheap init.
        """
        Splitter.__init__(self, **(kwargs))


    def _getSplitConfig(self, uniqueattrs):
        """Huka chaka!
        """
        return [uniqueattrs[:len(uniqueattrs)/2],
                uniqueattrs[len(uniqueattrs)/2:]]


    def __str__(self):
        """String summary over the object
        """
        return \
          "HalfSplitter / " + Splitter.__str__(self)



class NFoldSplitter(Splitter):
    """Generic N-fold data splitter.

    XXX: This docstring is a shame for such an important class!
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


    def _getSplitConfig(self, uniqueattrs):
        """Returns proper split configuration for N-M fold split.
        """
        return support.getUniqueLengthNCombinations(uniqueattrs,
                                                    self.__cvtype)
