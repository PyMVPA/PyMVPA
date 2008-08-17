#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Collection of dataset splitters."""

__docformat__ = 'restructuredtext'

import operator

import numpy as N

import mvpa.misc.support as support
from mvpa.base.dochelpers import enhancedDocString


class Splitter(object):
    """Base class of dataset splitters.

    Each splitter should be initialized with all its necessary parameters. The
    final splitting is done running the splitter object on a certain Dataset
    via __call__(). This method has to be implemented like a generator, i.e. it
    has to return every possible split with a yield() call.

    Each split has to be returned as a sequence of Datasets. The properties
    of the splitted dataset may vary between implementations. It is possible
    to declare a sequence element as 'None'. 

    Please note, that even if there is only one Dataset returned it has to be
    an element in a sequence and not just the Dataset object!
    """
    def __init__(self,
                 nperlabel='all',
                 nrunspersplit=1,
                 permute=False,
                 attr='chunks'):
        """Initialize splitter base.

        :Parameters:
          nperlabel : int or str (or list of them)
            Number of dataset samples per label to be included in each
            split. Two special strings are recognized: 'all' uses all available
            samples (default) and 'equal' uses the maximum number of samples
            the can be provided by all of the classes. This value might be
            provided as a sequence whos length matches the number of datasets
            per split and indicates the configuration for the respective dataset
            in each split.
          nrunspersplit: int
            Number of times samples for each split are chosen. This
            is mostly useful if a subset of the available samples
            is used in each split and the subset is randomly
            selected for each run (see the `nperlabel` argument).
          permute : bool
            If set to `True`, the labels of each generated dataset
            will be permuted on a per-chunk basis.
          attr : str
            Sample attribute used to determine splits.
        """
        # pylint happyness block
        self.__nperlabel = None
        self.__runspersplit = nrunspersplit
        self.__permute = permute
        self.__splitattr = attr

        # pattern sampling status vars
        self.setNPerLabel(nperlabel)


    def setNPerLabel(self, value):
        """Set the number of samples per label in the split datasets.

        'equal' sets sample size to highest possible number of samples that
        can be provided by each class. 'all' uses all available samples
        (default).
        """
        self.__nperlabel = value


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

        # local bindings to methods to gain some speedup
        ds_class = dataset.__class__
        DS_permuteLabels = ds_class.permuteLabels
        try:
            DS_getNSamplesPerLabel = ds_class._getNSamplesPerAttr
        except AttributeError:
            # Some "not-real" datasets e.g. MetaDataset, might not
            # have it
            pass
        DS_getRandomSamples = ds_class.getRandomSamples

        # for each split
        for split in self.splitcfg(dataset):

            # determine sample sizes
            if not operator.isSequenceType(self.__nperlabel) \
                   or isinstance(self.__nperlabel, str):
                nperlabel = [self.__nperlabel] * len(split)
            else:
                nperlabel = self.__nperlabel

            # get splitted datasets
            split_ds = self.splitDataset(dataset, split)

            # do multiple post-processing runs for this split
            for run in xrange(self.__runspersplit):

                # post-process all datasets
                finalized_datasets = []

                for i, ds in enumerate(split_ds):
                    # permute the labels
                    if self.__permute:
                        DS_permuteLabels(ds, True, perchunk=True)

                    # select subset of samples if requested
                    if nperlabel[i] == 'all':
                        finalized_datasets.append(ds)
                    else:
                        # just pass through if no real dataset
                        if ds == None:
                            finalized_datasets.append(None)
                        else:
                            # go for maximum possible number of samples provided
                            # by each label in this dataset
                            if nperlabel[i] == 'equal':
                                # determine number number of samples per class
                                npl = N.array(DS_getNSamplesPerLabel(
                                    ds, attrib='labels').values()).min()
                            else:
                                npl = nperlabel[i]

                            # finally select the patterns
                            finalized_datasets.append(
                                DS_getRandomSamples(ds, npl))

                yield finalized_datasets


    def splitDataset(self, dataset, specs):
        """Split a dataset by separating the samples where the configured
        sample attribute matches an element of `specs`.

        :Parameters:
          dataset : Dataset
            This is this source dataset.
          specs : sequence of sequences
            Contains ids of a sample attribute that shall be split into the
            another dataset.

        :Returns: Tuple of splitted datasets.
        """
        # collect the sample ids for each resulting dataset
        filters = []
        none_specs = 0
        cum_filter = None

        splitattr_data = eval('dataset.' + self.__splitattr)
        for spec in specs:
            if spec == None:
                filters.append(None)
                none_specs += 1
            else:
                filter_ = N.array([ i in spec \
                                    for i in splitattr_data])
                filters.append(filter_)
                if cum_filter == None:
                    cum_filter = filter_
                else:
                    cum_filter = N.logical_and(cum_filter, filter_)

        # need to turn possible Nones into proper ids sequences
        if none_specs > 1:
            raise ValueError, "Splitter cannot handle more than one `None` " \
                              "split definition."

        for i, filter_ in enumerate(filters):
            if filter_ == None:
                filters[i] = N.logical_not(cum_filter)

        # split data: return None if no samples are left
        # XXX: Maybe it should simply return an empty dataset instead, but
        #      keeping it this way for now, to maintain current behavior
        split_datasets = []

        # local bindings
        dataset_selectSamples = dataset.selectSamples
        for filter_ in filters:
            if (filter_ == False).all():
                split_datasets.append(None)
            else:
                split_datasets.append(dataset_selectSamples(filter_))

        return split_datasets


    def __str__(self):
        """String summary over the object
        """
        return \
          "SplitterConfig: nperlabel:%s runs-per-split:%d permute:%s" \
          % (self.__nperlabel, self.__runspersplit, self.__permute)


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


    __doc__ = enhancedDocString('NoneSplitter', locals(), Splitter)


    def _getSplitConfig(self, uniqueattrs):
        """Return just one full split: no first or second dataset.
        """
        if self.__mode == 'second':
            return [([], None)]
        else:
            return [(None, [])]


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


    __doc__ = enhancedDocString('OddEvenSplitter', locals(), Splitter)


    def _getSplitConfig(self, uniqueattrs):
        """Huka chaka!
        """
        if self.__usevalues:
            return [(None, uniqueattrs[(uniqueattrs % 2) == True]),
                    (None, uniqueattrs[(uniqueattrs % 2) == False])]
        else:
            return [(None, uniqueattrs[N.arange(len(uniqueattrs)) %2 == True]),
                    (None, uniqueattrs[N.arange(len(uniqueattrs)) %2 == False])]


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


    __doc__ = enhancedDocString('HalfSplitter', locals(), Splitter)


    def _getSplitConfig(self, uniqueattrs):
        """Huka chaka!
        """
        return [(None, uniqueattrs[:len(uniqueattrs)/2]),
                (None, uniqueattrs[len(uniqueattrs)/2:])]


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
            Additional parameters are passed to the `Splitter` base class.
        """
        Splitter.__init__(self, **(kwargs))

        # pylint happyness block
        self.__cvtype = cvtype


    __doc__ = enhancedDocString('NFoldSplitter', locals(), Splitter)


    def __str__(self):
        """String summary over the object
        """
        return \
          "N-%d-FoldSplitter / " % self.__cvtype + Splitter.__str__(self)


    def _getSplitConfig(self, uniqueattrs):
        """Returns proper split configuration for N-M fold split.
        """
        return [(None, i) for i in \
                    support.getUniqueLengthNCombinations(uniqueattrs,
                                                         self.__cvtype)]



class CustomSplitter(Splitter):
    """Split a dataset using an arbitrary custom rule.

    The splitter is configured by passing a custom spitting rule (`splitrule`)
    to its constructor. Such a rule is basically a sequence of split
    definitions. Every single element in this sequence results in excatly one
    split generated by the Splitter. Each element is another sequence for
    sequences of sample ids for each dataset that shall be generated in the
    split.

    Example:

      * Generate two splits. In the first split the *second* dataset
        contains all samples with sample attributes corresponding to
        either 0, 1 or 2. The *first* dataset of the first split contains
        all samples which are not split into the second dataset.

        The second split yields three datasets. The first with all samples
        corresponding to sample attributes 1 and 2, the second dataset
        contains only samples with attrbiute 3 and the last dataset
        contains the samples with attribute 5 and 6.

        CustomSplitter([(None, [0, 1, 2]), ([1,2], [3], [5, 6])])
    """
    def __init__(self, splitrule, **kwargs):
        """Cheap init.
        """
        Splitter.__init__(self, **(kwargs))

        self.__splitrule = splitrule


    __doc__ = enhancedDocString('CustomSplitter', locals(), Splitter)


    def _getSplitConfig(self, uniqueattrs):
        """Huka chaka!
        """
        return self.__splitrule


    def __str__(self):
        """String summary over the object
        """
        return "CustomSplitter / " + Splitter.__str__(self)
