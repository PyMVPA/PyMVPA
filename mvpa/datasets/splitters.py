# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Collection of dataset splitters.

Module Description
==================

Splitters are destined to split the provided dataset various ways to
simplify cross-validation analysis, implement boosting of the
estimates, or sample null-space via permutation testing.

Most of the splitters at the moment split 2-ways -- conventionally
first part is used for training, and 2nd part for testing by
`CrossValidatedTransferError` and `SplitClassifier`.

Brief Description of Available Splitters
========================================

* `NoneSplitter` - just return full dataset as the desired part (training/testing)
* `OddEvenSplitter` - 2 splits: (odd samples,even samples) and (even, odd)
* `HalfSplitter` - 2 splits: (first half, second half) and (second, first)
* `NFoldSplitter` - splits for N-Fold cross validation.

"""

__docformat__ = 'restructuredtext'

import operator

import numpy as N

import mvpa.misc.support as support
from mvpa.base.dochelpers import enhanced_doc_string
from mvpa.datasets.miscfx import coarsen_chunks, permute_targets, random_samples, \
                                 get_nsamples_per_attr

if __debug__:
    from mvpa.base import debug

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

    _STRATEGIES = ('first', 'random', 'equidistant')
    _NPERLABEL_STR = ['equal', 'all']

    def __init__(self,
                 nperlabel='all',
                 nrunspersplit=1,
                 permute=False,
                 count=None,
                 strategy='equidistant',
                 discard_boundary=None,
                 attr='chunks',
                 reverse=False,
                 noslicing=False):
        """Initialize splitter base.

        Parameters
        ----------
        nperlabel : int or str (or list of them) or float
          Number of dataset samples per label to be included in each
          split. If given as a float, it must be in [0,1] range and would
          mean the ratio of selected samples per each label.
          Two special strings are recognized: 'all' uses all available
          samples (default) and 'equal' uses the maximum number of samples
          the can be provided by all of the classes. This value might be
          provided as a sequence length of which matches the number of datasets
          per split and indicates the configuration for the respective dataset
          in each split.
        nrunspersplit : int
          Number of times samples for each split are chosen. This
          is mostly useful if a subset of the available samples
          is used in each split and the subset is randomly
          selected for each run (see the `nperlabel` argument).
        permute : bool
          If set to `True`, the labels of each generated dataset
          will be permuted on a per-chunk basis.
        count : None or int
          Desired number of splits to be output. It is limited by the
          number of splits possible for a given splitter
          (e.g. `OddEvenSplitter` can have only up to 2 splits). If None,
          all splits are output (default).
        strategy : str
          If `count` is not None, possible strategies are possible:
          'first': First `count` splits are chosen;
          'random': Random (without replacement) `count` splits are chosen;
          'equidistant': Splits which are equidistant from each other.
        discard_boundary : None or int or sequence of int
          If not `None`, how many samples on the boundaries between
          parts of the split to discard in the training part.
          If int, then discarded in all parts.  If a sequence, numbers
          to discard are given per part of the split.
          E.g. if splitter splits only into (training, testing)
          parts, then `discard_boundary=(2,0)` would instruct to discard
          2 samples from training which are on the boundary with testing.
        attr : str
          Sample attribute used to determine splits.
        reverse : bool
          If True, the order of datasets in the split is reversed, e.g.
          instead of (training, testing), (training, testing) will be spit
          out
        noslicing : bool
          If True, dataset splitting is not done by slicing (causing
          shared data between source and split datasets) even if it would
          be possible. By default slicing is performed whenever possible
          to reduce the memory footprint.
        """
        # pylint happyness block
        self.__nperlabel = None
        self.__runspersplit = nrunspersplit
        self.__permute = permute
        self.__splitattr = attr
        self.__noslicing = noslicing
        self._reverse = reverse
        self.discard_boundary = discard_boundary

        # we don't check it, thus no reason to make it private.
        # someone might find it useful to change post creation
        # TODO utilize such (or similar) policy through out the code
        self.count = count
        """Number (max) of splits to output on call"""

        self._set_strategy(strategy)

        # pattern sampling status vars
        self.set_n_per_label(nperlabel)


    __doc__ = enhanced_doc_string('Splitter', locals())

    ##REF: Name was automagically refactored
    def _set_strategy(self, strategy):
        """Set strategy to select splits out from available
        """
        strategy = strategy.lower()
        if not strategy in self._STRATEGIES:
            raise ValueError, "strategy is not known. Known are %s" \
                  % str(self._STRATEGIES)
        self.__strategy = strategy

    ##REF: Name was automagically refactored
    def set_n_per_label(self, value):
        """Set the number of samples per label in the split datasets.

        'equal' sets sample size to highest possible number of samples that
        can be provided by each class. 'all' uses all available samples
        (default).
        """
        if isinstance(value, basestring):
            if not value in self._NPERLABEL_STR:
                raise ValueError, "Unsupported value '%s' for nperlabel." \
                      " Supported ones are %s or float or int" \
                      % (value, self._NPERLABEL_STR)
        self.__nperlabel = value


    ##REF: Name was automagically refactored
    def _get_split_config(self, uniqueattr):
        """Return list with samples of 2nd dataset in a split.

        Each subclass has to implement this method. It gets a sequence with
        the unique attribute ids of a dataset and has to return a list of lists
        containing sample ids to split into the second dataset.
        """
        raise NotImplementedError


    def __call__(self, dataset):
        """Splits the dataset.

        This method behaves like a generator.
        """

        # local bindings to methods to gain some speedup
        ds_class = dataset.__class__

        # for each split
        cfgs = self.splitcfg(dataset)
        n_cfgs = len(cfgs)

        # Finally split the data
        for isplit, split in enumerate(cfgs):

            # determine sample sizes
            if not operator.isSequenceType(self.__nperlabel) \
                   or isinstance(self.__nperlabel, str):
                nperlabelsplit = [self.__nperlabel] * len(split)
            else:
                nperlabelsplit = self.__nperlabel

            # get splitted datasets
            split_ds = self.split_dataset(dataset, split)

            # do multiple post-processing runs for this split
            for run in xrange(self.__runspersplit):

                # post-process all datasets
                finalized_datasets = []

                for ds, nperlabel in zip(split_ds, nperlabelsplit):
                    # Set flag of dataset either this was the last split
                    # ??? per our discussion this might be the best
                    #     solution which would scale if we care about
                    #     thread-safety etc
                    if ds is not None:
                        ds_a = ds.a
                        lastsplit = (isplit == n_cfgs-1)
                        if not ds_a.has_key('lastsplit'):
                            # if not yet known -- add one
                            ds_a['lastsplit'] = lastsplit
                        else:
                            # otherwise just assign a new value
                            ds_a.lastsplit = lastsplit
                    # permute the labels
                    if self.__permute:
                        permute_targets(ds, chunks_attr='chunks')

                    # select subset of samples if requested
                    if nperlabel == 'all' or ds is None:
                        finalized_datasets.append(ds)
                    else:
                        # We need to select a subset of samples
                        # TODO: move all this logic within random_sample

                        # go for maximum possible number of samples provided
                        # by each label in this dataset
                        if nperlabel == 'equal':
                            # determine the min number of samples per class
                            npl = N.array(get_nsamples_per_attr(
                                ds, 'targets').values()).min()
                        elif isinstance(nperlabel, float) or (
                            operator.isSequenceType(nperlabel) and
                            len(nperlabel) > 0 and
                            isinstance(nperlabel[0], float)):
                            # determine number of samples per class and take
                            # a ratio
                            counts = N.array(get_nsamples_per_attr(
                                ds, 'targets').values())
                            npl = (counts * nperlabel).round().astype(int)
                        else:
                            npl = nperlabel

                        # finally select the patterns
                        finalized_datasets.append(
                            random_samples(ds, npl))

                if self._reverse:
                    yield finalized_datasets[::-1]
                else:
                    yield finalized_datasets


    ##REF: Name was automagically refactored
    def split_dataset(self, dataset, specs):
        """Split a dataset by separating the samples where the configured
        sample attribute matches an element of `specs`.

        Parameters
        ----------
        dataset : Dataset
          This is this source dataset.
        specs : sequence of sequences
          Contains ids of a sample attribute that shall be split into the
          another dataset.

        Returns
        -------
        Tuple of splitted datasets.
        """
        # collect the sample ids for each resulting dataset
        filters = []
        none_specs = 0
        cum_filter = None

        # Prepare discard_boundary
        discard_boundary = self.discard_boundary
        if isinstance(discard_boundary, int):
            if discard_boundary != 0:
                discard_boundary = (discard_boundary,) * len(specs)
            else:
                discard_boundary = None

        splitattr_data = dataset.sa[self.__splitattr].value
        for spec in specs:
            if spec is None:
                filters.append(None)
                none_specs += 1
            else:
                filter_ = N.array([ i in spec \
                                    for i in splitattr_data], dtype='bool')
                filters.append(filter_)
                if cum_filter is None:
                    cum_filter = filter_
                else:
                    cum_filter = N.logical_and(cum_filter, filter_)

        # need to turn possible Nones into proper ids sequences
        if none_specs > 1:
            raise ValueError, "Splitter cannot handle more than one `None` " \
                              "split definition."

        for i, filter_ in enumerate(filters):
            if filter_ is None:
                filters[i] = N.logical_not(cum_filter)

            # If it was told to discard samples on the boundary to the
            # other parts of the split
            if discard_boundary is not None:
                ndiscard = discard_boundary[i]
                if ndiscard != 0:
                    # XXX sloppy implementation for now. It still
                    # should not be the main reason for a slow-down of
                    # the whole analysis ;)
                    f, lenf = filters[i], len(filters[i])
                    f_pad = N.concatenate(([True]*ndiscard, f, [True]*ndiscard))
                    for d in xrange(2*ndiscard+1):
                        f = N.logical_and(f, f_pad[d:d+lenf])
                    filters[i] = f[:]

        # split data: return None if no samples are left
        # XXX: Maybe it should simply return an empty dataset instead, but
        #      keeping it this way for now, to maintain current behavior
        split_datasets = []


        for filter_ in filters:
            if (filter_ == False).all():
                split_datasets.append(None)
            else:
                # check whether we can do slicing instead of advanced
                # indexing -- if we can split the dataset without causing
                # the data to be copied, its is quicker and leaner.
                # However, it only works if we have a contiguous chunk or
                # regular step sizes for the samples to be split
                split_datasets.append(dataset[self._filter2slice(filter_)])

        return split_datasets


    def _filter2slice(self, bf):
        if self.__noslicing:
            # we are not allowed to help :-(
            return bf
        # the filter should be a boolean array
        if not len(bf):
            raise ValueError("'%s' recieved an empty filter. This is a "
                             "bug." % self.__class__.__name__)
        # get indices of non-zero filter elements
        idx = bf.nonzero()[0]
        idx_start = idx[0]
        idx_end = idx[-1] + 1
        idx_step = None
        if len(idx) > 1:
            # we need to figure out if there is a regular step-size
            # between elements
            stepsizes = N.unique(idx[1:] - idx[:-1])
            if len(stepsizes) > 1:
                # multiple step-sizes -> slicing is not possible -> return
                # orginal filter
                return bf
            else:
                idx_step = stepsizes[0]

        sl = slice(idx_start, idx_end, idx_step)
        if __debug__:
            debug("SPL", "Splitting by basic slicing is possible and permitted "
                         "(%s)." % sl)
        return sl


    def __str__(self):
        """String summary over the object
        """
        return \
          "SplitterConfig: nperlabel:%s runs-per-split:%d permute:%s" \
          % (self.__nperlabel, self.__runspersplit, self.__permute)


    def splitcfg(self, dataset):
        """Return splitcfg for a given dataset"""
        cfgs = self._get_split_config(dataset.sa[self.__splitattr].unique)

        # Select just some splits if desired
        count, n_cfgs = self.count, len(cfgs)

        # further makes sense only iff count < n_cfgs,
        # otherwise all strategies are equivalent
        if count is not None and count < n_cfgs:
            if count < 1:
                # we can only wish a good luck
                return []
            strategy = self.strategy
            if strategy == 'first':
                cfgs = cfgs[:count]
            elif strategy in ['equidistant', 'random']:
                if strategy == 'equidistant':
                    # figure out what step is needed to
                    # accommodate the `count` number
                    step = float(n_cfgs) / count
                    assert(step >= 1.0)
                    indexes = [int(round(step * i)) for i in xrange(count)]
                elif strategy == 'random':
                    indexes = N.random.permutation(range(n_cfgs))[:count]
                    # doesn't matter much but lets keep them in the original
                    # order at least
                    indexes.sort()
                else:
                    # who said that I am paranoid?
                    raise RuntimeError, "Really should not happen"
                if __debug__:
                    debug("SPL", "For %s strategy selected %s splits "
                          "from %d total" % (strategy, indexes, n_cfgs))
                cfgs = [cfgs[i] for i in indexes]

        return cfgs


    strategy = property(fget=lambda self:self.__strategy,
                        fset=_set_strategy)
    splitattr = property(fget=lambda self:self.__splitattr)



class NoneSplitter(Splitter):
    """This is a dataset splitter that does **not** split. It simply returns
    the full dataset that it is called with.

    The passed dataset is returned as the second element of the 2-tuple.
    The first element of that tuple will always be 'None'.
    """

    _known_modes = ['first', 'second']

    def __init__(self, mode='second', **kwargs):
        """
        Parameters
        ----------
        mode
          Either 'first' or 'second' (default) -- which output dataset
          would actually contain the samples
        """
        Splitter.__init__(self, **(kwargs))

        if not mode in NoneSplitter._known_modes:
            raise ValueError, "Unknown mode %s for NoneSplitter" % mode
        self.__mode = mode


    __doc__ = enhanced_doc_string('NoneSplitter', locals(), Splitter)


    ##REF: Name was automagically refactored
    def _get_split_config(self, uniqueattrs):
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
        """
        Parameters
        ----------
        usevalues : bool
          If True the values of the attribute used for splitting will be
          used to determine odd and even samples. If False odd and even
          chunks are defined by the order of attribute values, i.e. first
          unique attribute is odd, second is even, despite the
          corresponding values might indicate the opposite (e.g. in case
          of [2,3].
        """
        Splitter.__init__(self, **(kwargs))

        self.__usevalues = usevalues


    __doc__ = enhanced_doc_string('OddEvenSplitter', locals(), Splitter)


    ##REF: Name was automagically refactored
    def _get_split_config(self, uniqueattrs):
        """
        Returns
        -------
        list of tuples (None, list of int)
          2 items: odd samples into 1st split
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
        Splitter.__init__(self, **(kwargs))


    __doc__ = enhanced_doc_string('HalfSplitter', locals(), Splitter)


    ##REF: Name was automagically refactored
    def _get_split_config(self, uniqueattrs):
        """
        Returns
        -------
        list of tuples (None, list of int)
          2 items: first half of samples into 1st split
        """
        return [(None, uniqueattrs[:len(uniqueattrs)/2]),
                (None, uniqueattrs[len(uniqueattrs)/2:])]


    def __str__(self):
        """String summary over the object
        """
        return \
          "HalfSplitter / " + Splitter.__str__(self)



class NGroupSplitter(Splitter):
    """Split a dataset into N-groups of the sample attribute.

    For example, NGroupSplitter(2) is the same as the HalfSplitter and
    yields to splits: first (1st half, 2nd half) and second (2nd half,
    1st half).
    """
    def __init__(self, ngroups=4, **kwargs):
        """Initialize the N-group splitter.

        Parameters
        ----------
        ngroups : int
          Number of groups to split the attribute into.
        **kwargs
          Additional parameters are passed to the `Splitter` base class.
        """
        Splitter.__init__(self, **(kwargs))

        self.__ngroups = ngroups

    __doc__ = enhanced_doc_string('NGroupSplitter', locals(), Splitter)


    ##REF: Name was automagically refactored
    def _get_split_config(self, uniqueattrs):
        """
        Returns
        -------
        list of tuples (None, list of int)
          Indices for splitting
        """

        # make sure there are more of attributes than desired groups
        if len(uniqueattrs) < self.__ngroups:
            raise ValueError, "Number of groups (%d) " % (self.__ngroups) + \
                  "must be less than " + \
                  "or equal to the number of unique attributes (%d)" % \
                  (len(uniqueattrs))

        # use coarsen_chunks to get the split indices
        split_ind = coarsen_chunks(uniqueattrs, nchunks=self.__ngroups)
        split_ind = N.asarray(split_ind)

        # loop and create splits
        split_list = [(None, uniqueattrs[split_ind==i])
                       for i in range(self.__ngroups)]
        return split_list


    def __str__(self):
        """String summary over the object
        """
        return \
          "N-%d-GroupSplitter / " % self.__ngroup + Splitter.__str__(self)



class NFoldSplitter(Splitter):
    """Generic N-fold data splitter.

    Provide folding splitting. Given a dataset with N chunks, with
    cvtype=1 (which is default), it would generate N splits, where
    each chunk sequentially is taken out (with replacement) for
    cross-validation.  Example, if there is 4 chunks, splits for
    cvtype=1 are::

        [[1, 2, 3], [0]]
        [[0, 2, 3], [1]]
        [[0, 1, 3], [2]]
        [[0, 1, 2], [3]]

    If cvtype>1, then all possible combinations of cvtype number of
    chunks are taken out for testing, so for cvtype=2 in previous
    example::

        [[2, 3], [0, 1]]
        [[1, 3], [0, 2]]
        [[1, 2], [0, 3]]
        [[0, 3], [1, 2]]
        [[0, 2], [1, 3]]
        [[0, 1], [2, 3]]

    """

    def __init__(self,
                 cvtype = 1,
                 **kwargs):
        """Initialize the N-fold splitter.

        Parameters
        ----------
        cvtype : int
          Type of cross-validation: N-(cvtype)
        **kwargs
          Additional parameters are passed to the `Splitter` base class.
        """
        Splitter.__init__(self, **(kwargs))

        # pylint happiness block
        self.__cvtype = cvtype


    __doc__ = enhanced_doc_string('NFoldSplitter', locals(), Splitter)


    def __str__(self):
        """String summary over the object
        """
        return \
          "N-%d-FoldSplitter / " % self.__cvtype + Splitter.__str__(self)


    ##REF: Name was automagically refactored
    def _get_split_config(self, uniqueattrs):
        """Returns proper split configuration for N-M fold split.
        """
        return [(None, i) for i in \
                    support.get_unique_length_n_combinations(uniqueattrs,
                                                         self.__cvtype)]



class CustomSplitter(Splitter):
    """Split a dataset using an arbitrary custom rule.

    The splitter is configured by passing a custom spitting rule (`splitrule`)
    to its constructor. Such a rule is basically a sequence of split
    definitions. Every single element in this sequence results in excatly one
    split generated by the Splitter. Each element is another sequence for
    sequences of sample ids for each dataset that shall be generated in the
    split.

    Examples
    --------
    Generate two splits. In the first split the *second* dataset
    contains all samples with sample attributes corresponding to
    either 0, 1 or 2. The *first* dataset of the first split contains
    all samples which are not split into the second dataset.

    The second split yields three datasets. The first with all samples
    corresponding to sample attributes 1 and 2, the second dataset
    contains only samples with attrbiute 3 and the last dataset
    contains the samples with attribute 5 and 6.

    >>> splitter = CustomSplitter([(None, [0, 1, 2]), ([1,2], [3], [5, 6])])
    """
    def __init__(self, splitrule, **kwargs):
        """
        Parameters
        ----------
        splitrule : list of tuple
          Custom splits to use
        """
        Splitter.__init__(self, **(kwargs))

        self.__splitrule = splitrule


    __doc__ = enhanced_doc_string('CustomSplitter', locals(), Splitter)


    ##REF: Name was automagically refactored
    def _get_split_config(self, uniqueattrs):
        """
        Returns
        -------
        whatever was provided in splitrule argument
        """
        return self.__splitrule


    def __str__(self):
        """String summary over the object
        """
        return "CustomSplitter / " + Splitter.__str__(self)
