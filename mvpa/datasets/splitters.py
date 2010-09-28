# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Collection of dataset splitters.

Splitters are destined to split the provided dataset various ways to
simplify cross-validation analysis, implement boosting of the
estimates, or sample null-space via permutation testing.

Most of the splitters at the moment split 2-ways -- conventionally
first part is used for training, and 2nd part for testing by
`CrossValidatedTransferError` and `SplitClassifier`.
"""

__docformat__ = 'restructuredtext'

import operator

import numpy as np

import mvpa.misc.support as support
from mvpa.base.dochelpers import enhanced_doc_string, _str, _repr
from mvpa.datasets.miscfx import coarsen_chunks, random_samples, \
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
                 npertarget='all',
                 nrunspersplit=1,
                 count=None,
                 strategy='equidistant',
                 discard_boundary=None,
                 attr='chunks',
                 reverse=False,
                 noslicing=False):
        """Initialize splitter base.

        Parameters
        ----------
        npertarget : int or str (or list of them) or float
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
          selected for each run (see the `npertarget` argument).
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
        self.__npertarget = None
        self.__runspersplit = nrunspersplit
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
        self.set_n_per_label(npertarget)


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
                raise ValueError, "Unsupported value '%s' for npertarget." \
                      " Supported ones are %s or float or int" \
                      % (value, self._NPERLABEL_STR)
        self.__npertarget = value


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
        # for each split
        cfgs = self.splitcfg(dataset)
        n_cfgs = len(cfgs)

        # Finally split the data
        for isplit, split in enumerate(cfgs):

            # determine sample sizes
            if not operator.isSequenceType(self.__npertarget) \
                   or isinstance(self.__npertarget, str):
                npertargetsplit = [self.__npertarget] * len(split)
            else:
                npertargetsplit = self.__npertarget

            # get splitted datasets
            split_ds = self.split_dataset(dataset, split)

            # do multiple post-processing runs for this split
            for run in xrange(self.__runspersplit):

                # post-process all datasets
                finalized_datasets = []

                for ds, npertarget in zip(split_ds, npertargetsplit):
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

                    # select subset of samples if requested
                    if npertarget == 'all' or ds is None:
                        finalized_datasets.append(ds)
                    else:
                        # We need to select a subset of samples
                        # TODO: move all this logic within random_sample

                        # go for maximum possible number of samples provided
                        # by each label in this dataset
                        if npertarget == 'equal':
                            # determine the min number of samples per class
                            npl = np.array(get_nsamples_per_attr(
                                ds, 'targets').values()).min()
                        elif isinstance(npertarget, float) or (
                            operator.isSequenceType(npertarget) and
                            len(npertarget) > 0 and
                            isinstance(npertarget[0], float)):
                            # determine number of samples per class and take
                            # a ratio
                            counts = np.array(get_nsamples_per_attr(
                                ds, 'targets').values())
                            npl = (counts * npertarget).round().astype(int)
                        else:
                            npl = npertarget

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
                filter_ = np.array([ i in spec \
                                    for i in splitattr_data], dtype='bool')
                filters.append(filter_)
                if cum_filter is None:
                    cum_filter = filter_
                else:
                    cum_filter = np.logical_and(cum_filter, filter_)

        # need to turn possible Nones into proper ids sequences
        if none_specs > 1:
            raise ValueError, "Splitter cannot handle more than one `None` " \
                              "split definition."

        for i, filter_ in enumerate(filters):
            if filter_ is None:
                filters[i] = np.logical_not(cum_filter)

            # If it was told to discard samples on the boundary to the
            # other parts of the split
            if discard_boundary is not None:
                ndiscard = discard_boundary[i]
                if ndiscard != 0:
                    # XXX sloppy implementation for now. It still
                    # should not be the main reason for a slow-down of
                    # the whole analysis ;)
                    f, lenf = filters[i], len(filters[i])
                    f_pad = np.concatenate(([True]*ndiscard, f, [True]*ndiscard))
                    for d in xrange(2*ndiscard+1):
                        f = np.logical_and(f, f_pad[d:d+lenf])
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
            stepsizes = np.unique(idx[1:] - idx[:-1])
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


    def __repr__(self, *args, **kwargs):
        return _repr(self,
                     npertarget=self.__npertarget,
                     nrunspersplit=self.__runspersplit,
                     count=self.count,
                     strategy=self.__strategy,
                     discard_boundary=self.discard_boundary,
                     attr=self.__splitattr,
                     reverse=self._reverse,
                     noslicing=self.__noslicing,
                     *args,
                     **kwargs)


    def __str__(self):
        return _str(self)


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
                    indexes = np.random.permutation(range(n_cfgs))[:count]
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
    npertarget = property(fget=lambda self:self.__npertarget)



class NoneSplitter(Splitter):
    """Non-splitting Splitter for resampling purposes.

    This dataset splitter that does **not** split dataset, but it offers access
    to the full set of resampling techniques provided by the Splitter base
    class.
    """

    def __init__(self, **kwargs):
        Splitter.__init__(self, **(kwargs))


    __doc__ = enhanced_doc_string('NoneSplitter', locals(), Splitter)


    ##REF: Name was automagically refactored
    def _get_split_config(self, uniqueattrs):
        return [([], None)]
