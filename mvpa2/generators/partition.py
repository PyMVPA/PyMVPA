# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""""""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base.node import Node
from mvpa2.datasets.miscfx import coarsen_chunks
import mvpa2.misc.support as support

if __debug__:
    from mvpa2.base import debug


class Partitioner(Node):
    """Generator node to partition a dataset.

    Partitioning is done by adding a sample attribute that assigns samples to an
    arbitrary number of partitions. Subclasses offer a variety of partitioning
    technique that are useful in e.g. cross-validation procedures.

    it is important to note that other than adding a new sample attribute input
    datasets are not modified. In particular, there is no splitting of datasets
    into multiple pieces. If this is desired, a Partitioner can be chained to a
    `Splitter` node to achieve this.
    """

    _STRATEGIES = ('first', 'random', 'equidistant')
    _NPERLABEL_STR = ['equal', 'all']

    def __init__(self,
                 count=None,
                 selection_strategy='equidistant',
                 attr='chunks',
                 space='partitions',
                 **kwargs):
        """
        Parameters
        ----------
        count : None or int
          Desired number of splits to be output. It is limited by the
          number of splits possible for a given splitter
          (e.g. `OddEvenSplitter` can have only up to 2 splits). If None,
          all splits are output (default).
        selection_strategy : str
          If `count` is not None, possible strategies are possible:
          'first': First `count` splits are chosen;
          'random': Random (without replacement) `count` splits are chosen;
          'equidistant': Splits which are equidistant from each other.
        attr : str
          Sample attribute used to determine splits.
        space : str
          Name of the to be created sample attribute defining the partitions.
          In addition, a dataset attribute named '``space``\_set' will be added
          to each output dataset, indicating the number of the partition set
          it corresponds to.
        """
        Node.__init__(self, space=space, **kwargs)
        # pylint happyness block
        self.__splitattr = attr
        # we don't check it, thus no reason to make it private.
        # someone might find it useful to change post creation
        # TODO utilize such (or similar) policy through out the code
        self.count = count
        self._set_selection_strategy(selection_strategy)


    def _set_selection_strategy(self, strategy):
        """Set strategy to select splits out from available
        """
        strategy = strategy.lower()
        if not strategy in self._STRATEGIES:
            raise ValueError, "selection_strategy is not known. Known are %s" \
                  % str(self._STRATEGIES)
        self.__selection_strategy = strategy


    def _get_partition_specs(self, uniqueattr):
        """Return list with samples of 2nd dataset in a split.

        Each subclass has to implement this method. It gets a sequence with
        the unique attribute ids of a dataset and has to return a list of lists
        containing sample ids to split into the second dataset.
        """
        raise NotImplementedError


    def generate(self, ds):
        # for each split
        cfgs = self.get_partition_specs(ds)
        n_cfgs = len(cfgs)

        for iparts, parts in enumerate(cfgs):
            # give attribute array defining the current partition set
            pattr = self.get_partitions_attr(ds, parts)
            # shallow copy of the dataset
            pds = ds.copy(deep=False)
            pds.sa[self.get_space()] = pattr
            pds.a[self.get_space() + "_set"] = iparts
            pds.a['lastpartitionset'] = iparts == (n_cfgs - 1)
            yield pds


    def get_partitions_attr(self, ds, specs):
        """Create a partition attribute array for a particular partion spec.

        Parameters
        ----------
        ds : Dataset
          This is this source dataset.
        specs : sequence of sequences
          Contains ids of a sample attribute that shall go into each partition.

        Returns
        -------
        array(ints)
          Each partition is represented by a unique integer value.
        """
        # collect the sample ids for each resulting dataset
        filters = []
        none_specs = 0
        cum_filter = None

        splitattr_data = ds.sa[self.__splitattr].value
        # for each partition in this set
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
            raise ValueError("'%s' cannot handle more than one `None` " \
                              "partition spec." % self.__class__.__name__)

        # go with ints for simplicity. By default the attr is zeros, and the
        # first configured partition starts with one.
        part_attr = np.zeros(len(ds), dtype='int')
        for i, filter_ in enumerate(filters):
            # turn the one 'all the rest' filter into a slicing arg
            if filter_ is None:
                filter_ = np.logical_not(cum_filter)
            # now filter is guaranteed to be a slicing argument that can be used
            # to assign the attribute values
            part_attr[filter_] = i + 1
        return part_attr


    def get_partition_specs(self, ds):
        """Returns the specs for all to be generated partition sets.

        Returns
        -------
        list(lists)
        """
        # list (#splits) of lists (#partitions)
        cfgs = self._get_partition_specs(ds.sa[self.__splitattr].unique)

        # Select just some splits if desired
        count, n_cfgs = self.count, len(cfgs)

        # further makes sense only if count < n_cfgs,
        # otherwise all strategies are equivalent
        if count is not None and count < n_cfgs:
            if count < 1:
                # we can only wish a good luck
                return []
            strategy = self.selection_strategy
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
                    debug("SPL", "For %s selection strategy selected %s "
                          "partition specs from %d total"
                          % (strategy, indexes, n_cfgs))
                cfgs = [cfgs[i] for i in indexes]

        return cfgs


    selection_strategy = property(fget=lambda self:self.__selection_strategy,
                        fset=_set_selection_strategy)
    splitattr = property(fget=lambda self:self.__splitattr)



class OddEvenPartitioner(Partitioner):
    """Create odd and even partitions based on a sample attribute.

    The partitioner yields two datasets. In the first set all odd chunks are
    labeled '1' and all even runs are labeled '2'. In the second set the
    assignment is reversed (odd: '2', even: '1').
    """
    def __init__(self, usevalues=False, **kwargs):
        """
        Parameters
        ----------
        usevalues : bool
          If True the values of the attribute used for partitioning will be
          used to determine odd and even samples. If False odd and even
          chunks are defined by the order of attribute values, i.e. first
          unique attribute is odd, second is even, despite the
          corresponding values might indicate the opposite (e.g. in case
          of [2,3].
        """
        Partitioner.__init__(self, **(kwargs))
        self.__usevalues = usevalues


    def _get_partition_specs(self, uniqueattrs):
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
            return [(None, uniqueattrs[np.arange(len(uniqueattrs)) %2 == True]),
                    (None, uniqueattrs[np.arange(len(uniqueattrs)) %2 == False])]



class HalfPartitioner(Partitioner):
    """Partition a dataset into two halves of the sample attribute.

    The partitioner yields two datasets. In the first set second half of
    chunks are labeled '1' and the first half labeled '2'. In the second set the
    assignment is reversed (1st half: '1', 2nd half: '2').
    """
    def _get_partition_specs(self, uniqueattrs):
        """
        Returns
        -------
        list of tuples (None, list of int)
          2 items: first half of samples into 1st split
        """
        return [(None, uniqueattrs[:len(uniqueattrs)/2]),
                (None, uniqueattrs[len(uniqueattrs)/2:])]



class NGroupPartitioner(Partitioner):
    """Partition a dataset into N-groups of the sample attribute.

    For example, NGroupPartitioner(2) is the same as the HalfPartitioner and
    yields exactly the same partitions and labeling patterns.
    """
    def __init__(self, ngroups=4, **kwargs):
        """
        Parameters
        ----------
        ngroups : int
          Number of groups to split the attribute into.
        """
        Partitioner.__init__(self, **(kwargs))
        self.__ngroups = ngroups


    def _get_partition_specs(self, uniqueattrs):
        """
        Returns
        -------
        list of tuples (None, list of int)
          Indices for splitting
        """

        # make sure there are more of attributes than desired groups
        if len(uniqueattrs) < self.__ngroups:
            raise ValueError("Number of groups (%d) " % (self.__ngroups) + \
                  "must be less than " + \
                  "or equal to the number of unique attributes (%d)" % \
                  (len(uniqueattrs)))

        # use coarsen_chunks to get the split indices
        split_ind = coarsen_chunks(uniqueattrs, nchunks=self.__ngroups)
        split_ind = np.asarray(split_ind)

        # loop and create splits
        split_list = [(None, uniqueattrs[split_ind==i])
                       for i in range(self.__ngroups)]
        return split_list



class CustomPartitioner(Partitioner):
    """Partition a dataset using an arbitrary custom rule.

    The partitioner is configured by passing a custom rule (``splitrule``) to its
    constructor. Such a rule is basically a sequence of partition definitions.
    Every single element in this sequence results in exactly one partition set.
    Each element is another sequence of attribute values whose corresponding
    samples shall go into a particular partition.

    Examples
    --------
    Generate two sets. In the first set the *second* partition
    contains all samples with sample attributes corresponding to
    either 0, 1 or 2. The *first* partition of the first set contains
    all samples which are not part of the second partition.

    The second set yields three partitions. The first with all samples
    corresponding to sample attributes 1 and 2, the second contains only
    samples with attribute 3 and the last contains the samples with attribute 5
    and 6.

    >>> ptr = CustomPartitioner([(None, [0, 1, 2]), ([1,2], [3], [5, 6])])

    The numeric labels of all partitions correspond to their position in the
    ``splitrule`` of a particular set. Note that the actual labels start with
    '1' as all unselected elements are labeled '0'.
    """
    def __init__(self, splitrule, **kwargs):
        """
        Parameters
        ----------
        splitrule : list of tuple
          Custom partition set specs.
        """
        Partitioner.__init__(self, **(kwargs))
        self.__splitrule = splitrule


    def _get_partition_specs(self, uniqueattrs):
        """
        Returns
        -------
        whatever was provided in splitrule argument
        """
        return self.__splitrule



class NFoldPartitioner(Partitioner):
    """Generic N-fold data partitioner.

    Given a dataset with N chunks, with ``cvtype`` = 1 (which is default), it
    would generate N partition sets, where each chunk is sequentially taken out
    (with replacement) to form a second partition, while all other samples
    together form the first partition.  Example, if there are 4 chunks, partition
    sets for ``cvtype`` = 1 are::

        [[1, 2, 3], [0]]
        [[0, 2, 3], [1]]
        [[0, 1, 3], [2]]
        [[0, 1, 2], [3]]

    If ``cvtype``>1, then all possible combinations of ``cvtype`` number of
    chunks are taken out, so for ``cvtype`` = 2 in previous example yields::

        [[2, 3], [0, 1]]
        [[1, 3], [0, 2]]
        [[1, 2], [0, 3]]
        [[0, 3], [1, 2]]
        [[0, 2], [1, 3]]
        [[0, 1], [2, 3]]

    Note that the "taken-out" partition is always labeled '2' while the
    remaining elements are labeled '1'.
    """
    def __init__(self, cvtype = 1, **kwargs):
        """
        Parameters
        ----------
        cvtype : int
          Type of leave-one-out scheme: N-(cvtype)
        """
        Partitioner.__init__(self, **kwargs)
        self.__cvtype = cvtype


    def _get_partition_specs(self, uniqueattrs):
        return [(None, i) for i in \
                 support.xunique_combinations(uniqueattrs, self.__cvtype)]
