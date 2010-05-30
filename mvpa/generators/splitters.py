# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Generator nodes to split dataset into multiple parts.
"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa.base.node import Node
from mvpa.base import warning
from mvpa.misc.support import mask2slice


class Splitter(Node):
    """Generator node for dataset splitting.

    The splitter is configured with the name of an attribute. When its
    ``generate()`` methods is called with a dataset, it subsequently yields
    all possible subsets of this dataset, by selecting all dataset
    samples/features corresponding to a particular attribute value, for all
    unique attribute values.

    Dataset splitting is possible by sample attribute, or by feature attribute.
    The maximum number of splits can be limited, and custom attribute values
    may be provided.
    """
    def __init__(self, attr, attr_values=None, count=None, noslicing=False,
                 collection=None, reverse=False, **kwargs):
        """
        Parameters
        ----------
        attr : str
          Sample attribute used to determine splits.
        attr_values : list
          If not None, this is a list of value of the ``attr`` used to determine
          the splits. The order of values in this list defines the order of the
          resulting splits. It is possible to specify a particular value
          multiple times. All dataset samples with values that are not listed
          are going to be ignored.
        count : None or int
          Desired number of generated splits. If None, all splits are output
          (default), otherwise the number of splits is limited to the given
          ``count`` or the maximum number of possible split (whatever is less).
        noslicing : bool
          If True, dataset splitting is not done by slicing (causing
          shared data between source and split datasets) even if it would
          be possible. By default slicing is performed whenever possible
          to reduce the memory footprint.
        collection : {None, 'sa', 'fa'}
          Specify the collection that contains the split-defining attribute. If
          None the collections is auto-detected by searching the dataset
          collections (sample attributes first). Alternatively, it is possible
          to specified 'sa' (sample attribute) or 'fa' (feature attribute).
        reverse : bool
          If True, the order of datasets in the split is reversed, e.g.
          instead of (training, testing), (training, testing) will be spit
          out
        """
        Node.__init__(self, **kwargs)
        self.__splitattr = attr
        self.__splitattr_values = attr_values
        self.__count = count
        self.__noslicing = noslicing
        self.__collection = collection
        self.__reverse = reverse


    def generate(self, ds):
        """Yield dataset splits.

        Parameters
        ----------
        ds: Dataset
          Input dataset

        Returns
        -------
        generator
          The generator yields every possible split according to the splitter
          configuration. All generated dataset have a boolean 'lastsplit'
          attribute in their dataset attribute collection indicating whether
          this particular dataset is the last one.
        """
        # localbinding
        col_name = self.__collection
        noslicing = self.__noslicing
        count = self.__count
        splattr = self.__splitattr

        # get attribute and source collection from dataset
        splattr, collection = ds.get_attr(splattr)
        splattr_data = splattr.value
        cfgs = self.__splitattr_values
        if cfgs is None:
            cfgs = splattr.unique
        n_cfgs = len(cfgs)

        if self.__reverse:
            cfgs = cfgs[::-1]

        # split the data
        for isplit, split in enumerate(cfgs):
            if not count is None and isplit >= count:
                # number of max splits is reached
                break
            # boolean mask is 'selected' samples for this split
            filter_ = splattr_data == split

            if not noslicing:
                # check whether we can do slicing instead of advanced
                # indexing -- if we can split the dataset without causing
                # the data to be copied, its is quicker and leaner.
                # However, it only works if we have a contiguous chunk or
                # regular step sizes for the samples to be split
                filter_ = mask2slice(filter_)

            if collection is ds.sa:
                split_ds = ds[filter_]
            elif collection is ds.fa:
                split_ds = ds[:, filter_]
            else:
                RuntimeError("This should never happen.")

            # is this the last split
            if count is None:
                lastsplit = (isplit == n_cfgs - 1)
            else:
                lastsplit = (isplit == count - 1)

            if not split_ds.a.has_key('lastsplit'):
                # if not yet known -- add one
                split_ds.a['lastsplit'] = lastsplit
            else:
                # otherwise just assign a new value
                split_ds.a.lastsplit = lastsplit

            yield split_ds
