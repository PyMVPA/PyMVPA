# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset content resampling (e.g. balance number of samples per condition)"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base.node import Node
from mvpa2.base.dochelpers import _str, _repr
from mvpa2.misc.support import get_limit_filter, get_nelements_per_value
from mvpa2.misc.support import get_rng


class Balancer(Node):
    """Generator to (repeatedly) select subsets of a dataset.

    The Balancer can equalize the number of samples/features in a dataset, or
    select an absolute number or fraction of all available data. Selection is
    performed given a particular attribute and additionally can be limited to
    a subset of the dataset defined by more complex criteria (see ``limit``
    argument). The node can either "mark" elements as selected by adding a
    corresponding attribute to the output dataset, or actually apply the
    selection by returning a new dataset with only selected elements.
    """
    def __init__(self,
                 amount='equal',
                 attr='targets',
                 count=1,
                 limit='chunks',
                 apply_selection=False,
                 include_offlimit=False,
                 space='balanced_set',
                 rng=None,
                 **kwargs):
        """
        Parameters
        ----------
        amount : {'equal'} or int or float
          Specify the amount of elements to be selected (within the current
          ``limit``). The amount can be given as an integer value corresponding
          to the absolute number of elements per unique attribute (see ``attr``)
          value, as a float corresponding to the fraction of elements, or with
          the keyword 'equal'. In the latter case the number of to be selected
          elements is determined by the least number of available elements for
          any given unique attribute value within the current limit.
        attr : str
          Dataset attribute whose unique values define element classes that are
          to be balanced in number.
        count : int
          How many iterations to perform on ``generate()``.
        limit : None or str or dict
          If ``None`` the whole dataset is considered as one. If a single
          attribute name is given, its unique values will be used to define
          chunks of data that are balanced individually. Finally, if a
          dictionary is provided, its keys define attribute names and its values
          (single value or sequence thereof) attribute value, where all
          key-value combinations across all given items define a "selection" of
          to-be-balanced samples or features.
        apply_selection : bool
          Flag whether the balanced selection shall be applied, i.e. the output
          dataset only contains selected elements. If False, the selection is
          instead added as an attribute that merely marks selected elements (see
          ``space`` argument).
        include_offlimit : bool
          If True, all samples that were off limit (i.e. not included in the
          balancing input are included in the balanced selection. If False
          (default) they are excluded.
        space : str
          Name of the selection marker attribute in the output dataset that is
          created if the balanced selection is not applied to the output dataset
          (see ``apply_selection`` argument).
        rng : int or RandomState, optional
          Integer to seed a new RandomState upon each call, or instance of the
          numpy.random.RandomState to be reused across calls. If None, the
          numpy.random singleton would be used
        """
        Node.__init__(self, space=space, **kwargs)
        self._amount = amount
        self._attr = attr
        self.count = count
        self._limit = limit
        self._include_offlimit = include_offlimit
        self._apply_selection = apply_selection
        self._rng = rng

    def _get_call_kwargs(self, ds):
        attr, collection = ds.get_attr(self._attr)
        # _call might need to operate on the dedicated instantiated rng
        # e.g. if seed int is provided
        return {
            'limit_filter': get_limit_filter(self._limit, collection),
            'rng': get_rng(self._rng)
        }

    def _call(self, ds, limit_filter=None, rng=None):
        # local binding
        amount = self._amount
        attr, collection = ds.get_attr(self._attr)

        # ids of elements that are part of the balanced set
        balanced_set = []
        full_limit_set = []

        # for each chunk in the filter (might be just the selected ones)
        for limit_value in np.unique(limit_filter):
            if limit_filter.dtype == np.bool:
                # simple boolean filter -> do nothing on False
                if not limit_value:
                    continue
                # otherwise get indices of "selected ones"
                limit_idx = limit_filter.nonzero()[0]
            else:
                # non-boolean limiter -> determine "chunk" and balance within
                limit_idx = (limit_filter == limit_value).nonzero()[0]
            full_limit_set += list(limit_idx)

            # apply the current limit to the target attribute
            # need list to index properly
            attr_limited = attr[list(limit_idx)]
            uattr_limited = np.unique(attr_limited)

            # handle all types of supported arguments
            if amount == 'equal':
                # go for maximum possible number of samples provided
                # by each label in this dataset
                # determine the min number of samples per class
                epa = get_nelements_per_value(attr_limited)
                min_epa = min(epa.values())
                for k in epa:
                    epa[k] = min_epa
            elif isinstance(amount, float):
                epa = get_nelements_per_value(attr_limited)
                for k in epa:
                    epa[k] = int(round(epa[k] * amount))
            elif isinstance(amount, int):
                epa = dict(zip(uattr_limited, [amount] * len(uattr_limited)))
            else:
                raise ValueError("Unknown type of amount argument '%s'" % amount)

            # select determined number of elements per unique attribute value
            selected = []
            for ua in uattr_limited:
                selected += rng.permutation((attr_limited == ua).nonzero()[0])[:epa[ua]].tolist()

            # determine the final indices of selected elements and store
            # as part of the balanced set
            balanced_set += list(limit_idx[selected])

        # make full-sized boolean selection attribute and put it into
        # the right collection of the output dataset
        if self._include_offlimit:
            # start with all-in
            battr = np.ones(len(attr), dtype=np.bool)
            # throw out all samples that could have been limited
            battr[full_limit_set] = False
            # put back the ones that got into the balanced set
            battr[balanced_set] = True
        else:
            # start with nothing
            battr = np.zeros(len(attr), dtype=np.bool)
            # only keep the balanced set
            battr[balanced_set] = True

        if self._apply_selection:
            if collection is ds.sa:
                return ds[battr]
            elif collection is ds.fa:
                return ds[:, battr]
            else:
                # paranoid
                raise RuntimeError(
                        "Don't know where this collection comes from. "
                        "This should never happen!")
        else:
            # shallow copy of the dataset for output
            out = ds.copy(deep=False)
            if collection is ds.sa:
                out.sa[self.get_space()] = battr
            elif collection is ds.fa:
                out.fa[self.get_space()] = battr
            else:
                # paranoid
                raise RuntimeError(
                        "Don't know where this collection comes from. "
                        "This should never happen!")
            return out

    def generate(self, ds):
        """Generate the desired number of balanced datasets datasets."""
        # figure out filter for all runs at once
        # permute as often as requested, reusing the same kwargs
        kwargs = self._get_call_kwargs(ds)
        for i in xrange(self.count):
            yield self(ds, _call_kwargs=kwargs)

    def __str__(self):
        return _str(self, str(self._amount), n=self._attr, count=self.count,
                    apply_selection=self._apply_selection)
