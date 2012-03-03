# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Generator nodes to permute datasets.
"""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base.node import Node
from mvpa2.base.dochelpers import _str, _repr
from mvpa2.misc.support import get_limit_filter

from mvpa2.support.utils import deprecated

class AttributePermutator(Node):
    """Node to permute one a more attributes in a dataset.

    This node can permute arbitrary sample or feature attributes in a dataset.
    Moreover, it supports limiting the permutation to a subset of samples or
    features (see ``limit`` argument). The node can simply be called with a
    dataset for a one time permutation, or used as a generator to produce
    multiple permutations.

    This node only permutes dataset attributes, dataset samples are no affected.
    The permuted output dataset shares the samples container with the input
    dataset.
    """
    def __init__(self, attr, count=1, limit=None, assure=False, **kwargs):
        """
        Parameters
        ----------
        attr : str or list(str)
          Name of the to-be-permuted attribute. This can also be a list of
          attribute names, in which case the *identical* shuffling is applied to
          all listed attributes.
        count : int
          Number of permutations to be yielded by .generate()
        limit : None or str or dict
          If ``None`` all attribute values will be permuted. If an single
          attribute name is given, its unique values will be used to define
          chunks of data that are permuted individually (i.e. no attributed
          values will be replaced across chunks). Finally, if a dictionary is
          provided, its keys define attribute names and its values (single value
          or sequence thereof) attribute value, where all key-value combinations
          across all given items define a "selection" of to-be-permuted samples
          or features.
        assure : bool
          If set, by-chance non-permutations will be prevented, i.e. it is
          checked that at least two items change their position. Since this
          check adds a runtime penalty it is off by default.
        """
        Node.__init__(self, **kwargs)
        self._pattr = attr

        self.count = count
        self._limit = limit
        self._pcfg = None
        self._assure_permute = assure


    def _get_pcfg(self, ds):
        # determine to be permuted attribute to find the collection
        pattr = self._pattr
        if isinstance(pattr, str):
            pattr, collection = ds.get_attr(pattr)
        else:
            # must be sequence of attrs, take first since we only need the shape
            pattr, collection = ds.get_attr(pattr[0])

        return get_limit_filter(self._limit, collection)


    def _call(self, ds):
        # local binding
        pattr = self._pattr
        assure_permute = self._assure_permute

        # get permutation setup if not set already (maybe from generate())
        if self._pcfg is None:
            pcfg = self._get_pcfg(ds)
        else:
            pcfg = self._pcfg

        if isinstance(pattr, str):
            # wrap single attr name into tuple to simplify the code
            pattr = (pattr,)

        # shallow copy of the dataset for output
        out = ds.copy(deep=False)

        for limit_value in np.unique(pcfg):
            if pcfg.dtype == np.bool:
                # simple boolean filter -> do nothing on False
                if not limit_value:
                    continue
                # otherwise get indices of "selected ones"
                limit_idx = pcfg.nonzero()[0]
            else:
                # non-boolean limiter -> determine "chunk" and permute within
                limit_idx = (pcfg == limit_value).nonzero()[0]

            # permute indices once and later apply the same permutation to all
            # desired attributes
            # make ten attempts of assure is set
            if assure_permute:
                proceed = False
                for i in range(10):
                    perm_idx = np.random.permutation(limit_idx)
                    if not np.all(perm_idx == limit_idx):
                        proceed = True
                        break
                if not proceed:
                    raise RuntimeError(
                          "Cannot assure permutation of %s.%s for "
                          "some reason (dataset %s). Should not happen"
                          % (pattr, ds))
            else:
                perm_idx = np.random.permutation(limit_idx)

            # need list to index properly
            limit_idx = list(limit_idx)

            # for all to be permuted attrs
            for pa in pattr:
                # input attr and collection
                in_pattr, in_collection = ds.get_attr(pa)
                # output attr and collection
                out_pattr, out_collection = out.get_attr(pa)
                # make a copy of the attr value array to decouple ownership
                out_values = out_pattr.value.copy()
                # replace all values in current limit with permutations
                out_values[limit_idx] = in_pattr.value[perm_idx]
                # reassign the attribute to overwrite any previous one
                out_pattr.value = out_values

        return out


    def generate(self, ds):
        """Generate the desired number of permuted datasets."""
        # figure out permutation setup once for all runs
        self._pcfg = self._get_pcfg(ds)
        # permute as often as requested
        for i in xrange(self.count):
            yield self(ds)

        # reset permutation setup to do the right thing upon next call to object
        self._pcfg = None


    def __str__(self):
        return _str(self, self._pattr, n=self.count, limit=self._limit,
                    assure=self._assure_permute)

    def __repr__(self, prefixes=[]):
        return super(AttributePermutator, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['attr'])
            + _repr_attrs(self, ['count'], default=1)
            + _repr_attrs(self, ['limit'])
            + _repr_attrs(self, ['assure'], default=False)
            )

    @property
    @deprecated("to be removed in 2.1 -- use .count instead")
    def nruns(self):
        return self.count

    attr = property(fget=lambda self: self._pattr)
    limit = property(fget=lambda self: self._limit)
    assure = property(fget=lambda self: self._assure_permute)
