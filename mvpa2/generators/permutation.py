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

from mvpa2.base import warning
from mvpa2.base.dochelpers import _repr_attrs

from mvpa2.base.node import Node
from mvpa2.base.dochelpers import _str, _repr
from mvpa2.misc.support import get_limit_filter
from mvpa2.misc.support import get_rng

from mvpa2.support.utils import deprecated
from mvpa2.mappers.fx import _product

if __debug__:
    from mvpa2.base import debug

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
    def __init__(self, attr, count=1, limit=None, assure=False,
                 strategy='simple', chunk_attr=None, rng=None, **kwargs):
        """
        Parameters
        ----------
        attr : str or list(str)
          Name of the to-be-permuted attribute. This can also be a list of
          attribute names, in which case the *identical* shuffling is applied to
          all listed attributes.
        count : int
          Number of permutations to be yielded by .generate()
        limit : None or str or list or dict
          If ``None`` all attribute values will be permuted. If a single
          attribute name is given, its unique values will be used to define
          chunks of data that are permuted individually (i.e. no attributed
          values will be replaced across chunks). If a list given, then combination
          of those attributes per each sample is used together. Finally, if a dictionary is
          provided, its keys define attribute names and its values (single value
          or sequence thereof) attribute value, where all key-value combinations
          across all given items define a "selection" of to-be-permuted samples
          or features.
        strategy : 'simple', 'uattrs', 'chunks'
          'simple' strategy is the straightforward permutation of attributes (given
          the limit).  In some sense it assumes independence of those samples.
          'uattrs' strategy looks at unique values of attr (or their unique
          combinations in case of `attr` being a list), and "permutes" those
          unique combinations values thus breaking their assignment to the samples
          but preserving any dependencies between samples within the same unique
          combination. The 'chunks' strategy swaps attribute values of entire chunks.
          Naturally, this will only work if there is the same number of samples in
          all chunks.
        assure : bool
          If set, by-chance non-permutations will be prevented, i.e. it is
          checked that at least two items change their position. Since this
          check adds a runtime penalty it is off by default.
        rng : int or RandomState, optional
          Integer to seed a new RandomState upon each call, or instance of the
          numpy.random.RandomState to be reused across calls. If None, the
          numpy.random singleton would be used


        """
        Node.__init__(self, **kwargs)
        self._pattr = attr

        self.count = count
        self._limit = limit

        self._assure_permute = assure
        self.strategy = strategy
        self.rng = rng
        self.chunk_attr = chunk_attr

    def _get_call_kwargs(self, ds):
        # determine to be permuted attribute to find the collection
        pattr = self._pattr
        if isinstance(pattr, str):
            pattr, collection = ds.get_attr(pattr)
        else:
            # must be sequence of attrs, take first since we only need the shape
            pattr, collection = ds.get_attr(pattr[0])

        # _call might need to operate on the dedicated instantiated rng
        # e.g. if seed int is provided
        return {
            'limit_filter': get_limit_filter(self._limit, collection),
            'rng': get_rng(self.rng)
        }

    def _call(self, ds, limit_filter=None, rng=None):
        # local binding
        pattr = self._pattr
        assure_permute = self._assure_permute

        if isinstance(pattr, str):
            # wrap single attr name into tuple to simplify the code
            pattr = (pattr,)

        # get actual attributes
        in_pattrs = [ds.get_attr(pa)[0] for pa in pattr]

        # Method to use for permutations
        try:
            permute_fx = getattr(self, "_permute_%s" % self.strategy)
            permute_kwargs = {'rng': rng}
        except AttributeError:
            raise ValueError("Unknown permutation strategy %r" % self.strategy)

        if self.chunk_attr is not None:
            permute_kwargs['chunks'] = ds.sa[self.chunk_attr].value

        for i in xrange(10):  # for the case of assure_permute
            # shallow copy of the dataset for output
            out = ds.copy(deep=False)

            out_pattrs = [out.get_attr(pa)[0] for pa in pattr]
            # replace .values with copies in out_pattrs so we do
            # not override original values
            for pa in out_pattrs:
                pa.value = pa.value.copy()

            for limit_value in np.unique(limit_filter):
                if limit_filter.dtype == np.bool:
                    # simple boolean filter -> do nothing on False
                    if not limit_value:
                        continue
                    # otherwise get indices of "selected ones"
                    limit_idx = limit_filter.nonzero()[0]
                else:
                    # non-boolean limiter -> determine "chunk" and permute within
                    limit_idx = (limit_filter == limit_value).nonzero()[0]

                # need list to index properly
                limit_idx = list(limit_idx)

                permute_fx(limit_idx, in_pattrs, out_pattrs, **permute_kwargs)

            if not assure_permute:
                break

            # otherwise check if we differ from original, and if so -- break
            differ = False
            for in_pattr, out_pattr in zip(in_pattrs, out_pattrs):
                differ = differ or np.any(in_pattr.value != out_pattr.value)
                if differ:
                    break                 # leave check loop if differ
            if differ:
                break                     # leave 10 loop, otherwise go to the next round

        if assure_permute and not differ:
            raise RuntimeError(
                "Cannot assure permutation of %s with limit %r for "
                "some reason (dataset %s). Should not happen"
                % (pattr, self._limit, ds))

        return out


    def _permute_simple(self, limit_idx, in_pattrs, out_pattrs, rng=None):
        """The simplest permutation
        """
        perm_idx = rng.permutation(limit_idx)

        if __debug__:
            debug('APERM', "Obtained permutation %s", (perm_idx, ))

        # for all to be permuted attrs
        for in_pattr, out_pattr in zip(in_pattrs, out_pattrs):
            # replace all values in current limit with permutations
            # of the original ds's attributes
            out_pattr.value[limit_idx] = in_pattr.value[perm_idx]


    def _permute_uattrs(self, limit_idx, in_pattrs, out_pattrs, rng=None):
        """Provide a permutation given a specified strategy
        """
        # Select given limit_idx
        pattrs_lim = [p.value[limit_idx] for p in in_pattrs]
        # convert to list of tuples
        pattrs_lim_zip = zip(*pattrs_lim)
        # find unique groups
        unique_groups = list(set(pattrs_lim_zip))
        # now we need to permute the groups to generate remapping
        # get permutation indexes first
        perm_idx = rng.permutation(np.arange(len(unique_groups)))
        # generate remapping
        remapping = dict([(t, unique_groups[i])
                          for t, i in zip(unique_groups, perm_idx)])
        if __debug__:
            debug('APERM', "Using remapping %s", (remapping,))

        for i, in_group in zip(limit_idx, pattrs_lim_zip):
            out_group = remapping[in_group]
            # now we need to assign them ot out_pattrs
            for pa, out_v in zip(out_pattrs, out_group):
                pa.value[i] = out_v

    @staticmethod
    def _permute_chunks_sanity_check(in_pattrs, chunks, uniques):
        #  Verify that we are not dealing with some degenerate scenario

        for in_pattr in in_pattrs:
            sample_targets = in_pattr.value[np.where(chunks == uniques[0])]

            for orig in uniques[1:]:
                chunk_targets = in_pattr.value[np.where(chunks == orig)]
                # must be of the same length
                if np.any(chunk_targets != sample_targets):
                    # Escape as early as possible
                    return

        warning("Permutation via strategy='chunk' makes no sense --"
                " all chunks have the same order of targets: %s"
                % (sample_targets,))

    def _permute_chunks(self, limit_idx, in_pattrs, out_pattrs, chunks=None, rng=None):
        # limit_idx is doing nothing

        if chunks is None:
            raise ValueError("Missing 'chunk_attr' for strategy='chunk'")

        uniques = np.unique(chunks)

        if __debug__ and len(uniques):
            # Somewhat a duplication, since could be checked within the loop,
            # but IMHO makes it cleaner and shouldn't be that big of an impact
            self._permute_chunks_sanity_check(in_pattrs, chunks, uniques)

        for in_pattr, out_pattr in zip(in_pattrs, out_pattrs):
            shuffled = uniques.copy()
            rng.shuffle(shuffled)

            for orig, new in zip(uniques, shuffled):
                out_pattr.value[np.where(chunks == orig)] = \
                    in_pattr.value[np.where(chunks == new)]

    def generate(self, ds):
        """Generate the desired number of permuted datasets."""
        # figure out permutation setup once for all runs
        # permute as often as requested
        for i in xrange(self.count):
            kwargs = self._get_call_kwargs(ds)
            ## if __debug__:
            ##     debug('APERM', "%s generating %i-th permutation", (self, i))
            yield self(ds, _call_kwargs=kwargs)

    def __str__(self):
        return _str(self, self._pattr, n=self.count, limit=self._limit,
                    assure=self._assure_permute)

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        return super(AttributePermutator, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['attr'])
            + _repr_attrs(self, ['count'], default=1)
            + _repr_attrs(self, ['limit'])
            + _repr_attrs(self, ['assure'], default=False)
            + _repr_attrs(self, ['strategy'], default='simple')
            + _repr_attrs(self, ['rng'], default=None)
            )

    attr = property(fget=lambda self: self._pattr)
    limit = property(fget=lambda self: self._limit)
    assure = property(fget=lambda self: self._assure_permute)
