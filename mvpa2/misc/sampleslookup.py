# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helper to map and validate samples' origids into indices"""

import numpy as np

if __debug__:
    from mvpa2.base import debug

class SamplesLookup(object):
    """Map to translate sample origids into unique indices.
    """

    def __init__(self, ds):
        """
        Parameters
        ----------
        ds : Dataset
            Dataset for which to create the map
        """

        # TODO: Generate origids and magic_id in Dataset!!
        # They are simply added here for development convenience, but they
        # should be removed.  We should also consider how exactly to calculate
        # the magic ids and sample ids as this is not necessarily the fastest/
        # most robust method --SG
        try:
            sample_ids = ds.sa.origids
        except AttributeError:
            # origids not yet generated
            if __debug__:
                debug('SAL',
                      "Generating dataset origids in SamplesLookup for %(ds)s",
                      msgargs=dict(ds=ds))

            ds.init_origids('samples')  # XXX may be both?
            sample_ids = ds.sa.origids

        try:
            self._orig_ds_id = ds.a.magic_id
        except AttributeError:
            ds.a.update({'magic_id': hash(ds)})
            self._orig_ds_id = ds.a.magic_id
            if __debug__:
                debug('SAL',
                      "Generating dataset magic_id in SamplesLookup for %(ds)s",
                      msgargs=dict(ds=ds))

        nsample_ids = len(sample_ids)
        self._map = dict(zip(sample_ids,
                             range(nsample_ids)))
        if __debug__:
            # some sanity checks
            if len(self._map) != nsample_ids:
                raise ValueError, \
                    "Apparently samples' origids are not uniquely identifying" \
                    " samples in %s.  You must change them so they are unique" \
                    ". Use ds.init_origids('samples')" % ds

    def __call__(self, ds):
        """
        .. note:
           Will raise KeyError if lookup for sample_ids fails, or ds has not
           been mapped at all
           """
        if (not 'magic_id' in ds.a) or ds.a.magic_id != self._orig_ds_id:
            raise KeyError, \
                  'Dataset %s is not indexed by %s' % (ds, self)

        _map = self._map
        _origids = ds.sa.origids

        res = np.array([_map[i] for i in _origids])
        if __debug__:
            debug('SAL',
                  "Successful lookup: %(inst)s on %(ds)s having "
                  "origids=%(origids)s resulted in %(res)s",
                  msgargs=dict(inst=self, ds=ds, origids=_origids, res=res))
        return res
