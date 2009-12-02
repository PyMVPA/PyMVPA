# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helper to map and validate samples' origids into indices"""

import numpy as N

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
        sample_ids = ds.sa.origids
        self._orig_ds_id = ds.a.magic_id
        self._map = dict(zip(sample_ids,
                             range(len(sample_ids))))

    def __call__(self, ds):
        """
        .. note:
           Would raise KeyError if lookup for sample_ids fails.
           Does not validate if it is for the same ds as was inited for
           """
        if ds.a.magic_id != self._orig_ds_id:
            raise KeyError()
        _map = self._map
        return N.array([_map[i] for i in ds.sa.origids])

