# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helper to map literal attribute to numerical ones (and back)"""

import numpy as N

class AttributeMap(object):
    """
    """
    def __init__(self, map=None):
        self.reset()
        if not map is None:
            self._nmap = map

    def reset(self):
        # map from literal TO numeric
        self._nmap = None
        # map from numeric TO literal
        self._lmap = None

    def to_numeric(self, attr):
        """
        """
        attr = N.asanyarray(attr)

        if self._nmap is None:
            # sorted list of unique attr values
            ua = N.unique(attr)
            self._nmap = dict(zip(ua, range(len(ua))))

        num = N.empty(attr.shape, dtype=N.int)
        for k, v in self._nmap.iteritems():
            num[attr == k] = v

        return num


    def to_literal(self, attr):
        """
        """
        # we need one or the other map
        if self._lmap is None and self._nmap is None:
            raise RuntimeError("AttributeMap has no mapping information. "
                               "Ever called to_numeric()?")

        if self._lmap is None:
            self._lmap = dict([(v, k) for k, v in self._nmap.iteritems()])

        lmap = self._lmap

        return N.asanyarray([lmap[k] for k in attr])
