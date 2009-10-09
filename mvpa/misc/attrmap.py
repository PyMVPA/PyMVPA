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
    # might be derived from dict, but do not see advantages right now,
    # since this one has forward and reverse map
    # however, it might be desirable to implement more of the dict interface
    """Map to translate literal values to numeric ones (and back).

    A translation map is derived automatically from the argument of the first
    call to to_numeric(). The default mapping is to map unique value
    (in sorted order) to increasing integer values starting from zero.

    In case the default mapping is undesired a custom map can be specified to
    the constructor call.

    Regardless of how the mapping has been specified or derived, it remains
    constant (i.e. it is not influenced by subsequent calls to meth:`to_numeric`
    or meth:`to_literal`. However, the translation map can be removed with
    meth:`clear`.

    Both conversion methods take sequence-like input and return arrays.

    Examples
    --------

    Default mapping procedure using an automatically derived translation map:

    >>> am = AttributeMap()
    >>> am.to_numeric(['eins', 'zwei', 'drei'])
    array([1, 2, 0])

    >>> print am.to_literal([1, 2, 0])
    ['eins' 'zwei' 'drei']

    Custom mapping:

    >>> am = AttributeMap(map={'eins': 11, 'zwei': 22, 'drei': 33})
    >>> am.to_numeric(['eins', 'zwei', 'drei'])
    array([11, 22, 33])
    """
    def __init__(self, map=None):
        """
        Parameters
        ----------
        map : dict
          Custom dict with literal keys mapping to numerical values.

        Please see the class documentation for more information.
        """
        self.clear()
        if not map is None:
            if not isinstance(map, dict):
                raise ValueError("Custom map need to be a dict.")
            self._nmap = map

    def clear(self):
        """Remove previously established mappings."""
        # map from literal TO numeric
        self._nmap = None
        # map from numeric TO literal
        self._lmap = None

    def keys(self):
        """Returns the literal names of the attribute map."""
        if self._nmap is None:
            return None
        else:
            return self._nmap.keys()

    def values(self):
        """Returns the numerical values of the attribute map."""
        if self._nmap is None:
            return None
        else:
            return self._nmap.values()

    def iteritems(self):
        """Dict-like generator yielding literal/numerical pairs."""
        if self._nmap is None:
            raise StopIteration
        else:
            for k, v in self._nmap:
                yield k, v

    def to_numeric(self, attr):
        """Map literal attribute values to numerical ones.

        Parameters
        ----------
        attr : sequence
          Literal values to be mapped.

        Please see the class documentation for more information.
        """
        attr = N.asanyarray(attr)

        # no mapping if already numeric
        if not N.issubdtype(attr.dtype, 'c'):
            return attr

        if self._nmap is None:
            # sorted list of unique attr values
            ua = N.unique(attr)
            self._nmap = dict(zip(ua, range(len(ua))))
        elif __debug__:
            ua = N.unique(attr)
            mkeys = sorted(self._nmap.keys())
            if (ua != mkeys).any():
                # maps to not match
                raise KeyError("Existing attribute map not suitable for "
                        "to be mapped attribute (i.e. unknown values. "
                        "Attribute has '%s', but map has '%s'."
                        % (str(ua), str(mkeys)))


        num = N.empty(attr.shape, dtype=N.int)
        for k, v in self._nmap.iteritems():
            num[attr == k] = v
        return num

    def to_literal(self, attr):
        """Map numerical value back to literal ones.

        Parameters
        ----------
        attr : sequence
          Numerical values to be mapped.

        Please see the class documentation for more information.
        """
        # we need one or the other map
        if self._lmap is None and self._nmap is None:
            raise RuntimeError("AttributeMap has no mapping information. "
                               "Ever called to_numeric()?")

        if self._lmap is None:
            self._lmap = dict([(v, k) for k, v in self._nmap.iteritems()])
        lmap = self._lmap
        return N.asanyarray([lmap[k] for k in attr])
