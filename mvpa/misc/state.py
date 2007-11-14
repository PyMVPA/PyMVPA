#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Class to control and store state information"""

__docformat__ = 'restructuredtext'

from mvpa.misc.exceptions import UnknownStateError

class State(dict):
    def __init__(self, *args, **kwargs):
        self.__registered = []
        dict.__init__(self, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, index):
        if not index in self.__registered:
            raise KeyError, \
                  "State of %s has no key %s registered" \
                  % (self.__class__.__name__, index)

        if not self.has_key(index):
            raise UnknownStateError

        return dict.__getitem__(self, index)

    def _register(self, key):
        self.__registered.append(key)
