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

    def __checkIndex(self, index):
        if not index in self.__registered:
            raise KeyError, \
                  "State of %s has no key %s registered" \
                  % (self.__class__.__name__, index)

    # actually we have to provide some simple way to set the State
    def __setitem__(self, index, *args, **kwargs):
        self.__checkIndex(index)
        dict.__setitem__(self, index, *args, **kwargs)

    def __getitem__(self, index):
        # XXX Maybe unnecessary to check dict twice for matching member
        # if it is not registered the will be no key like this in the
        # dict, but if registered there need not be a key anyway.
        # Therefore it should be sufficient to check for key in dict.
        # Or do first test only in __debug__
        self.__checkIndex(index)
        if not self.has_key(index):
            raise UnknownStateError("Unknown yet value for '%s'" % index)
        else:
            return dict.__getitem__(self, index)

        return dict.__getitem__(self, index)
    # XXX think about it -- may be it is worth making whole State
    # handling via static methods to remove any possible overhead of
    # registering the same keys in each constructor
    # Michael: Depends on what kind of objects we want to have a state.
    #          Anyway as we will inherent this class I think the method name
    #          should be a bit more distinctive. What about:
    def _registerState(self, key):
        self.__registered.append(key)
