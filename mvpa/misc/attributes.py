# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Module with some special objects to be used as magic attributes with
dedicated containers aka. `Collections`.
"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.base.collections import Collectable

from mvpa.misc.exceptions import UnknownStateError
import mvpa.support.copy as copy

if __debug__:
    from mvpa.base import debug



##################################################################
# Various attributes which will be collected into collections
#
class IndexedCollectable(Collectable):
    """Collectable with position information specified with index

    Derived classes will have specific semantics:

    * StateVariable: conditional storage
    * Parameter: attribute with validity ranges.

    `IndexedAttributes` instances are to be automagically grouped into
    corresponding collections for each class by `StateCollector`
    metaclass, i.e. it would be done on a class creation (i.e. not per
    each instance).  Upon instance creation those collection templates
    will be copied for the instance.
    """

    _instance_index = 0

    def __init__(self, index=None, **kwargs):
        """
        Parameters
        ----------
        value : arbitrary (see derived implementations)
          The actual value of this attribute.
        **kwargs
          Passed to `Collectable`
        """
        if index is None:
            IndexedCollectable._instance_index += 1
            index = IndexedCollectable._instance_index
        else:
            # TODO: there can be collision between custom provided indexes
            #       and the ones automagically assigned.
            #       Check might be due
            pass
        self._instance_index = index

        self._isset = False
        self.reset()

        Collectable.__init__(self, **kwargs)

        if __debug__ and 'COL' in debug.active:
            debug("COL",
                  "Initialized new IndexedCollectable #%d:%s %r"
                  % (index, self.name, self))

    # XXX shows how indexing was screwed up -- not copied etc
    #def __copy__(self):
    #    # preserve attribute type
    #    copied = self.__class__(name=self.name, doc=self.__doc__)
    #    # just get a view of the old data!
    #    copied.value = copy.copy(self.value)
    #    return copied


    # XXX had to override due to _isset, init=
    def _set(self, val, init=False):
        """4Developers: Override this method in derived classes if you desire
           some logic (drop value in case of states, or not allow to set value
           for read-only Parameters unless called with init=1) etc)
        """
        if __debug__: # Since this call is quite often, don't convert
            # values to strings here, rely on passing them # withing
            debug("COL",
                  "%(istr)s %(self)s to %(val)s ",
                  msgargs={'istr':{True: 'Initializing',
                                   False: 'Setting'}[init],
                           'self':self, 'val':val})
        self._value = val
        self._isset = True


    @property
    def isSet(self):
        return self._isset


    def reset(self):
        """Simply reset the flag"""
        if __debug__ and self._isset:
            debug("COL", "Reset %s to being non-modified" % self.name)
        self._isset = False


    # TODO XXX unify all bloody __str__
    def __str__(self):
        res = "%s" % (self.name)
        if self.isSet:
            res += '*'          # so we have the value already
        return res

    # XXX  reports value depending on _isset
    def __repr__(self):
        if not self._isset:
            value = None
        else:
            value = self.value
        return "%s(index=%s, value=%s, name=%s, doc=%s)" % (
            self.__class__.__name__,
            self._instance_index,
            repr(self.name),
            repr(self.__doc__),
            repr(value))


class StateVariable(IndexedCollectable):
    """Simple container intended to conditionally store the value
    """

    def __init__(self, enabled=True, **kwargs):
        """
        Parameters
        ----------
        enabled : bool
          If a StateVariable is not enabled then assignment of any value has no
          effect, i.e. nothing is stored.
        **kwargs
          Passed to `IndexedCollectable`
        """
        # Force enabled state regardless of the input
        # to facilitate testing
        if __debug__ and 'ENFORCE_STATES_ENABLED' in debug.active:
            enabled = True
        IndexedCollectable.__init__(self, **kwargs)
        self._isenabled = enabled
        self._defaultenabled = enabled


    def _get(self):
        if not self.isSet:
            raise UnknownStateError("Unknown yet value of %s" % (self.name))
        return IndexedCollectable._get(self)


    def _set(self, val, init=False):
        if self.isEnabled:
            # XXX may be should have left simple assignment
            # self._value = val
            IndexedCollectable._set(self, val)
        elif __debug__:
            debug("COL",
                  "Not setting disabled %(self)s to %(val)s ",
                  msgargs={'self':self, 'val':val})


    def reset(self):
        """Simply detach the value, and reset the flag"""
        IndexedCollectable.reset(self)
        self._value = None


    @property
    def isEnabled(self):
        return self._isenabled


    def enable(self, value=False):
        if self._isenabled == value:
            # Do nothing since it is already in proper state
            return
        if __debug__:
            debug("STV", "%s %s" %
                  ({True: 'Enabling', False: 'Disabling'}[value],
                   self))
        self._isenabled = value


    def __str__(self):
        res = IndexedCollectable.__str__(self)
        if self.isEnabled:
            res += '+'          # it is enabled but no value is assigned yet
        return res
