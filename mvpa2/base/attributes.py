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

import numpy as np

from mvpa2.base.collections import Collectable

from mvpa2.misc.exceptions import UnknownStateError
import mvpa2.support.copy as copy

if __debug__:
    from mvpa2.base import debug



##################################################################
# Various attributes which will be collected into collections
#
class IndexedCollectable(Collectable):
    """Collectable with position information specified with index

    Derived classes will have specific semantics:

    * ConditionalAttribute: conditional storage
    * Parameter: attribute with validity ranges.

    `IndexedAttributes` instances are to be automagically grouped into
    corresponding collections for each class by `StateCollector`
    metaclass, i.e. it would be done on a class creation (i.e. not per
    each instance).  Upon instance creation those collection templates
    will be copied for the instance.
    """

    _instance_index = 0

    def __init__(self, index=None, *args, **kwargs):
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

        Collectable.__init__(self, *args, **kwargs)

        if __debug__ and 'COL' in debug.active:
            debug("COL", "Initialized new IndexedCollectable #%d:%s %r",
                  (index, self.name, self))

    # XXX shows how indexing was screwed up -- not copied etc
    #def __copy__(self):
    #    # preserve attribute type
    #    copied = self.__class__(name=self.name, doc=self.__doc__)
    #    # just get a view of the old data!
    #    copied.value = copy.copy(self.value)
    #    return copied

    def __reduce__(self):
        cr = Collectable.__reduce__(self)
        assert(len(cr) == 2)            # otherwise we need to change logic below
        res = (cr[0],
                (self._instance_index,) + cr[1],
                {'_isset' : self._isset})
        #if __debug__ and 'COL_RED' in debug.active:
        #    debug('COL_RED', 'Returning %s for %s' % (res, self))
        return res


    # XXX had to override due to _isset, init=
    def _set(self, val, init=False):
        """4Developers: Override this method in derived classes if you desire
           some logic (drop value in case of ca, or not allow to set value
           for read-only Parameters unless called with init=1) etc)
        """
        if __debug__: # Since this call is quite often, don't convert
            # values to strings here, rely on passing them # withing
            debug("COL", "%s %s to %s ",
                  ({True: 'Initializing', False: 'Setting'}[init],
                   self, val))
        self._value = val
        self._isset = True


    @property
    def is_set(self):
        return self._isset


    def reset(self):
        """Simply reset the flag"""
        if __debug__ and self._isset:
            debug("COL", "Reset %s to being non-modified", (self.name,))
        self._isset = False


    # TODO XXX unify all bloody __str__
    def __str__(self):
        res = "%s" % (self.name)
        if self.is_set:
            res += '*'          # so we have the value already
        return res

    # XXX  reports value depending on _isset
    def __repr__(self):
        if not self._isset:
            value = None
        else:
            value = self.value
        return "%s(value=%s, name=%s, doc=%s, index=%s)" % (
            self.__class__.__name__,
            repr(value),
            repr(self.name),
            repr(self.__doc__),
            self._instance_index,
            )


class ConditionalAttribute(IndexedCollectable):
    """Simple container intended to conditionally store the value
    """

    def __init__(self, enabled=True, *args, **kwargs):
        """
        Parameters
        ----------
        enabled : bool
          If a ConditionalAttribute is not enabled then assignment of any value has no
          effect, i.e. nothing is stored.
        **kwargs
          Passed to `IndexedCollectable`
        """
        # Force enabled state regardless of the input
        # to facilitate testing
        if __debug__ and 'ENFORCE_CA_ENABLED' in debug.active:
            enabled = True
        self.__enabled = enabled
        self._defaultenabled = enabled
        IndexedCollectable.__init__(self, *args, **kwargs)

    def __reduce__(self):
        icr = IndexedCollectable.__reduce__(self)
        icr[2].update({'_defaultenabled' : self._defaultenabled,
                       '_value': self._value})
        # kill the value from Collectable, because we have to put it in the dict
        # to prevent loosing it during reconstruction when the CA is disabled
        res = (icr[0], (self.__enabled, icr[1][0], None) + icr[1][2:], icr[2])
        #if __debug__ and 'COL_RED' in debug.active:
        #    debug('COL_RED', 'Returning %s for %s' % (res, self))
        return res

    def __str__(self):
        res = IndexedCollectable.__str__(self)
        if self.__enabled:
            res += '+'          # it is enabled but no value is assigned yet
        return res


    def _get(self):
        if not self.is_set:
            raise UnknownStateError("Unknown yet value of %s" % (self.name))
        return IndexedCollectable._get(self)


    def _set(self, val, init=False):
        if self.__enabled:
            # XXX may be should have left simple assignment
            # self._value = val
            IndexedCollectable._set(self, val)
        elif __debug__:
            debug("COL", "Not setting disabled %s to %s ",
                  (self, val))


    def reset(self):
        """Simply detach the value, and reset the flag"""
        IndexedCollectable.reset(self)
        self._value = None


    def _get_enabled(self):
        return self.__enabled


    def _set_enabled(self, value=False):
        if self.__enabled == value:
            # Do nothing since it is already in proper state
            return
        if __debug__:
            debug("STV", "%s %s",
                  ({True: 'Enabling', False: 'Disabling'}[value],
                   self))
        self.__enabled = value


    enabled = property(fget=_get_enabled, fset=_set_enabled)
