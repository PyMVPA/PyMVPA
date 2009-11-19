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

from mvpa.misc.exceptions import UnknownStateError
import mvpa.support.copy as copy

if __debug__:
    from mvpa.base import debug



##################################################################
# Various attributes which will be collected into collections
#
class CollectableAttribute(object):
    # XXX Most of the stuff below should go into the module docstring
    """Base class for any custom behaving attribute intended to become
    part of a collection.

    Derived classes will have specific semantics:

    * StateVariable: conditional storage
    * AttributeWithArray: easy access to a set of unique values
      within a container
    * Parameter: attribute with validity ranges.

      - ClassifierParameter: specialization to become a part of
        Classifier's params collection
      - KernelParameter: --//-- to become a part of Kernel Classifier's
        kernel_params collection

    Those CollectableAttributes are to be groupped into corresponding
    collections for each class by statecollector metaclass, ie it
    would be done on a class creation (ie not per each object)
    """

    _instance_index = 0

    def __init__(self, name=None, doc=None, index=None, value=None):
        """
        Parameters
        ----------
        name : str
          Name of the attribute under which it should be available in its
          respective collection.
        doc : str
          Documentation about the purpose of this attribute.
        index : ???
          ???
        value : arbitrary (see derived implementations)
          The actual value of this attribute.
        """
        if index is None:
            CollectableAttribute._instance_index += 1
            index = CollectableAttribute._instance_index
        self._instance_index = index
        self.__doc__ = doc
        self.__name = name
        self._value = None
        self._isset = False
        self.reset()
        if not value is None:
            self._set(value)
        if __debug__:
            debug("COL",
                  "Initialized new collectable #%d:%s" % (index,name) + `self`)


    def __copy__(self):
        # preserve attribute type
        copied = self.__class__(name=self.name, doc=self.__doc__)
        # just get a view of the old data!
        copied.value = copy.copy(self.value)
        return copied


    # Instead of going for VProperty lets make use of virtual method
    def _getVirtual(self):
        return self._get()


    def _setVirtual(self, value):
        return self._set(value)


    def _get(self):
        return self._value


    def _set(self, val):
        if __debug__:
            # Since this call is quite often, don't convert
            # values to strings here, rely on passing them
            # withing msgargs
            debug("COL",
                  "Setting %(self)s to %(val)s ",
                  msgargs={'self':self, 'val':val})
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


    def __repr__(self):
        if not self._isset:
            value = None
        else:
            value = self.value
        return "%s(name=%s, doc=%s, value=%s)" % (self.__class__.__name__,
                                                  repr(self.name),
                                                  repr(self.__doc__),
                                                  repr(value))


    def _getName(self):
        return self.__name


    def _setName(self, name):
        """Set the name of parameter

        .. note::
          Should not be called for an attribute which is already assigned
          to a collection
        """
        if name is not None:
            if isinstance(name, basestring):
                if name[0] == '_':
                    raise ValueError, \
                          "Collectable attribute name must not start " \
                          "with _. Got %s" % name
            else:
                raise ValueError, \
                      "Collectable attribute name must be a string. " \
                      "Got %s" % `name`
        self.__name = name


    # XXX should become vproperty?
    # YYY yoh: not sure... someone has to do performance testing
    #     to see which is more effective. My wild guess is that
    #     _[gs]etVirtual would be faster
    value = property(_getVirtual, _setVirtual)
    name = property(_getName) #, _setName)



class AttributeWithArray(CollectableAttribute):
    """Container which embeds an array.

    It also takes care about caching and recomputing unique values.
    """

    def __init__(self, name=None, doc="Attribute with array",
                 value=None):
        """
        Parameters
        ----------
        name : str
          Name of the attribute under which it should be available in its
          respective collection.
        doc : str
          Documentation about the purpose of this attribute.
        value : arbitrary (see derived implementations)
          The actual value of this attribute.
        """
        self.__target_length = None
        CollectableAttribute.__init__(self, name=name, doc=doc, value=value)
        self._resetUnique()
        if __debug__:
            debug("UATTR",
                  "Initialized new AttributeWithArray %s " % name + `self`)


    def __copy__(self):
        # preserve attribute type
        copied = self.__class__(name=self.name, doc=self.__doc__)
        # just get a view of the old data!
        copied.value = self.value.view()
        return copied


    def reset(self):
        super(AttributeWithArray, self).reset()
        self._resetUnique()


    def _resetUnique(self):
        self._uniqueValues = None


    def _set(self, val):
        # check if the new value has the desired length -- fi length checking is
        # desired at all
        if not self.__target_length is None \
           and len(val) != self.__target_length:
            raise ValueError("Value length [%i] does not match the required "
                             "length [%i] of attribute '%s'."
                             % (len(val),
                                self.__target_length,
                                str(self.name)))

        self._resetUnique()
        CollectableAttribute._set(self, N.asanyarray(val))


    def set_length_check(self, value):
        """
        Parameters
        ----------
        value : int
          When setting the value of this attribute it is checked if it has
          this length.
        """
        self.__target_length = value


    def _getUniqueValues(self):
        if self.value is None:
            return None
        if self._uniqueValues is None:
            # XXX we might better use Set, but yoh recalls that
            #     N.unique was more efficient. May be we should check
            #     on the the class and use Set only if we are not
            #     dealing with ndarray (or lists/tuples)
            self._uniqueValues = N.unique(self.value)
        return self._uniqueValues

    unique = property(fget=_getUniqueValues)



# Hooks for comprehendable semantics and automatic collection generation
class SampleAttribute(AttributeWithArray):
    pass



class FeatureAttribute(AttributeWithArray):
    pass



class DatasetAttribute(CollectableAttribute):
    pass



class StateVariable(CollectableAttribute):
    """Simple container intended to conditionally store the value
    """

    def __init__(self, name=None, enabled=True, doc="State variable"):
        """
        Parameters
        ----------
        name : str
          Name of the attribute under which it should be available in its
          respective collection.
        doc : str
          Documentation about the purpose of this attribute.
        enabled : bool
          If a StateVariable is not enabled then assignment of any value has no
          effect, i.e. nothing is stored.
        value : arbitrary (see derived implementations)
          The actual value of this attribute.
        """
        # Force enabled state regardless of the input
        # to facilitate testing
        if __debug__ and 'ENFORCE_STATES_ENABLED' in debug.active:
            enabled = True
        CollectableAttribute.__init__(self, name=name, doc=doc)
        self._isenabled = enabled
        self._defaultenabled = enabled
        if __debug__:
            debug("STV",
                  "Initialized new state variable %s " % name + `self`)


    def _get(self):
        if not self.isSet:
            raise UnknownStateError("Unknown yet value of %s" % (self.name))
        return CollectableAttribute._get(self)


    def _set(self, val):
        if self.isEnabled:
            # XXX may be should have left simple assignment
            # self._value = val
            CollectableAttribute._set(self, val)
        elif __debug__:
            debug("COL",
                  "Not setting disabled %(self)s to %(val)s ",
                  msgargs={'self':self, 'val':val})


    def reset(self):
        """Simply detach the value, and reset the flag"""
        CollectableAttribute.reset(self)
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
                  ({True: 'Enabling', False: 'Disabling'}[value], str(self)))
        self._isenabled = value


    def __str__(self):
        res = CollectableAttribute.__str__(self)
        if self.isEnabled:
            res += '+'          # it is enabled but no value is assigned yet
        return res
