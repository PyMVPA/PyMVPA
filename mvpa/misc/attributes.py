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


from mvpa.misc.exceptions import UnknownStateError

if __debug__:
    from mvpa.base import debug



##################################################################
# Various attributes which will be collected into collections
#
class CollectableAttribute(object):
    """Base class for any custom behaving attribute intended to become
    part of a collection.

    Derived classes will have specific semantics:

    * StateVariable: conditional storage
    * AttributeWithUnique: easy access to a set of unique values
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

    def __init__(self, name=None, doc=None, index=None):
        if index is None:
            CollectableAttribute._instance_index += 1
            index = CollectableAttribute._instance_index
        self._instance_index = index
        self.__doc__ = doc
        self.__name = name
        self._value = None
        self._isset = False
        self.reset()
        if __debug__:
            debug("COL",
                  "Initialized new collectable #%d:%s" % (index,name) + `self`)


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


    def _getName(self):
        return self.__name


    def _setName(self, name):
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
    name = property(_getName, _setName)



# XXX think that may be discard hasunique and just devise top
#     class DatasetAttribute
class AttributeWithUnique(CollectableAttribute):
    """Container which also takes care about recomputing unique values

    XXX may be we could better link original attribute to additional
    attribute which actually stores the values (and do reverse there
    as well).

    Pros:
      * don't need to mess with getattr since it would become just another
        attribute

    Cons:
      * might be worse design in terms of comprehension
      * take care about _set, since we shouldn't allow
        change it externally

    For now lets do it within a single class and tune up getattr
    """

    def __init__(self, name=None, hasunique=True, doc="Attribute with unique"):
        CollectableAttribute.__init__(self, name, doc)
        self._hasunique = hasunique
        self._resetUnique()
        if __debug__:
            debug("UATTR",
                  "Initialized new AttributeWithUnique %s " % name + `self`)


    def reset(self):
        super(AttributeWithUnique, self).reset()
        self._resetUnique()


    def _resetUnique(self):
        self._uniqueValues = None


    def _set(self, *args, **kwargs):
        self._resetUnique()
        CollectableAttribute._set(self, *args, **kwargs)


    def _getUniqueValues(self):
        if self.value is None:
            return None
        if self._uniqueValues is None:
            # XXX we might better use Set, but yoh recalls that
            #     N.unique was more efficient. May be we should check
            #     on the the class and use Set only if we are not
            #     dealing with ndarray (or lists/tuples)
            self._uniqueValues = N.unique(N.asanyarray(self.value))
        return self._uniqueValues

    uniqueValues = property(fget=_getUniqueValues)
    hasunique = property(fget=lambda self:self._hasunique)



# Hooks for comprehendable semantics and automatic collection generation
class SampleAttribute(AttributeWithUnique):
    pass



class FeatureAttribute(AttributeWithUnique):
    pass



class DatasetAttribute(AttributeWithUnique):
    pass



class StateVariable(CollectableAttribute):
    """Simple container intended to conditionally store the value
    """

    def __init__(self, name=None, enabled=True, doc="State variable"):
        CollectableAttribute.__init__(self, name, doc)
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
