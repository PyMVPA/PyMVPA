#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Classes to control and store state information.

It was devised to provide conditional storage 
"""

__docformat__ = 'restructuredtext'

import operator, copy
from sets import Set
from textwrap import TextWrapper

from mvpa.misc.vproperty import VProperty
from mvpa.misc.exceptions import UnknownStateError
from mvpa.base import warning
from mvpa.base.dochelpers import enhancedClassDocString, enhancedDocString

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

    def __init__(self, name=None, doc=None):
        self.__doc__ = doc
        self.__name = name
        self._value = None
        self._isset = False
        self.reset()
        if __debug__:
            debug("COL",
                  "Initialized new collectable %s " % name + `self`)


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


###################################################################
# Collections
#
# TODO: refactor into attributes.py and collections.py. state.py now has
#       little in common with the main part of this file
#
class Collection(object):
    """Container of some CollectableAttributes.

    :Groups:
     - `Public Access Functions`: `isKnown`
     - `Access Implementors`: `_getListing`, `_getNames`
     - `Mutators`: `__init__`
     - `R/O Properties`: `listing`, `names`, `items`

     XXX Seems to be not used and duplicating functionality: `_getListing`
     (thus `listing` property)
    """

    def __init__(self, items=None, owner=None):
        """Initialize the Collection

        :Parameters:
          items : dict of CollectableAttribute's
            items to initialize with
          owner : object
            an object to which collection belongs
        """

        self.__owner = owner

        if items == None:
            items = {}
        self._items = items
        """Dictionary to contain registered states as keys and
        values signal either they are enabled
        """


    def __str__(self):
        num = len(self._items)
        if __debug__ and "ST" in debug.active:
            maxnumber = 1000            # I guess all
        else:
            maxnumber = 4
        res = "{"
        for i in xrange(min(num, maxnumber)):
            if i>0: res += " "
            res += "%s" % str(self._items.values()[i])
        if len(self._items) > maxnumber:
            res += "..."
        res += "}"
        if __debug__:
            if "ST" in debug.active:
                res += " owner:%s#%s" % (self.owner.__class__.__name__,
                                         id(self.owner))
        return res


    def _cls_repr(self):
        """Collection specific part of __repr__ for a class containing
        it, ie a part of __repr__ for the owner object

        :Return:
          list of items to be appended within __repr__ after a .join()
        """
        # XXX For now we do not expect any pure non-specialized
        # collection , thus just override in derived classes
        raise NotImplementedError, "Class %s should override _cls_repr" \
              % self.__class__.__name__

    def _is_initializable(self, index):
        """Checks if index could be assigned within collection via
        _initialize

        :Return: bool value for a given `index`

        It is to facilitate dynamic assignment of collections' items
        within derived classes' __init__ depending on the present
        collections in the class.
        """
        # XXX Each collection has to provide what indexes it allows
        #     to be set within constructor. Custom handling of some
        #     arguments (like (dis|en)able_states) is to be performed
        #     in _initialize
        # raise NotImplementedError, \
        #      "Class %s should override _is_initializable" \
        #      % self.__class__.__name__

        # YYY lets just check if it is in the keys
        return index in self._items.keys()


    def _initialize(self, index, value):
        """Initialize `index` (no check performed) with `value`
        """
        # by default we just set corresponding value
        self.setvalue(index, value)


    def __repr__(self):
        s = "%s(" % self.__class__.__name__
        items_s = ""
        sep = ""
        for item in self._items:
            try:
                itemvalue = "%s" % `self._items[item].value`
                if len(itemvalue)>50:
                    itemvalue = itemvalue[:10] + '...' + itemvalue[-10:]
                items_s += "%s'%s':%s" % (sep, item, itemvalue)
                sep = ', '
            except:
                pass
        if items_s != "":
            s += "items={%s}" % items_s
        if self.owner is not None:
            s += "%sowner=%s" % (sep, `self.owner`)
        s += ")"
        return s


    #
    # XXX TODO: figure out if there is a way to define proper
    #           __copy__'s for a hierarchy of classes. Probably we had
    #           to define __getinitargs__, etc... read more...
    #
    #def __copy__(self):
# TODO Remove or refactor?
#    def _copy_states_(self, fromstate, deep=False):
#        """Copy known here states from `fromstate` object into current object
#
#        Crafted to overcome a problem mentioned above in the comment
#        and is to be called from __copy__ of derived classes
#
#        Probably sooner than later will get proper __getstate__,
#        __setstate__
#        """
#        # Bad check... doesn't generalize well...
#        # if not issubclass(fromstate.__class__, self.__class__):
#        #     raise ValueError, \
#        #           "Class  %s is not subclass of %s, " % \
#        #           (fromstate.__class__, self.__class__) + \
#        #           "thus not eligible for _copy_states_"
#        # TODO: FOR NOW NO TEST! But this beast needs to be fixed...
#        operation = { True: copy.deepcopy,
#                      False: copy.copy }[deep]
#
#        if isinstance(fromstate, Stateful):
#            fromstate = fromstate.states
#
#        self.enabled = fromstate.enabled
#        for name in self.names:
#            if fromstate.isKnown(name):
#                self._items[name] = operation(fromstate._items[name])


    def isKnown(self, index):
        """Returns `True` if state `index` is known at all"""
        return self._items.has_key(index)


    def __isSet1(self, index):
        """Returns `True` if state `index` has value set"""
        self._checkIndex(index)
        return self._items[index].isSet


    def isSet(self, index=None):
        """If item (or any in the present or listed) was set

        :Parameters:
          index : None or basestring or list of basestring
            What items to check if they were set in the collection
        """
        if not (index is None):
            if isinstance(index, basestring):
                 self._checkIndex(index) # process just that single index
                 return self._items[index].isSet
            else:
                items = index           # assume that we got some list
        else:
            items = self._items         # go through all the items

        for index in items:
            self._checkIndex(index)
            if self._items[index].isSet:
                return True
        return False


    def whichSet(self):
        """Return list of indexes which were set"""
        result = []
        # go through all members and if any isSet -- return True
        for index in self._items:
            if self.isSet(index):
                result.append(index)
        return result


    def _checkIndex(self, index):
        """Verify that given `index` is a known/registered state.

        :Raise `KeyError`: if given `index` is not known
        """
        if not self.isKnown(index):
            raise KeyError, \
                  "%s of %s has no key '%s' registered" \
                  % (self.__class__.__name__,
                     self.__owner.__class__.__name__,
                     index)


    def add(self, item):
        """Add a new CollectableAttribute to the collection

        :Parameters:
          item : CollectableAttribute
            or of derived class. Must have 'name' assigned

        TODO: we should make it stricter to don't add smth of
              wrong type into Collection since it might lead to problems

              Also we might convert to __setitem__
        """
        if not isinstance(item, CollectableAttribute):
            raise ValueError, \
                  "Collection can add only instances of " + \
                  "CollectableAttribute-derived classes. Got %s" % `item`
        if item.name is None:
            raise ValueError, \
                  "CollectableAttribute to be added %s must have 'name' set" % \
                  item
        self._items[item.name] = item

        if not self.owner is None:
            self._updateOwner(item.name)


    def remove(self, index):
        """Remove item from the collection
        """
        self._checkIndex(index)
        self._updateOwner(index, register=False)
        discard = self._items.pop(index)


    def __getattribute__(self, index):
        """
        """
        #return all private and protected ones first since we will not have
        # collectable's with _ (we should not have!)
        if index[0] == '_':
            return object.__getattribute__(self, index)
        if self._items.has_key(index):
            return self._items[index].value
        return object.__getattribute__(self, index)


    def __setattr__(self, index, value):
        if index[0] == '_':
            return object.__setattr__(self, index, value)
        if self._items.has_key(index):
            self._items[index].value = value
            return
        object.__setattr__(self, index, value)


    def __getitem__(self, index):
        if self._items.has_key(index):
            self._checkIndex(index)
            return self._items[index]
        else:
            raise AttributeError("State collection %s has no %s attribute" 
                                 % (self, index))


    # Probably not needed -- enable if need arises
    #
    #def __setattr__(self, index, value):
    #    if self._items.has_key(index):
    #        self._updateOwner(index, register=False)
    #        self._items[index] = value
    #        self._updateOwner(index, register=True)
    #
    #    object.__setattr__(self, index, value)


    def getvalue(self, index):
        """Returns the value by index"""
        self._checkIndex(index)
        return self._items[index].value


    def setvalue(self, index, value):
        """Sets the value by index"""
        self._checkIndex(index)
        self._items[index].value = value


    def _action(self, index, func, missingok=False, **kwargs):
        """Run specific func either on a single item or on all of them

        :Parameters:
          index : basestr
            Name of the state variable
          func
            Function (not bound) to call given an item, and **kwargs
          missingok : bool
            If True - do not complain about wrong index
        """
        if isinstance(index, basestring):
            if index.upper() == 'ALL':
                for index_ in self._items:
                    self._action(index_, func, missingok=missingok, **kwargs)
            else:
                try:
                    self._checkIndex(index)
                    func(self._items[index], **kwargs)
                except:
                    if missingok:
                        return
                    raise
        elif operator.isSequenceType(index):
            for item in index:
                self._action(item, func, missingok=missingok, **kwargs)
        else:
            raise ValueError, \
                  "Don't know how to handle  variable given by %s" % index


    def reset(self, index=None):
        """Reset the state variable defined by `index`"""

        if not index is None:
            indexes = [ index ]
        else:
            indexes = self.names

        if len(self.items):
            for index in indexes:
                # XXX Check if that works as desired
                self._action(index, self._items.values()[0].__class__.reset,
                             missingok=False)


    def _getListing(self):
        """Return a list of registered states along with the documentation"""

        # lets assure consistent litsting order
        items = self._items.items()
        items.sort()
        return [ "%s: %s" % (str(x[1]), x[1].__doc__) for x in items ]


    def _getNames(self):
        """Return ids for all registered state variables"""
        return self._items.keys()


    def _getOwner(self):
        return self.__owner


    def _setOwner(self, owner):
        if not isinstance(owner, Stateful):
            raise ValueError, \
                  "Owner of the StateCollection must be Stateful object"
        if __debug__:
            try:    strowner = str(owner)
            except: strowner = "UNDEF: <%s#%s>" % (owner.__class__, id(owner))
            debug("ST", "Setting owner for %s to be %s" % (self, strowner))
        if not self.__owner is None:
            # Remove attributes which were registered to that owner previousely
            self._updateOwner(register=False)
        self.__owner = owner
        if not self.__owner is None:
            self._updateOwner(register=True)


    def _updateOwner(self, index=None, register=True):
        """Define an entry within owner's __dict__
         so ipython could easily complete it

         :Parameters:
           index : basestring or list of basestring
             Name of the attribute. If None -- all known get registered
           register : bool
             Register if True or unregister if False

         XXX Needs refactoring since we duplicate the logic of expansion of
         index value
        """
        if not index is None:
            if not index in self._items:
                raise ValueError, \
                      "Attribute %s is not known to %s" % (index, self)
            indexes = [ index ]
        else:
            indexes = self.names

        ownerdict = self.owner.__dict__
        selfdict = self.__dict__

        for index_ in indexes:
            if register:
                if index_ in ownerdict:
                    raise RuntimeError, \
                          "Cannot register attribute %s within %s " % \
                          (index_, self.owner) + "since it has one already"
                ownerdict[index_] = self._items[index_]
                if index_ in selfdict:
                    raise RuntimeError, \
                          "Cannot register attribute %s within %s " % \
                          (index_, self) + "since it has one already"
                selfdict[index_] = self._items[index_]
            else:
                if index_ in ownerdict:
                    # yoh doesn't think that we need to complain if False
                    ownerdict.pop(index_)
                if index_ in selfdict:
                    selfdict.pop(index_)


    # Properties
    names = property(fget=_getNames)
    items = property(fget=lambda x:x._items)
    owner = property(fget=_getOwner, fset=_setOwner)

    # Virtual properties
    listing = VProperty(fget=_getListing)



class ParameterCollection(Collection):
    """Container of Parameters for a stateful object.
    """

#    def __init__(self, items=None, owner=None, name=None):
#        """Initialize the state variables of a derived class
#
#        :Parameters:
#          items : dict
#            dictionary of states
#        """
#        Collection.__init__(self, items, owner, name)
#

    def _cls_repr(self):
        """Part of __repr__ for the owner object
        """
        prefixes = []
        for k in self.names:
            # list only params with not default values
            if self[k].isDefault: continue
            prefixes.append("%s=%s" % (k, self[k].value))
        return prefixes


    def resetvalue(self, index, missingok=False):
        """Reset all parameters to default values"""
        from param import Parameter
        self._action(index, Parameter.resetvalue, missingok=False)



class StateCollection(Collection):
    """Container of StateVariables for a stateful object.

    :Groups:
     - `Public Access Functions`: `isKnown`, `isEnabled`, `isActive`
     - `Access Implementors`: `_getListing`, `_getNames`, `_getEnabled`
     - `Mutators`: `__init__`, `enable`, `disable`, `_setEnabled`
     - `R/O Properties`: `listing`, `names`, `items`
     - `R/W Properties`: `enabled`
    """

    def __init__(self, items=None, owner=None):
        """Initialize the state variables of a derived class

        :Parameters:
          items : dict
            dictionary of states
          owner : Stateful
            object which owns the collection
          name : basestring
            literal description. Usually just attribute name for the
            collection, e.g. 'states'
        """
        Collection.__init__(self, items=items, owner=owner)

        self.__storedTemporarily = []
        """List to contain sets of enabled states which were enabled
        temporarily.
        """

    #
    # XXX TODO: figure out if there is a way to define proper
    #           __copy__'s for a hierarchy of classes. Probably we had
    #           to define __getinitargs__, etc... read more...
    #
    #def __copy__(self):

    def _cls_repr(self):
        """Part of __repr__ for the owner object
        """
        prefixes = []
        for name, invert in ( ('enable', False), ('disable', True) ):
            states = self._getEnabled(nondefault=False,
                                      invert=invert)
            if len(states):
                prefixes.append("%s_states=%s" % (name, str(states)))
        return prefixes


    def _is_initializable(self, index):
        """Checks if index could be assigned within collection via
        setvalue
        """
        return index in ['enable_states', 'disable_states']


    def _initialize(self, index, value):
        if value is None:
            value = []
        if index == 'enable_states':
            self.enable(value, missingok=True)
        elif index == 'disable_states':
            self.disable(value)
        else:
            raise ValueError, "StateCollection can accept only enable_states " \
                  "and disable_states arguments for the initialization. " \
                  "Got %s" % index


    def _copy_states_(self, fromstate, deep=False):
        """Copy known here states from `fromstate` object into current object

        Crafted to overcome a problem mentioned above in the comment
        and is to be called from __copy__ of derived classes

        Probably sooner than later will get proper __getstate__,
        __setstate__
        """
        # Bad check... doesn't generalize well...
        # if not issubclass(fromstate.__class__, self.__class__):
        #     raise ValueError, \
        #           "Class  %s is not subclass of %s, " % \
        #           (fromstate.__class__, self.__class__) + \
        #           "thus not eligible for _copy_states_"
        # TODO: FOR NOW NO TEST! But this beast needs to be fixed...
        operation = { True: copy.deepcopy,
                      False: copy.copy }[deep]

        if isinstance(fromstate, Stateful):
            fromstate = fromstate.states

        self.enabled = fromstate.enabled
        for name in self.names:
            if fromstate.isKnown(name):
                self._items[name] = operation(fromstate._items[name])


    def isEnabled(self, index):
        """Returns `True` if state `index` is enabled"""
        self._checkIndex(index)
        return self._items[index].isEnabled


    def isActive(self, index):
        """Returns `True` if state `index` is known and is enabled"""
        return self.isKnown(index) and self.isEnabled(index)


    def enable(self, index, value=True, missingok=False):
        """Enable  state variable given in `index`"""
        self._action(index, StateVariable.enable, missingok=missingok,
                     value=value)


    def disable(self, index):
        """Disable state variable defined by `index` id"""
        self._action(index, StateVariable.enable, missingok=False, value=False)


    # TODO XXX think about some more generic way to grab temporary
    # snapshot of CollectableAttributes to be restored later on...
    def _changeTemporarily(self, enable_states=None,
                           disable_states=None, other=None):
        """Temporarily enable/disable needed states for computation

        Enable or disable states which are enabled in `other` and listed in
        `enable _states`. Use `resetEnabledTemporarily` to reset
        to previous state of enabled.

        `other` can be a Stateful object or StateCollection
        """
        if enable_states == None:
            enable_states = []
        if disable_states == None:
            disable_states = []
        self.__storedTemporarily.append(self.enabled)
        other_ = other
        if isinstance(other, Stateful):
            other = other.states

        if not other is None:
            # lets take states which are enabled in other but not in
            # self
            add_enable_states = list(Set(other.enabled).difference(
                 Set(enable_states)).intersection(self.names))
            if len(add_enable_states)>0:
                if __debug__:
                    debug("ST",
                          "Adding states %s from %s to be enabled temporarily" %
                          (add_enable_states, other_) +
                          " since they are not enabled in %s" %
                          (self))
                enable_states += add_enable_states

        # Lets go one by one enabling only disabled once... but could be as
        # simple as
        self.enable(enable_states)
        self.disable(disable_states)


    def _resetEnabledTemporarily(self):
        """Reset to previousely stored set of enabled states"""
        if __debug__:
            debug("ST", "Resetting to previous set of enabled states")
        if len(self.enabled)>0:
            self.enabled = self.__storedTemporarily.pop()
        else:
            raise ValueError("Trying to restore not-stored list of enabled " \
                             "states")


    def _getEnabled(self, nondefault=True, invert=False):
        """Return list of enabled states

        :Parameters:
          nondefault : bool
            Either to return also states which are enabled simply by default
          invert : bool
            Would invert the meaning, ie would return disabled states
        """
        if invert:
            fmatch = lambda y: not self.isEnabled(y)
        else:
            fmatch = lambda y: self.isEnabled(y)

        if nondefault:
            ffunc = fmatch
        else:
            ffunc = lambda y: fmatch(y) and \
                        self._items[y]._defaultenabled != self.isEnabled(y)
        return filter(ffunc, self.names)


    def _setEnabled(self, indexlist):
        """Given `indexlist` make only those in the list enabled

        It might be handy to store set of enabled states and then to restore
        it later on. It can be easily accomplished now::

        >>> states_enabled = stateful.enabled
        >>> stateful.enabled = ['blah']
        >>> stateful.enabled = states_enabled

        """
        for index in self._items.keys():
            self.enable(index, index in indexlist)


    # Properties
    enabled = property(fget=_getEnabled, fset=_setEnabled)


##################################################################
# Base classes (and metaclass) which use collections
#


#
# Helper dictionaries for AttributesCollector
#
_known_collections = {
    'StateVariable': ("states", StateCollection),
    'Parameter': ("params", ParameterCollection),
    'KernelParameter': ("kernel_params", ParameterCollection)}


_col2class = dict(_known_collections.values())
"""Mapping from collection name into Collection class"""


_COLLECTIONS_ORDER = ['params', 'kernel_params', 'states']


class AttributesCollector(type):
    """Intended to collect and compose StateCollection for any child
    class of this metaclass
    """


    def __init__(cls, name, bases, dict):

        if __debug__:
            debug("COLR",
                  "AttributesCollector call for %s.%s, where bases=%s, dict=%s " \
                  % (cls, name, bases, dict))

        super(AttributesCollector, cls).__init__(name, bases, dict)

        collections = {}
        for name, value in dict.iteritems():
            if isinstance(value, CollectableAttribute):
                baseclassname = value.__class__.__name__
                col = _known_collections[baseclassname][0]
                # XXX should we allow to throw exceptions here?
                if not collections.has_key(col):
                    collections[col] = {}
                collections[col][name] = value
                # and assign name if not yet was set
                if value.name is None:
                    value.name = name

        # XXX can we first collect parent's states and then populate with ours?
        # TODO

        for base in bases:
            if hasattr(base, "__metaclass__") and \
                   base.__metaclass__ == AttributesCollector:
                # TODO take care about overriding one from super class
                # for state in base.states:
                #    if state[0] =
                newcollections = base._collections_template
                if len(newcollections) == 0:
                    continue
                if __debug__:
                    debug("COLR",
                          "Collect collections %s for %s from %s" %
                          (newcollections, cls, base))
                for col, collection in newcollections.iteritems():
                    newitems = collection.items
                    if collections.has_key(col):
                        collections[col].update(newitems)
                    else:
                        collections[col] = newitems


        if __debug__:
            debug("COLR",
                  "Creating StateCollection template %s" % cls)

        # if there is an explicit
        if hasattr(cls, "_ATTRIBUTE_COLLECTIONS"):
            for col in cls._ATTRIBUTE_COLLECTIONS:
                if not col in _col2class:
                    raise ValueError, \
                          "Requested collection %s is unknown to collector" % \
                          col
                if not col in collections:
                    collections[col] = None

        # TODO: check on conflict in names of Collections' items!  since
        # otherwise even order is not definite since we use dict for
        # collections.
        # XXX should we switch to tuple?

        for col, colitems in collections.iteritems():
            collections[col] = _col2class[col](colitems)

        setattr(cls, "_collections_template", collections)

        #
        # Expand documentation for the class based on the listed
        # parameters an if it is stateful
        #
        # TODO -- figure nice way on how to alter __init__ doc directly...
        paramsdoc = ""
        textwrapper = TextWrapper(subsequent_indent="    ",
                                  initial_indent="    ",
                                  width=70)

        paramscols = []
        for col in ('params', 'kernel_params'):
            if collections.has_key(col):
                paramscols.append(col)
                for param, parameter in collections[col].items.iteritems():
                    paramsdoc += "  %s" % param
                    try:
                        paramsdoc += " : %s" % parameter.allowedtype
                    except:
                        pass
                    paramsdoc += "\n"
                    try:
                        doc = parameter.__doc__
                        try:
                            doc += " (Default: %s)" % parameter.default
                        except:
                            pass
                        paramsdoc += '\n'.join(textwrapper.wrap(doc))+'\n'
                    except Exception, e:
                        pass

        setattr(cls, "_paramscols", paramscols)
        if paramsdoc != "":
            if __debug__:
                debug("COLR", "Assigning __paramsdoc to be %s" % paramsdoc)
            setattr(cls, "_paramsdoc", paramsdoc)

            cls.__doc__ = enhancedClassDocString(cls, *bases)



class ClassWithCollections(object):
    """Base class for objects which contain any known collection

    Classes inherited from this class gain ability to access
    collections and their items as simple attributes. Access to
    collection items "internals" is done via <collection_name> attribute
    and interface of a corresponding `Collection`.

    TODO: rename 'descr'? -- it should simply
          be 'doc' -- no need to drag classes docstring imho.
    """

    __metaclass__ = AttributesCollector

    def __init__(self, descr=None, **kwargs):
        """Initialize ClassWithCollections object

        :Parameters:
          descr : basestring
            Description of the instance
        """
        if not hasattr(self, '_collections'):
            # need to check to avoid override of enabled states in the case
            # of multiple inheritance, like both Statefull and Harvestable
            object.__setattr__(
                self, '_collections',
                copy.deepcopy(
                    object.__getattribute__(self, '_collections_template')))

            collections = self._collections

            # Assign owner to all collections
            for col, collection in collections.iteritems():
                if col in self.__dict__:
                    raise ValueError, \
                          "Object %s has already attribute %s" % \
                          (self, col)
                self.__dict__[col] = collection
                collection.owner = self

            self.__descr = descr

            # Assign attributes values if they are given among
            # **kwargs
            for arg, argument in kwargs.items():
                set = False
                for collection in collections.itervalues():
                    if collection._is_initializable(arg):
                        collection._initialize(arg, argument)
                        set = True
                        break
                if set:
                    trash = kwargs.pop(arg)
                else:
                    known_params = reduce(
                       lambda x,y:x+y, [x.items.keys() for x in collections.itervalues()], [])
                    raise TypeError, \
                          "Unexpected keyword argument %s=%s for %s." \
                           % (arg, argument, self) \
                          + " Valid parameters are %s" % known_params

            ## Initialize other base classes
            ##  commented out since it seems to be of no use for now
            #if init_classes is not None:
            #    # return back stateful arguments since they might be
            #    # processed by underlying classes
            #    kwargs.update(kwargs_stateful)
            #    for cls in init_classes:
            #        cls.__init__(self, **kwargs)
            #else:
            #    if len(kwargs)>0:
            #        known_params = reduce(lambda x, y: x + y, \
            #                            [x.items.keys() for x in collections], [])
            #        raise TypeError, \
            #              "Unknown parameters %s for %s." % (kwargs.keys(), self) \
            #              + " Valid parameters are %s" % known_params


        if __debug__:
            debug("COL", "ClassWithCollections.__init__ was done "
                  "for %s id %s with descr=%s" \
                  % (self.__class__.__name__, id(self), descr))


    #__doc__ = enhancedDocString('Stateful', locals())


    def __getattribute__(self, index):
        # return all private ones first since smth like __dict__ might be
        # queried by copy before instance is __init__ed
        if index[0] == '_':
            return object.__getattribute__(self, index)
        for col, collection in \
          object.__getattribute__(self, '_collections').iteritems():
            if index in [col]:
                return collection
            if collection.items.has_key(index):
                return collection.getvalue(index)
        return object.__getattribute__(self, index)


    def __setattr__(self, index, value):
        if index[0] == '_':
            return object.__setattr__(self, index, value)
        for col, collection in \
          object.__getattribute__(self, '_collections').iteritems():
            if collection.items.has_key(index):
                collection.setvalue(index, value)
                return
        object.__setattr__(self, index, value)


    # XXX not sure if we shouldn't implement anything else...
    def reset(self):
        for collection in self._collections.values():
            collection.reset()


    def __str__(self):
        s = "%s:" % (self.__class__.__name__)
        if self.__descr is not None:
            s += "/%s " % self.__descr
        if hasattr(self, "_collections"):
            for col, collection in self._collections.iteritems():
                s += " %d %s:%s" %(len(collection.items), col, str(collection))
        return s


    def __repr__(self, prefixes=[], fullname=False):
        """String definition of the object of ClassWithCollections object

        :Parameters:
          fullname : bool
            Either to include full name of the module
          prefixes : list of strings
            What other prefixes to prepend to list of arguments
        """
        prefixes = prefixes[:]          # copy list
        id_str = ""
        module_str = ""
        if __debug__:
            if 'MODULE_IN_REPR' in debug.active:
                fullname = True
            if 'ID_IN_REPR' in debug.active:
                id_str = '#%s' % id(self)

        if fullname:
            modulename = '%s' % self.__class__.__module__
            if modulename != "__main__":
                module_str = "%s." % modulename

        # Collections' attributes
        collections = self._collections
        # we want them in this particular order
        for col in _COLLECTIONS_ORDER:
            collection = collections.get(col, None)
            if collection is None:
                continue
            prefixes += collection._cls_repr()

        # Description if present
        descr = self.__descr
        if descr is not None:
            prefixes.append("descr='%s'" % (descr))

        return "%s%s(%s)%s" % (module_str, self.__class__.__name__,
                               ', '.join(prefixes), id_str)


    descr = property(lambda self: self.__descr,
                     doc="Description of the object if any")



# No actual separation is needed now between ClassWithCollections
# and a specific usecase.
Stateful = ClassWithCollections
Parametrized = ClassWithCollections



class Harvestable(Stateful):
    """Classes inherited from this class intend to collect attributes
    within internal processing.

    Subclassing Harvestable we gain ability to collect any internal
    data from the processing which is especially important if an
    object performs something in loop and discards some intermidiate
    possibly interesting results (like in case of
    CrossValidatedTransferError and states of the trained classifier
    or TransferError).

    """

    harvested = StateVariable(enabled=False, doc=
       """Store specified attributes of classifiers at each split""")

    _KNOWN_COPY_METHODS = [ None, 'copy', 'deepcopy' ]


    def __init__(self, attribs=None, copy_attribs='copy', **kwargs):
        """Initialize state of harvestable

        :Parameters:
            attribs : list of basestr or dicts
                What attributes of call to store and return within
                harvested state variable. If an item is a dictionary,
                following keys are used ['name', 'copy']
            copy_attribs : None or basestr
                Default copying. If None -- no copying, 'copy'
                - shallow copying, 'deepcopy' -- deepcopying

        """
        Stateful.__init__(self, **kwargs)

        self.__atribs = attribs
        self.__copy_attribs = copy_attribs

        self._setAttribs(attribs)


    def _setAttribs(self, attribs):
        """Set attributes to harvest

        Each attribute in self.__attribs must have following fields
         - name : functional (or arbitrary if 'obj' or 'attr' is set)
                  description of the thing to harvest,
                  e.g. 'transerror.clf.training_time'
         - obj : name of the object to harvest from (if empty,
                 'self' is assumed),
                 e.g 'transerror'
         - attr : attribute of 'obj' to harvest,
                 e.g. 'clf.training_time'
         - copy : None, 'copy' or 'deepcopy' - way to copy attribute
        """
        if attribs:
            # force the state
            self.states.enable('harvested')
            self.__attribs = []
            for i, attrib in enumerate(attribs):
                if isinstance(attrib, dict):
                    if not 'name' in attrib:
                        raise ValueError, \
                              "Harvestable: attribute must be a string or " + \
                              "a dictionary with 'name'"
                else:
                    attrib = {'name': attrib}

                # assign default method to copy
                if not 'copy' in attrib:
                    attrib['copy'] = self.__copy_attribs

                # check copy method
                if not attrib['copy'] in self._KNOWN_COPY_METHODS:
                    raise ValueError, "Unknown method %s. Known are %s" % \
                          (attrib['copy'], self._KNOWN_COPY_METHODS)

                if not ('obj' in attrib or 'attr' in attrib):
                    # Process the item to harvest
                    # split into obj, attr. If obj is empty, then assume self
                    split = attrib['name'].split('.', 1)
                    if len(split)==1:
                        obj, attr = split[0], None
                    else:
                        obj, attr = split
                    attrib.update({'obj':obj, 'attr':attr})

                if attrib['obj'] == '':
                    attrib['obj'] = 'self'

                # TODO: may be enabling of the states??

                self.__attribs.append(attrib)     # place value back
        else:
            # just to make sure it is not None or 0
            self.__attribs = []


    def _harvest(self, vars):
        """The harvesting function: must obtain dictionary of variables
        from the caller.

        :Parameters:
            vars : dict
                Dictionary of available data. Most often locals() could be
                passed as `vars`. Mention that desired to be harvested
                private attributes better be bound locally to some variable

        :Returns:
            nothing
        """

        if not self.states.isEnabled('harvested') or len(self.__attribs)==0:
            return

        if not self.states.isSet('harvested'):
            self.harvested = dict([(a['name'], []) for a in self.__attribs])

        for attrib in self.__attribs:
            attrv = vars[attrib['obj']]

            # access particular attribute if needed
            if not attrib['attr'] is None:
                attrv = eval('attrv.%s' % attrib['attr'])

            # copy the value if needed
            attrv = {'copy':copy.copy,
                     'deepcopy':copy.deepcopy,
                     None:lambda x:x}[attrib['copy']](attrv)

            self.harvested[attrib['name']].append(attrv)


    harvest_attribs = property(fget=lambda self:self.__attribs,
                               fset=_setAttribs)



