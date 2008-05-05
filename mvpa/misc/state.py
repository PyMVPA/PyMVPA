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

from mvpa.misc.vproperty import VProperty
from mvpa.misc.exceptions import UnknownStateError
from mvpa.misc import warning

if __debug__:
    from mvpa.misc import debug

class CollectableAttribute(object):
    """Base class for any custom behaving attribute intended to become part of a collection

    Derived classes will have specific semantics:
    * StateVariable: conditional storage
    * Parameter: attribute with validity ranges.
     - ClassifierParameter: specialization to become a part of
       Classifier's params collection
     - KernelParameter: --//-- to become a part of Kernel Classifier's
       kernel_params collection

    Those CollectableAttributes are to be groupped into corresponding groups for each class
    by statecollector metaclass
    """

    def __init__(self, name=None, doc=None):
        self.__doc__ = doc
        self.__name = name
        self._value = None
        self.reset()
        if __debug__:
            debug("COL",
                  "Initialized new collectable %s " % name + `self`)

    # Instead of going for VProperty lets make use of virtual method
    def _getVirtual(self): return self._get()
    def _setVirtual(self, value): return self._set(value)

    def _get(self):
        return self._value

    def _set(self, val):
        if __debug__:
            debug("COL",
                  "Setting %s to %s " % (str(self), val))
        self._value = val
        self._isset = True

    @property
    def isSet(self):
        return self._isset

    def reset(self):
        """Simply reset the flag"""
        if __debug__:
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
                if name.startswith('_'):
                    raise ValueError, \
                          "Collectable attribute name must not start with _. Got %s" % name
            else:
                raise ValueError, \
                      "Collectable attribute name must be a string. Got %s" % `name`
        self.__name = name

    # XXX should become vproperty?
    value = property(_getVirtual, _setVirtual)
    name = property(_getName, _setName)


class StateVariable(CollectableAttribute):
    """Simple container intended to conditionally store the value

    Statefull class provides easy interfact to access the variable
    (simply through an attribute), or modifying internal state
    (enable/disable) via .states attribute of type StateCollection.
    """

    def __init__(self, name=None, enabled=True, doc="State variable"):
        CollectableAttribute.__init__(self, name, doc)
        self._isenabled = enabled
        if __debug__:
            debug("STV",
                  "Initialized new state variable %s " % name + `self`)

    def _get(self):
        if not self.isSet:
            raise UnknownStateError("Unknown yet value of %s" % (self.name))
        return CollectableAttribute._get(self)

    def _set(self, val):
        if __debug__:
            debug("STV",
                  "Setting %s to %s " % (str(self), val))

        if self.isEnabled:
            # XXX may be should have left simple assignment
            # self._value = val
            CollectableAttribute._set(self, val)

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


class Collection(object):
    """Container of some CollectableAttributes.

    :Groups:
     - `Public Access Functions`: `isKnown`
     - `Access Implementors`: `_getListing`, `_getNames`
     - `Mutators`: `__init__`
     - `R/O Properties`: `listing`, `names`, `items`

     XXX Seems to be not used and duplicating functionality: `_getListing` (thus `listing` property)
    """

    def __init__(self, items=None, owner=None):
        """Initialize the Collection

        :Parameters:
          items : dict of CollectableAttribute's
            items to initialize with
          enable_states : list
            list of states to enable. If it contains 'all' (in any casing),
            then all states (besides the ones in disable_states) will be enabled
          disable_states : list
            list of states to disable
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
        res = "{"
        for i in xrange(min(num, 4)):
            if i>0: res += " "
            res += "%s" % str(self._items.values()[i])
        if len(self._items) > 4:
            res += "..."
        res += "}"
        if __debug__:
            if "ST" in debug.active:
                res += " owner:%s" % `self.owner`
        return res

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

    def isSet(self, index):
        """Returns `True` if state `index` has value set"""
        self._checkIndex(index)
        return self._items[index].isSet

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
        if index.startswith('_'):
            return object.__getattribute__(self, index)
        if self._items.has_key(index):
            return self._items[index].value
        return object.__getattribute__(self, index)


    def __setattr__(self, index, value):
        if index.startswith('_'):
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
            raise AttributeError("State collection %s has no %s attribute" % (self, index))

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

         XXX Needs refactoring since we duplicate the logic of expansion of index value
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

    def resetvalue(self, index, missingok=False):
        """Reset all parameters to default values"""
        from param import Parameter
        self._action(index, Parameter.resetvalue, missingok=False)


    def isSet(self, index=None):
        if not index is None:
            return Collection.isSet(self, index)
        # go through all members and if any isSet -- return True
        for index in self._items:
            if Collection.isSet(self, index):
                return True
        return False


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


    def _getEnabled(self):
        """Return list of enabled states"""
        return filter(lambda y: self.isEnabled(y), self.names)


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


#
# Helper dictionaries for collector
#
_known_collections = {
    'StateVariable': ("states", StateCollection),
    'Parameter': ("params", ParameterCollection),
    'KernelParameter': ("kernel_params", ParameterCollection)}

_col2class = dict(_known_collections.values())
"""Mapping from collection name into Collection class"""


class collector(type):
    """Intended to collect and compose StateCollection for any child
    class of this metaclass
    """


    def __init__(cls, name, bases, dict):

        if __debug__:
            debug("COLR",
                  "Collector call for %s.%s, where bases=%s, dict=%s " \
                  % (cls, name, bases, dict))

        super(collector, cls).__init__(name, bases, dict)

        collections = {}

        for name, value in dict.iteritems():
            if isinstance(value, CollectableAttribute):
                baseclassname = value.__class__.__name__
                colname = _known_collections[baseclassname][0]
                # XXX should we allow to throw exceptions here?
                if not collections.has_key(colname):
                    collections[colname] = {}
                collections[colname][name] = value
                # and assign name if not yet was set
                if value.name is None:
                    value.name = name

        # XXX can we first collect parent's states and then populate with ours? TODO

        for base in bases:
            if hasattr(base, "__metaclass__") and \
                   base.__metaclass__ == collector:
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
                for colname, collection in newcollections.iteritems():
                    newitems = collection.items
                    if collections.has_key(colname):
                        collections[colname].update(newitems)
                    else:
                        collections[colname] = newitems


        if __debug__:
            debug("COLR",
                  "Creating StateCollection template %s" % cls)

        # if there is an explicit
        if hasattr(cls, "_ATTRIBUTE_COLLECTIONS"):
            for colname in cls._ATTRIBUTE_COLLECTIONS:
                if not colname in _col2class:
                    raise ValueError, \
                          "Requested collection %s is unknown to collector" % \
                          colname
                if not colname in collections:
                    collections[colname] = None

        # TODO: check on conflict in names of Collections' items!
        # since otherwise even order is not definite since we use dict for collections.
        # XXX should we switch to tuple?

        for colname, colitems in collections.iteritems():
            collections[colname] = _col2class[colname](colitems)

        setattr(cls, "_collections_template", collections)


class Stateful(object):
    """Base class for stateful objects.

    Classes inherited from this class gain ability to provide state
    variables, accessed as simple properties. Access to state variables
    "internals" is done via states property and interface of
    `StateCollection`.

    NB This one is to replace old State base class
    TODO: fix drunk Yarik decision to add 'descr' -- it should simply
    be 'doc' -- no need to drag classes docstring imho.
    """

    __metaclass__ = collector

    def __init__(self,
                 enable_states=None,
                 disable_states=None,
                 descr=None):

        if not hasattr(self, '_collections'):
            # need to check to avoid override of enabled states in the case
            # of multiple inheritance, like both Statefull and Harvestable
            object.__setattr__(self, '_collections',
                               copy.deepcopy( \
                                object.__getattribute__(self,
                                                        '_collections_template')))

            # Assign owner to all collections
            for colname, colvalue in self._collections.iteritems():
                if colname in self.__dict__:
                    raise ValueError, \
                          "Stateful object %s has already attribute %s" % \
                          (self, colname)
                self.__dict__[colname] = colvalue
                colvalue.owner = self

            if self._collections.has_key('states'):
                if enable_states == None:
                    enable_states = []
                if disable_states == None:
                    disable_states = []

                states = self._collections['states']
                states.enable(enable_states, missingok=True)
                states.disable(disable_states)
            elif not (enable_states is None and disable_states is None):
                warning("Provided enable_states and disable_states are " + \
                        "ignored since object %s has no states"  % `self`)

            self.__descr = descr

        if __debug__:
            debug("ST", "Stateful.__init__ was done for %s id %s with descr=%s" \
                % (self.__class__, id(self), descr))


    def __getattribute__(self, index):
        # return all private ones first since smth like __dict__ might be
        # queried by copy before instance is __init__ed
        if index.startswith('_'):
            return object.__getattribute__(self, index)
        for colname, colvalues in object.__getattribute__(self, '_collections').iteritems():
            if index in [colname]:
                return colvalues
            if colvalues.items.has_key(index):
                return colvalues.getvalue(index)
        return object.__getattribute__(self, index)

    def __setattr__(self, index, value):
        if index.startswith('_'):
            return object.__setattr__(self, index, value)
        for colname, colvalues in object.__getattribute__(self, '_collections').iteritems():
            if colvalues.items.has_key(index):
                colvalues.setvalue(index, value)
                return
        object.__setattr__(self, index, value)

    # XXX not sure if we shouldn't implement anything else...
    def reset(self):
        for collection in self._collections.values():
            collection.reset()

    def __str__(self):
        s = "%s:" % (self.__class__.__name__)
        if hasattr(self, "_collections"):
            for colname,colvalues in self._collections.iteritems():
                s += " %d %s:%s" %(len(colvalues.items), colname, str(colvalues))
        return s

    def __repr__(self):
        return "<%s.%s#%d>" % (self.__class__.__module__, self.__class__.__name__, id(self))

    descr = property(lambda self: self.__descr,
                     doc="Description of the object if any")



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
        """The harvesting function: must obtain dictionary of variables from the caller.

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
