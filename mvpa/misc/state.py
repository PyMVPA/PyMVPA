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

import operator, copy

from mvpa.misc.exceptions import UnknownStateError

from mvpa.misc import warning

if __debug__:
    from mvpa.misc import debug


class StateVariable(object):
    """Descriptor intended to conditionally store the value

    Unfortunately manipulation of enable is not straightforward and
    has to be done via class object, e.g.

      StateVariable.enable(self.__class__.values, self, False)

      if self is an instance of the class which has
    """

    def __init__(self, name=None, enabled=False, doc="State variable"):
        self._values = {}
        self._isenabled_default = enabled
        self._isenabled = {}
        self.__doc__ = doc
        self.name = name
        if __debug__:
            debug("STV",
                  "Initialized new state variable %s " % name + `self`)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not self._values.has_key(obj):
            raise UnknownStateError("Unknown yet value for '%s'" % `obj`)
        return self._values[obj]

    def __set__(self, obj, val):
        # print "! I am in set"
         if self.isEnabled(obj):
            self._values[obj] = val

    def isEnabled(self, obj):
        return (self._isenabled.has_key(obj) and self._isenabled[obj]) or \
                   (not self._isenabled.has_key(obj) and self._isenabled_default)

        return self._isenabled[obj]

    def enable(self, obj, val=False):
        if __debug__:
            debug("STV", "%s variable %s for %s" %
                  ({True: 'Enabling', False: 'Disabling'}[val], self.name, `obj`))
        self._isenabled[obj] = val

    def __delete__(self, obj):
        try:
            del self._values[key]
        except:
            raise AttributeError


class StateCollection(object):
    """Container of states class for stateful object.

    Classes inherited from this class gain ability to provide state
    variables, accessed via __getitem__ method (currently implemented
    by inherining `dict` class).
    XXX

    :Groups:
     - `Public Access Functions`: `isKnown`, `isEnabled`, `isActive`
     - `Access Implementors`: `_getListing`, `_getNames`, `_getEnabled`
     - `Mutators`: `__init__`, `enable`, `disable`, `_setEnabled`
     - `R/O Properties`: `listing`, `names`, `items`
     - `R/W Properties`: `enabled`
    """

    def __init__(self, ownercls, items = {}):
                 #enable_states=[],
                 #disable_states=[]):
        """Initialize the state variables of a derived class

        :Parameters:
          states : dict
            dictionary of states
          enable_states : list
            list of states to enable. If it contains 'all' (in any casing),
            then all states (besides the ones in disable_states) will be enabled
          disable_states : list
            list of states to disable
        """

        self.__ownercls = ownercls

        self.__items = items
        """Dictionary to contain registered states as keys and
        values signal either they are enabled
        """

        self.__storedTemporarily = []
        """List to contain sets of enabled states which were enabled temporarily"""

    def __str__(self):
        num = len(self.__items)
        res = "%d states:" % (num)
        for i in xrange(min(num, 4)):
            index = self.__items.keys()[i]
            res += " %s" % index
            if self.isEnabled(index):
                res += '+'              # it is enabled but no value is assigned yet
            if self.isKnown(index):
                res += '*'              # so we have the value already

        if len(self.__items) > 4:
            res += "..."
        return res

    #
    # XXX TODO: figure out if there is a way to define proper
    #           __copy__'s for a hierarchy of classes. Probably we had
    #           to define __getinitargs__, etc... read more...
    #
    #def __copy__(self):

    def _copy_states_(self, fromstate, deep=False):
        """Copy states from `fromstate` object into current object

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
        raise NotImplementedError
        self.enabled = operation(fromstate.enabled)
        self.__dict = operation(fromstate.__dict)
        self.__items = operation(fromstate.__items)
        self.__enable_states = operation(fromstate.__enable_states)
        self.__disable_states = operation(fromstate.__disable_states)
        self.__enable_all = operation(fromstate.__enable_all)
        self.__disable_all = operation(fromstate.__disable_all)
        self.__storedTemporarily = operation(fromstate.__storedTemporarily)


    def __checkIndex(self, index):
        """Verify that given `index` is a known/registered state.

        :Raise `KeyError`: if given `index` is not known
        """
        if not self.isKnown(index):
            raise KeyError, \
                  "State of %s has no key '%s' registered" \
                  % (self.__class__.__name__, index)


    def isKnown(self, index):
        """Returns `True` if state `index` is enabled"""
        return self.__items.has_key(index)

    def isEnabled(self, index):
        """Returns `True` if state `index` is enabled"""
        self.__checkIndex(index)
        return StateVariable.isEnabled(self.__items[index], self)

    def isActive(self, index):
        """Returns `True` if state `index` is known and is enabled"""
        return self.isKnown(index) and self.isEnabled(index)


    def enable(self, index, value=True):
        if isinstance(index, basestring):
            if index.upper() == 'ALL':
                for index_ in self.__items:
                    self.enable(index_, value)
            else:
                self.__checkIndex(index)
                StateVariable.enable(self.__items[index], self, value)
        elif operator.isSequenceType(index):
            for item in index:
                self.enable(item, value)
        else:
            raise ValueError, \
                  "Don't know how to handle state variable given by %s" % index

    def disable(self, index):
        """Disable state variable defined by `index` id"""
        self.enable(index, False)


    def _enableTemporarily(self, enable_states, other=None):
        """Temporarily enable needed states for computation

        Enable states which are enabled in `other` and listed in
        `enable _states`. Use `resetEnabledTemporarily` to reset
        to previous state of enabled.
        """
        self.__storedTemporarily.append(self.enabled)
        # Lets go one by one enabling only disabled once... but could be as simple as
        # self.enable(enable_states)
        for state in enable_states:
            if not self.isEnabled(state) and \
               ((other is None) or other.isEnabled(state)):
                if __debug__:
                    debug("ST", "Temporarily enabling state %s" % state)
                self.enableState(state)

    def _resetEnabledTemporarily(self):
        """Reset to previousely stored set of enabled states"""
        if __debug__:
            debug("ST", "Resetting to previous set of enabled states")
        if len(self.enabled)>0:
            self.enabled = self.__storedTemporarily.pop()
        else:
            raise ValueError("Trying to restore not-stored list of enabled states")


    def _getListing(self):
        """Return a list of registered states along with the documentation"""

        # lets assure consistent litsting order
        items = self.__items.items()
        items.sort()
        return [ "%s%s: %s" % (x[0],
                               {True:"[enabled]",
                                False:""}[self.isEnabled(x[0])],
                               x[1].__doc__) for x in items ]


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
        for index in self.__items.keys():
            self.enable(index, index in indexlist)


    def _getNames(self):
        """Return ids for all registered state variables"""
        return self.__items.keys()


    # Properties
    listing = property(fget=_getListing)
    names = property(fget=_getNames)
    items = property(fget=lambda x:x.__items)
    enabled = property(fget=_getEnabled, fset=_setEnabled)


class statecollector(type):
    """Intended to collect and compose StateCollection for any child
    class of this metaclass
    """

    def __init__(cls, name, bases, dict):

        if __debug__:
            debug("STCOL", "Collector call for %s.%s, where bases=%s, dict=%s " %
                  (cls, name, bases, dict))

        super(statecollector, cls).__init__(name, bases, dict)

        items = {}
        for name, value in dict.iteritems():
            if isinstance(value, StateVariable):
                items[name] = value
                # and assign name if not yet was set
                if value.name is None:
                    value.name = name

        for base in bases:
            if hasattr(base, "__metaclass__") and \
                   base.__metaclass__ == statecollector:
                # TODO take care about overriding one from super class
                # for state in base.states:
                #    if state[0] =
                newitems = base._states.items
                if len(newitems) == 0:
                    continue
                if __debug__:
                    debug("STCOL",
                          "Collect states %s for %s from %s" %
                          (newitems, cls, base))
                items.update(newitems)

        if __debug__:
            debug("STCOL",
                  "Creating StateCollection %s" % cls)

        statecollection = StateCollection(cls, items)
        setattr(cls, "_states", statecollection)


class Statefull(object):
    """Base class for stateful objects.

    Classes inherited from this class gain ability to provide state
    variables, accessed as simple properties. Access to state variables
    "internals" is done via states property and interface of
    `StateCollection`.

    NB This one is to replace old State base class
    """

    __metaclass__ = statecollector

    def __init__(self,
                 enable_states=[],
                 disable_states=[]):
        if __debug__:
            debug("ST", "Statefull.__init__ for %s" % self)
        self._states.enable(enable_states)
        self._states.disable(disable_states)

    @property
    def states(self):
        return self._states


    # TODO remove _registerState
    def _registerState(self, index, enabled=True, doc=None):
        if not hasattr(self.__class__, index):
            warning("!!!! deprecated: call to _registerState for %s" % index)
            setattr(self.__class__, index, StateVariable(enabled=enabled, doc=doc))


class OldState(object):
    """Base class for stateful objects.

    Classes inherited from this class gaining ability to provide state
    variables, accessed via __getitem__ method (currently implemented
    by inherining `dict` class).

    :Groups:
     - `Access Functions`: `isStateEnabled`, `listStates`, `_getEnabledStates`
     - `Mutators`: `__init__`, `enableState`, `enableStates`,
       `disableState`, `disableStates`, `_setEnabledStates`
    """

    _register_states = {}
    #
    # children classes can compactly describe state variables as a static
    # members instead of explicit calling _registerState
    # should not be used at the moment since deeper child simpler overrides
    # defined in some parent class

    def __init__(self,
                 enable_states=[],
                 disable_states=[]):
        """Initialize the state variables of a derived class

        :Parameters:
          enable_states : list
            list of states to enable. If it contains 'all' (in any casing),
            then all states (besides the ones in disable_states) will be enabled
          disable_states : list
            list of states to disable
        """

        self.__registered = {}
        """Dictionary to contain registered states as keys and
        values signal either they are enabled
        """
        self.__dict = {}
        """Actual storage to use for state variables"""

        register_states = {}
        # if class defined default states to register -- use them

        # XXX Yarik's attempt to overcome the problem that derived
        # child class would override parent's list of register_states.
        # Left as a comment for history
        # register_states_fields = \
        #   filter(lambda x: x.startswith('_register_states'),
        #          self.__class__.__dict__.keys())
        if self.__class__.__dict__.has_key('_register_states'):
            register_states = self.__class__._register_states
        # no need to compain actually since this method doesn't work
        # nicely (see above)
        #elif __debug__:
        #   debug('ST', 'Class %s is a child of State class but has no states' %
        #          (self.__class__.__name__))

        # store states to enable later on
        self.__enable_states = enable_states
        self.__disable_states = disable_states

        if isinstance(enable_states, basestring):
            enable_states = [ enable_states ]
        if isinstance(disable_states, basestring):
            disable_states = [ disable_states ]

        assert(operator.isSequenceType(enable_states))

        self.__enable_all = "ALL" in [x.upper() for x in enable_states]
        self.__disable_all = "ALL" in [x.upper() for x in disable_states]

        if self.__enable_all and self.__disable_all:
            raise ValueError,\
                  "Cannot enable and disable all states at the same time " + \
                  " in %s" % `self`

        if self.__enable_all:
            if __debug__:
                debug('ST',
                      'All states (besides explicitely disabled ' + \
                      'via disable_states) will be enabled')

        if self.__disable_all:
            if __debug__:
                debug('ST',
                      'All states will be disabled')

        for key, enabled in register_states.iteritems():
            self._registerState(key, enabled)

        self.__storedTemporarily = []
        """List to contain sets of enabled states which were enabled temporarily"""

    def __str__(self):
        num = len(self.__registered)
        #res = "%s: %d states:" % (self.__class__.__name__,
        res = "%d states:" % (num)
        for i in xrange(min(num, 4)):
            index = self.__registered.keys()[i]
            res += " %s" % index
            if self.hasState(index):
                res += '*'              # so we have the value already
            elif self.isStateEnabled(index):
                res += '+'              # it is enabled but no value is assigned yet

        if len(self.__registered) > 4:
            res += "..."
        return res

    #
    # XXX TODO: figure out if there is a way to define proper
    #           __copy__'s for a hierarchy of classes. Probably we had
    #           to define __getinitargs__, etc... read more...
    #
    #def __copy__(self):

    def _copy_states_(self, fromstate, deep=False):
        """Copy states from `fromstate` object into current object

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

        self.enabledStates = operation(fromstate.enabledStates)
        self.__dict = operation(fromstate.__dict)
        self.__registered = operation(fromstate.__registered)
        self.__enable_states = operation(fromstate.__enable_states)
        self.__disable_states = operation(fromstate.__disable_states)
        self.__enable_all = operation(fromstate.__enable_all)
        self.__disable_all = operation(fromstate.__disable_all)
        self.__storedTemporarily = operation(fromstate.__storedTemporarily)


    def __checkIndex(self, index):
        """Verify that given `index` is a known/registered state.

        :Raise `KeyError`: if given `index` is not known
        """
        if not self.__registered.has_key(index):
            raise KeyError, \
                  "State of %s has no key '%s' registered" \
                  % (self.__class__.__name__, index)


    # actually we have to provide some simple way to set the State
    def __setitem__(self, index, value):
        """Set value for the `index`.
        """
        self.__checkIndex(index)
        if self.__registered[index]['enabled']:
            self.__dict[index] = value


    def __getitem__(self, index):
        """Return a value for the given `index`

        :Raise `KeyError`: if given `index` is not known
        :Raise `UnknownStateError`: if no value yet was assigned to `index`
        """
        self.__checkIndex(index)
        if not self.__dict.has_key(index):
            raise UnknownStateError("Unknown yet value for '%s'" % index)
        return self.__dict[index]


    # XXX think about it -- may be it is worth making whole State
    # handling via static methods to remove any possible overhead of
    # registering the same keys in each constructor
    # Michael: Depends on what kind of objects we want to have a state.
    #          Anyway as we will inherent this class I think the method name
    #          should be a bit more distinctive. What about:
    def _registerState(self, index, enabled=True, doc=None):
        """Register a new state

        :Parameters:
          `index`
            the index
          `enabled` : Bool
            either the state should be enabled
          `doc` : string
            description for the state
        """
        # retrospectively enable state
        if not enabled:
            if self.__enable_all or (index in self.__enable_states):
                if not (index in self.__disable_states) and \
                       not self.__disable_all:
                    enabled = True
                    if __debug__:
                        debug("ST",
                              "State '%s' will be registered enabled" % index +
                              " since it was mentioned in enable_states")
        else:
            if (index in self.__disable_states) or (self.__disable_all):
                enabled = False
                if __debug__:
                    debug("ST",
                          "State '%s' will be registered disabled" % index +
                          " since it was mentioned in disable_states")

        if __debug__:
            debug('ST', "Registering %s state '%s' for %s" %
                  ({True:'enabled', False:'disabled'}[enabled],
                   index, self.__class__.__name__))
        self.__registered[index] = {'enabled' : enabled,
                                    'doc' : doc}

    def hasState(self, index):
        """Checks if there is a state for `index`

        Simple a wrapper around self.__dict.has_key
        """
        return self.__dict.has_key(index)


    def __enabledisableall(self, index, value):
        if index.upper() == 'ALL':
            if value:
                self.__enable_all = True
                self.__disable_all = False
            else:
                self.__disable_all = True
                self.__enable_all = False
            for index in self.states:
                self.__registered[index]['enabled'] = value
            return True
        else:
            return False

    def enableState(self, index):
        """Enable state variable defined by `index` id"""
        if not self.__enabledisableall(index, True):
            self.__checkIndex(index)
            self.__registered[index]['enabled'] = True


    def disableState(self, index):
        """Disable state variable defined by `index` id"""
        if not self.__enabledisableall(index, False):
            self.__checkIndex(index)
            self.__registered[index]['enabled'] = False


    def enableStates(self, indexlist):
        """Enable all states listed in `indexlist`"""

        for index in indexlist:
            self.enableState(index)


    def disableStates(self, indexlist):
        """Disable all states listed in `indexlist`"""

        for index in indexlist:
            self.disableState(index)


    def isStateEnabled(self, index):
        """Returns `True` if state `index` is enabled"""

        self.__checkIndex(index)
        return self.__registered[index]['enabled']


    def isStateActive(self, index):
        """Returns `True` if state `index` is known and is enabled"""
        return self.__registered.has_key(index) and \
               self.__registered[index]['enabled']


    def _enableStatesTemporarily(self, enable_states, other=None):
        """Temporarily enable needed states for computation

        Enable states which are enabled in `other` and listed in
        `enable _states`. Use `resetEnabledTemporarily` to reset
        to previous state of enabled.
        """
        self.__storedTemporarily.append(self.enabledStates)
        for state in enable_states:
            if not self.isStateEnabled(state) and \
               ((other is None) or other.isStateEnabled(state)):
                if __debug__:
                    debug("ST", "Temporarily enabling state %s" % state)
                self.enableState(state)


    def _resetEnabledTemporarily(self):
        """Reset to previousely stored set of enabled states"""
        if __debug__:
            debug("ST", "Resetting to previous set of enabled states")
        if len(self.enabledStates)>0:
            self.enabledStates = self.__storedTemporarily.pop()
        else:
            raise ValueError("Trying to restore not-stored list of enabled states")


    def listStates(self):
        """Return a list of registered states along with the documentation"""

        # lets assure consistent litsting order
        items = self.__registered.items()
        items.sort()
        return [ "%s%s: %s" % (x[0],
                               {True:"[enabled]",
                                False:""}[x[1]['enabled']],
                               x[1]['doc']) for x in items ]


    def _getEnabledStates(self):
        """Return list of enabled states"""

        return filter(lambda y:
                      self.__registered[y]['enabled'],
                      self.__registered.keys())


    def _setEnabledStates(self, indexlist):
        """Given `indexlist` make only those in the list enabled

        It might be handy to store set of enabled states and then to restore
        it later on. It can be easily accomplished now::

        >>> states_enabled = stateful.enabledStates
        >>> stateful.enabledState('blah')
        >>> stateful.enabledStates = states_enabled

        """
        for state in self.__registered.items():
            state[1]['enabled'] = state[0] in indexlist


    def _getRegisteredStates(self):
        """Return ids for all registered state variables"""

        return self.__registered.keys()


    # Properties
    states = property(fget=_getRegisteredStates)
    enabledStates = property(fget=_getEnabledStates, fset=_setEnabledStates)
