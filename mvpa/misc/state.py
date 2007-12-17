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

if __debug__:
    from mvpa.misc import debug


class State(object):
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
                 enable_states=None,
                 disable_states=None):
        """Initialize the state variables of a derived class

        :Parameters:
          enable_states : list
            list of states to enable. If it contains 'all' (in any casing),
            then all states (besides the ones in disable_states) will be enabled
          disable_states : list
            list of states to disable
          args : list
            as well as `kwargs` is passed to dict.__init__

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

        if isinstance(enable_states, basestring):
            self.__enable_all = enable_states.upper() == "ALL"
        elif operator.isSequenceType(enable_states):
            self.__enable_all = "ALL" in [x.upper() for x in enable_states]
        else:
            self.__enable_all = False

        if self.__enable_all:
            if __debug__:
                debug('ST',
                      'All states (besides explicitely disabled ' + \
                      'via disable_states) will be enabled')

        for key, enabled in register_states.iteritems():
            if (not enable_states is None):
                if not self.__enable_all and (not key in enable_states):
                    if __debug__:
                        debug('ST', 'Disabling state %s since it is not' \
                              'listed' % key + \
                              ' among explicitely enabled ones for %s' %
                              (self.__class__.__name__))
                    enabled = False
                else:
                    enabled = True

            if (not disable_states is None) and (key in disable_states):
                if __debug__:
                    debug('ST', 'Disabling state %s since it is listed' % key +
                          ' among explicitely disabled ones for %s' %
                          (self.__class__.__name__))
                enabled = False

            self._registerState(key, enabled)


    def __str__(self):
        num = len(self.__registered)
        res = "%s: %d state variables registered:" % (self.__class__.__name__,
                                                      num)
        for i in xrange(min(num, 4)):
            index = self.__registered.keys()[i]
            res += " %s" % index
            if self.hasState(index):
                res += '*'              # so we have the value already
            elif self.isStateEnabled:
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
        """
        if fromstate.__class__ != self.__class__:
            raise ValueError, \
                  "Different class got into %s._copy_states_: %s" % \
                  (self.__class__, into.__class__)

        operation = { True: copy.deepcopy,
                      False: copy.copy }[deep]

        self.enabledStates = operation(fromstate.enabledStates)
        self.__dict = operation(fromstate.__dict)
        self.__registered = operation(fromstate.__registered)
        self.__enable_states = operation(fromstate.__enable_states)
        self.__enable_all = operation(fromstate.__enable_all)


    def __checkIndex(self, index):
        """Verify that given `index` is a known/registered state.

        :Raise `KeyError`: if given `index` is not known
        """
        if not self.__registered.has_key(index):
            raise KeyError, \
                  "State of %s has no key %s registered" \
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
        if self.__enable_all or \
               ((not self.__enable_states is None) and \
                (index in self.__enable_states)):
            if enabled == False:
                enabled = True
                if __debug__:
                    debug("ST",
                          "State %s will be registered enabled" % index +
                          " since it was mentioned in enable_states")
        if __debug__:
            debug('ST', 'Registering %s state %s for %s' %
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
        if index.upper == 'ALL':
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
