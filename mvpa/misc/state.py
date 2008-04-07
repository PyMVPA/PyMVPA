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

from mvpa.misc.exceptions import UnknownStateError

if __debug__:
    from mvpa.misc import debug

class Collectable(object):
    """Base class for any custom behaving attribute intended to become part of a collection

    Derived classes will have specific semantics:
    * StateVariable: conditional storage
    * Parameter: attribute with validity ranges.
     - ClassifierParameter: specialization to become a part of
       Classifier's params collection
     - KernelParameter: --//-- to become a part of Kernel Classifier's
       kernel_params collection

    Those Collectables are to be groupped into corresponding groups for each class
    by statecollector metaclass
    """

    def __init__(self, name=None, doc=""):
        self.__doc__ = doc
        self.name = name
        self._value = None
        if __debug__:
            debug("COL",
                  "Initialized new state variable %s " % name + `self`)

    def _get(self):
        return self._value

    def _set(self, val):
        if __debug__:
            debug("COL",
                  "Setting %s to %s " % (str(self), val))
        self._value = val

    def reset(self):
        """Simply detach the value, and reset the flag"""
        self._value = None

    def __str__(self):
        return "Collectable %s id %d" % \
            (self.name, id(self))

    # XXX should become vproperty?
    value = property(_get, _set)



class StateVariable(Collectable):
    """Simple container intended to conditionally store the value

    Statefull class provides easy interfact to access the variable
    (simply through an attribute), or modifying internal state
    (enable/disable) via .states attribute of type StateCollection.
    """

    def __init__(self, name=None, enabled=True, doc="State variable"):
        Collectable.__init__(self, name, doc)
        self._isset = False
        self._isenabled = enabled
        if __debug__:
            debug("STV",
                  "Initialized new state variable %s " % name + `self`)

    def _get(self):
        if not self.isSet:
            raise UnknownStateError("Unknown yet value of %s" % (self.name))
        # XXX leave simple return?
        #return self._value
        return Collectable._get(self)

    def _set(self, val):
        if __debug__:
            debug("STV",
                  "Setting %s to %s " % (str(self), val))

        if self.isEnabled:
            self._isset = True
            # XXX may be should have left simple assignment
            # self._value = val
            Collectable._set(self, val)

    @property
    def isSet(self):
        return self._isset

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


    def reset(self):
        """Simply detach the value, and reset the flag"""
        # XXX simpler?
        #self._value = None
        Collectable.reset(self)
        self._isset = False

    def __str__(self):
        return "%s variable %s id %d" % \
            ({True: 'Enabled',
              False: 'Disabled'}[self.isEnabled], self.name, id(self))

    value = property(_get, _set)


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

    def __init__(self, items=None, owner = None):
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

        self.__owner = owner

        if items == None:
            items = {}
        self.__items = items
        """Dictionary to contain registered states as keys and
        values signal either they are enabled
        """

        self.__storedTemporarily = []
        """List to contain sets of enabled states which were enabled
        temporarily.
        """

    def __str__(self):
        num = len(self.__items)
        res = "%d states:" % (num)
        for i in xrange(min(num, 4)):
            index = self.__items.keys()[i]
            res += " %s" % index
            if self.isEnabled(index):
                res += '+'          # it is enabled but no value is assigned yet
            if self.isSet(index):
                res += '*'          # so we have the value already

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
                self.__items[name] = operation(fromstate.__items[name])

    def isKnown(self, index):
        """Returns `True` if state `index` is known at all"""
        return self.__items.has_key(index)

    def __checkIndex(self, index):
        """Verify that given `index` is a known/registered state.

        :Raise `KeyError`: if given `index` is not known
        """
        if not self.isKnown(index):
            raise KeyError, \
                  "%s of %s has no key '%s' registered" \
                  % (self.__class__.__name__,
                     self.__owner.__class__.__name__,
                     index)


    def isEnabled(self, index):
        """Returns `True` if state `index` is enabled"""
        self.__checkIndex(index)
        return self.__items[index].isEnabled

    def isSet(self, index):
        """Returns `True` if state `index` has value set"""
        self.__checkIndex(index)
        return self.__items[index].isSet


    def get(self, index):
        """Returns the value by index"""
        self.__checkIndex(index)
        return self.__items[index].value

    def set(self, index, value):
        """Sets the value by index"""
        self.__checkIndex(index)
        self.__items[index].value = value


    def isActive(self, index):
        """Returns `True` if state `index` is known and is enabled"""
        return self.isKnown(index) and self.isEnabled(index)


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
                for index_ in self.__items:
                    self._action(index_, func, missingok=missingok, **kwargs)
            else:
                try:
                    self.__checkIndex(index)
                    func(self.__items[index], **kwargs)
                except:
                    if missingok:
                        return
                    raise
        elif operator.isSequenceType(index):
            for item in index:
                self._action(item, func, missingok=missingok, **kwargs)
        else:
            raise ValueError, \
                  "Don't know how to handle state variable given by %s" % index

    def enable(self, index, value=True, missingok=False):
        """Enable  state variable given in `index`"""
        self._action(index, StateVariable.enable, missingok=missingok,
                     value=value)

    def disable(self, index):
        """Disable state variable defined by `index` id"""
        self._action(index, StateVariable.enable, missingok=False, value=False)

    def reset(self, index=None):
        """Reset the state variable defined by `index`"""
        if not index is None:
            indexes = [ index ]
        else:
            indexes = self.names

        # do for all
        for index in indexes:
            self._action(index, StateVariable.reset, missingok=False)



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


    def _getOwner(self):
        return self.__owner

    def _setOwner(self, owner):
        if not isinstance(owner, Stateful):
            raise ValueError, \
                  "Owner of the StateCollection must be Stateful object"
        self.__owner = owner


    # Properties
    listing = property(fget=_getListing)
    names = property(fget=_getNames)
    items = property(fget=lambda x:x.__items)
    enabled = property(fget=_getEnabled, fset=_setEnabled)
    owner = property(fget=_getOwner, fset=_setOwner)



class statecollector(type):
    """Intended to collect and compose StateCollection for any child
    class of this metaclass
    """

    def __init__(cls, name, bases, dict):

        if __debug__:
            debug("STCOL",
                  "Collector call for %s.%s, where bases=%s, dict=%s " \
                  % (cls, name, bases, dict))

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
                newitems = base._states_template.items
                if len(newitems) == 0:
                    continue
                if __debug__:
                    debug("STCOL",
                          "Collect states %s for %s from %s" %
                          (newitems, cls, base))
                items.update(newitems)

        if __debug__:
            debug("STCOL",
                  "Creating StateCollection template %s" % cls)

        # and give it ownwership of class
        statecollection = StateCollection(items, cls)
        setattr(cls, "_states_template", statecollection)



class Stateful(object):
    """Base class for stateful objects.

    Classes inherited from this class gain ability to provide state
    variables, accessed as simple properties. Access to state variables
    "internals" is done via states property and interface of
    `StateCollection`.

    NB This one is to replace old State base class
    """

    __metaclass__ = statecollector

    def __init__(self,
                 enable_states=None,
                 disable_states=None,
                 descr=None):

        if enable_states == None:
            enable_states = []
        if disable_states == None:
            disable_states = []

        if not hasattr(self, '_states'):
            # need to check to avoid override of enabled states in the case
            # of multiple inheritance, like both Statefull and Harvestable
            object.__setattr__(self, '_states',
                               copy.deepcopy( \
                                object.__getattribute__(self,
                                                        '_states_template')))

            self._states.owner = self
            self._states.enable(enable_states, missingok=True)
            self._states.disable(disable_states)

            self.__descr = descr

        # bad to have str(self) here since it is a base class and
        # some attributes most probably are not yet set in the original
        # child's __str__
        #if __debug__:
        #    debug("ST", "Stateful.__init__ done for %s" % self)

        if __debug__:
            debug("ST", "Stateful.__init__ was done for %s id %s with descr=%s" \
                % (self.__class__, id(self), descr))


    def __getattribute__(self, index):
        # return all private ones first since smth like __dict__ might be
        # queried by copy before instance is __init__ed
        if index.startswith('__'):
            return object.__getattribute__(self, index)
        states = object.__getattribute__(self, '_states')
        if index in ["states", "_states"]:
            return states
        if states.items.has_key(index):
            return states.get(index)
        else:
            return object.__getattribute__(self, index)

    def __setattr__(self, index, value):
        states = object.__getattribute__(self, '_states')
        if states.items.has_key(index):
            states.set(index, value)
        else:
            object.__setattr__(self, index, value)


    @property
    def states(self):
        return self._states

    def __str__(self):
        return "%s with %s" % (self.__class__.__name__, str(self.states))

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
