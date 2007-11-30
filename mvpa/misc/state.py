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

if __debug__:
    from mvpa.misc import debug


class State(dict):

    # _register_states = {'statevariable': Bool}
    #
    # children classes can compactly describe state variables as a static members
    # instead of explicit calling _registerState

    def __init__(self, enable_states=None, disable_states=None, *args, **kwargs):
        """Initialize the state variables of a derived class

        :Parameters:
          `enable_states` : list
             list of states to enable
          `disable_states` : list
             list of states to disable
        """

        self.__registered = {}
        """Dictionary to contain registered states as keys and
        values signal either they are enabled
        """
        dict.__init__(self, *args, **kwargs)

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
        #    debug('ST', 'Class %s is a child of State class but has no states' %
        #          (self.__class__.__name__))

        for key,enabled in register_states.iteritems():
            if (not enable_states is None):
                if (not key in enable_states):
                    if __debug__:
                        debug('ST', 'Disabling state %s since it is not listed' % key +
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


    def __checkIndex(self, index):
        """Verify that given `index` is a known/registered state.

        :Raise `KeyError`: if given `index` is not known
        """
        if not self.__registered.has_key(index):
            raise KeyError, \
                  "State of %s has no key %s registered" \
                  % (self.__class__.__name__, index)


    # actually we have to provide some simple way to set the State
    def __setitem__(self, index, *args, **kwargs):
        """Set value for the `index`.
        """
        self.__checkIndex(index)
        dict.__setitem__(self, index, *args, **kwargs)


    def __getitem__(self, index):
        """Return a value for the given `index`

        :Raise `KeyError`: if given `index` is not known
        :Raise `UnknownStateError`: if no value yet was assigned to `index`
        """
        # XXX Maybe unnecessary to check dict twice for matching member
        # if it is not registered the will be no key like this in the
        # dict, but if registered there need not be a key anyway.
        # Therefore it should be sufficient to check for key in dict.
        # Or do first test only in __debug__
        # XXX yoh: No! registered doesn't imply assigned already value.
        #          That is why it raises UnknownStateError.
        #     otherwise there is an ambiguity -- either it wasn't
        #     registered or it wasn't set in the code
        self.__checkIndex(index)
        if not self.has_key(index):
            raise UnknownStateError("Unknown yet value for '%s'" % index)
        return dict.__getitem__(self, index)


    # XXX think about it -- may be it is worth making whole State
    # handling via static methods to remove any possible overhead of
    # registering the same keys in each constructor
    # Michael: Depends on what kind of objects we want to have a state.
    #          Anyway as we will inherent this class I think the method name
    #          should be a bit more distinctive. What about:
    def _registerState(self, index, enabled=True):
        """Register a new state

        :Parameters:
          `index`
            the index
          `enabled` : Bool
            either the state should be enabled
        """
        if __debug__:
            debug('ST', 'Registering %s state %s for %s' %
                  ({True:'enabled', False:'disabled'}[enabled],
                   index, self.__class__.__name__))
        self.__registered[index] = enabled


    def enableState(self, index):
        self.__checkIndex(index)
        self.__registered[index] = True


    def disableState(self, index):
        self.__checkIndex(index)
        self.__registered[index] = False


    def isStateEnabled(self, index):
        self.__checkIndex(index)
        return self.__registered[index]

    # Properties
    registeredStates = property(fget=lambda x:x.__registered.keys())
    enabledStates = property(fget=lambda x:filter(
        lambda y:x.__registered[y], x.__registered.keys()))
