# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA State parent class"""

import unittest, copy

import numpy as N
from sets import Set

from mvpa.base import externals

from mvpa.misc.state import StateVariable, ClassWithCollections, \
     ParameterCollection, _def_sep
from mvpa.misc.param import *
from mvpa.misc.exceptions import UnknownStateError

if __debug__:
    from mvpa.base import debug

class TestClassEmpty(ClassWithCollections):
    pass

class TestClassBlank(ClassWithCollections):
    # We can force to have 'states' present even though we don't have
    # any StateVariable defined here -- it might be added later on at run time
    _ATTRIBUTE_COLLECTIONS = ['states']
    pass

class TestClassBlankNoExplicitStates(ClassWithCollections):
    pass

class TestClassProper(ClassWithCollections):

    state1 = StateVariable(enabled=False, doc="state1 doc")
    state2 = StateVariable(enabled=True, doc="state2 doc")


class TestClassProperChild(TestClassProper):

    state4 = StateVariable(enabled=False, doc="state4 doc")


class TestClassParametrized(TestClassProper, ClassWithCollections):
    p1 = Parameter(0)
    state0 = StateVariable(enabled=False)

    def __init__(self, **kwargs):
        # XXX make such example when we actually need to invoke
        # constructor
        # TestClassProper.__init__(self, **kwargs)
        ClassWithCollections.__init__(self, **kwargs)


class StateTests(unittest.TestCase):

    def testBlankState(self):
        empty  = TestClassEmpty()
        blank  = TestClassBlank()
        blank2 = TestClassBlank()

        self.failUnlessRaises(AttributeError, empty.__getattribute__, 'states')

        self.failUnlessEqual(blank.states.items, {})
        self.failUnless(blank.states.enabled == [])
        self.failUnlessRaises(AttributeError, blank.__getattribute__, 'dummy')
        self.failUnlessRaises(AttributeError, blank.__getattribute__, '_')

        # we shouldn't use _registerState now since metaclass statecollector wouldn't
        # update the states... may be will be implemented in the future if necessity comes
        return

        # add some state variable
        blank._registerState('state1', False)
        self.failUnless(blank.states == ['state1'])

        self.failUnless(blank.states.isEnabled('state1') == False)
        self.failUnless(blank.states.enabled == [])
        self.failUnlessRaises(UnknownStateError, blank.__getattribute__, 'state1')

        # assign value now
        blank.state1 = 123
        # should have no effect since the state variable wasn't enabled
        self.failUnlessRaises(UnknownStateError, blank.__getattribute__, 'state1')

        # lets enable and assign
        blank.states.enable('state1')
        blank.state1 = 123
        self.failUnless(blank.state1 == 123)

        # we should not share states across instances at the moment, so an arbitrary
        # object could carry some custom states
        self.failUnless(blank2.states == [])
        self.failUnlessRaises(AttributeError, blank2.__getattribute__, 'state1')


    def testProperState(self):
        proper   = TestClassProper()
        proper2  = TestClassProper(enable_states=['state1'], disable_states=['state2'])

        # disable_states should override anything in enable_states
        proper3 = TestClassProper(enable_states=['all'], disable_states='all')

        self.failUnlessEqual(len(proper3.states.enabled), 0,
            msg="disable_states should override anything in enable_states")

        proper.state2 = 1000
        value = proper.state2
        self.failUnlessEqual(proper.state2, 1000, msg="Simple assignment/retrieval")

        proper.states.disable('state2')
        proper.state2 = 10000
        self.failUnlessEqual(proper.state2, 1000, msg="Simple assignment after being disabled")

        proper4 = copy.deepcopy(proper)

        proper.states.reset('state2')
        self.failUnlessRaises(UnknownStateError, proper.__getattribute__, 'state2')
        """Must be blank after being reset"""

        self.failUnlessEqual(proper4.state2, 1000,
            msg="Simple assignment after being reset in original instance")


        proper.states.enable(['state2'])
        self.failUnlessEqual(Set(proper.states.names), Set(['state1', 'state2']))
        self.failUnless(proper.states.enabled == ['state2'])

        self.failUnless(Set(proper2.states.enabled) == Set(['state1']))

        self.failUnlessRaises(AttributeError, proper.__getattribute__, 'state12')

        # if documentary on the state is appropriate
        self.failUnlessEqual(proper2.states.listing,
                             ['%sstate1+%s: state1 doc' % (_def_sep, _def_sep),
                              '%sstate2%s: state2 doc' % (_def_sep, _def_sep)])

        # if __str__ lists correct number of states
        str_ = str(proper2)
        self.failUnless(str_.find('2 states:') != -1)

        # check if disable works
        self.failUnless(Set(proper2.states.enabled), Set(['state1']))

        proper2.states.disable("all")
        self.failUnlessEqual(Set(proper2.states.enabled), Set())

        proper2.states.enable("all")
        self.failUnlessEqual(len(proper2.states.enabled), 2)

        proper2.state1, proper2.state2 = 1,2
        self.failUnlessEqual(proper2.state1, 1)
        self.failUnlessEqual(proper2.state2, 2)

        # now reset them
        proper2.states.reset('all')
        self.failUnlessRaises(UnknownStateError, proper2.__getattribute__, 'state1')
        self.failUnlessRaises(UnknownStateError, proper2.__getattribute__, 'state2')


    def testGetSaveEnabled(self):
        """Check if we can store/restore set of enabled states"""

        proper  = TestClassProper()
        enabled_states = proper.states.enabled
        proper.states.enable('state1')

        self.failUnless(enabled_states != proper.states.enabled,
                        msg="New enabled states should differ from previous")

        self.failUnless(Set(proper.states.enabled) == Set(['state1', 'state2']),
                        msg="Making sure that we enabled all states of interest")

        proper.states.enabled = enabled_states
        self.failUnless(enabled_states == proper.states.enabled,
                        msg="List of enabled states should return to original one")


    # TODO: make test for _copy_states_ or whatever comes as an alternative

    def testStoredTemporarily(self):
        proper   = TestClassProper()
        properch = TestClassProperChild(enable_states=["state1"])

        self.failUnlessEqual(proper.states.enabled, ["state2"])
        proper.states._changeTemporarily(
            enable_states=["state1"], other=properch)
        self.failUnlessEqual(Set(proper.states.enabled),
                             Set(["state1", "state2"]))
        proper.states._resetEnabledTemporarily()
        self.failUnlessEqual(proper.states.enabled, ["state2"])

        # allow to enable disable without other instance
        proper.states._changeTemporarily(
            enable_states=["state1", "state2"])
        self.failUnlessEqual(Set(proper.states.enabled),
                             Set(["state1", "state2"]))
        proper.states._resetEnabledTemporarily()
        self.failUnlessEqual(proper.states.enabled, ["state2"])


    def testProperStateChild(self):
        """
        Simple test if child gets state variables from the parent as well
        """
        proper = TestClassProperChild()
        self.failUnlessEqual(Set(proper.states.names),
                             Set(['state1', 'state2', 'state4']))


    def testStateVariables(self):
        """To test new states"""

        class S1(ClassWithCollections):
            v1 = StateVariable(enabled=True, doc="values1 is ...")
            v1XXX = StateVariable(enabled=False, doc="values1 is ...")


        class S2(ClassWithCollections):
            v2 = StateVariable(enabled=True, doc="values12 is ...")

        class S1_(S1):
            pass

        class S1__(S1_):
            v1__ = StateVariable(enabled=False)

        class S12(S1__, S2):
            v12 = StateVariable()

        s1, s2, s1_, s1__, s12 = S1(), S2(), S1_(), S1__(), S12()

        self.failUnlessEqual(s1.states.isEnabled("v1"), True)
        s1.v1 = 12
        s12.v1 = 120
        s2.v2 = 100

        self.failUnlessEqual(len(s2.states.listing), 1)

        self.failUnlessEqual(s1.v1, 12)
        try:
            tempvalue = s1__.v1__
            self.fail("Should have puked since values were not enabled yet")
        except:
            pass


    def testParametrized(self):

        self.failUnlessRaises(TypeError, TestClassParametrized,
            p2=34, enable_states=['state1'],
            msg="Should raise an exception if argument doesn't correspond to"
                "any parameter")
        a = TestClassParametrized(p1=123, enable_states=['state1'])
        self.failUnlessEqual(a.p1, 123, msg="We must have assigned value to instance")
        self.failUnless('state1' in a.states.enabled,
                        msg="state1 must have been enabled")

        if (__debug__ and 'ID_IN_REPR' in debug.active):
            # next tests would fail due to ID in the tails
            return

        # validate that string representation of the object is valid and consistent
        a_str = `a`
        try:
            import test_state
            exec "a2=%s" % a_str
        except Exception, e:
            self.fail(msg="Failed to generate an instance out of "
                      "representation %s. Got exception: %s" % (a_str, e))

        a2_str = `a2`
        self.failUnless(a2_str == a_str,
            msg="Generated object must have the same repr. Got %s and %s" %
            (a_str, a2_str))

        # Test at least that repr of collection is of correct syntax
        aparams_str = `a.params`
        try:
            import test_state
            exec "aparams2=%s" % aparams_str
        except Exception, e:
            self.fail(msg="Failed to generate an instance out of "
                      "representation  %s of params. Got exception: %s" % (aparams_str, e))


def suite():
    return unittest.makeSuite(StateTests)


if __name__ == '__main__':
    import runner

