#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA State parent class"""

import unittest

import numpy as N
from sets import Set

from mvpa.misc.state import Statefull, StateVariable
from mvpa.misc.exceptions import UnknownStateError

class TestClassBlank(Statefull):
    pass

class TestClassProper(Statefull):

    state1 = StateVariable(enabled=False, doc="state1 doc")
    state2 = StateVariable(enabled=True, doc="state2 doc")


class TestClassProperChild(TestClassProper):

    state4 = StateVariable(enabled=False, doc="state4 doc")


class StateTests(unittest.TestCase):

    def _testBlankState(self):
        blank  = TestClassBlank()
        blank2 = TestClassBlank()

        self.failUnless(blank.states == [])
        self.failUnless(blank.enabledStates == [])
        self.failUnlessRaises(KeyError, blank.__getitem__, 'dummy')
        self.failUnlessRaises(KeyError, blank.__getitem__, '')

        # add some state variable
        blank._registerState('state1', False)
        self.failUnless(blank.states == ['state1'])

        self.failUnless(blank.isStateEnabled('state1') == False)
        self.failUnless(blank.enabledStates == [])
        self.failUnlessRaises(UnknownStateError, blank.__getitem__, 'state1')

        # assign value now
        blank['state1'] = 123
        # should have no effect since the state variable wasn't enabled
        self.failUnlessRaises(UnknownStateError, blank.__getitem__, 'state1')

        # lets enable and assign
        blank.enableState('state1')
        blank['state1'] = 123
        self.failUnless(blank['state1'] == 123)

        # we should not share states across instances at the moment, so an arbitrary
        # object could carry some custom states
        self.failUnless(blank2.states == [])
        self.failUnlessRaises(KeyError, blank2.__getitem__, 'state1')


    def _testProperState(self):
        proper   = TestClassProper()
        proper2  = TestClassProper(enable_states=['state1'], disable_states=['state2'])

        # can't enable disable all at the same time
        self.failUnlessRaises(ValueError, TestClassProper,
                              enable_states=['all'], disable_states='all')
        self.failUnless(Set(proper.states) == Set(proper._register_states.keys()))
        self.failUnless(proper.enabledStates == ['state2'])

        self.failUnless(Set(proper2.states) == Set(proper._register_states.keys()))
        self.failUnless(Set(proper2.enabledStates) == Set(['state1']))

        self.failUnlessRaises(KeyError, proper.__getitem__, 'state12')
        proper2._registerState('state3', doc="State3 Doc")

        # if documentary on the state is appropriate
        self.failUnless(proper2.listStates() == \
                        ['state1[enabled]: None',
                         'state2: None',
                         'state3[enabled]: State3 Doc'])

        # if __str__ lists correct number of states
        str_ = str(proper2)
        self.failUnless(str_.find('3 states:') != -1)

        # check default enabled
        self.failUnless(Set(proper2.enabledStates) == Set(['state1', 'state3']))
        # check if disable works
        proper2.disableState('state3')
        self.failUnless(Set(proper2.enabledStates) == Set(['state1']))

        proper2.disableState("all")
        self.failUnlessEqual(Set(proper2.enabledStates), Set())

        proper2.enableState("all")
        self.failUnlessEqual(len(proper2.enabledStates), 3)


    def _testGetSaveEnabled(self):
        """Check if we can store/restore set of enabled states"""

        proper  = TestClassProper()
        enabled_states = proper.enabledStates
        proper.enableState('state1')

        self.failUnless(enabled_states != proper.enabledStates,
                        msg="New enabled states should differ from previous")

        self.failUnless(Set(proper.enabledStates) == Set(['state1', 'state2']),
                        msg="Making sure that we enabled all states of interest")
        proper.enabledStates = enabled_states
        self.failUnless(enabled_states == proper.enabledStates,
                        msg="List of enabled states should return to original one")


    def _testStoredEnableStates(self):
        """Check if the states mentioned in enable_states
        are retroactively enabled while being registered"""
        proper  = TestClassProper(enable_states=['newstate'])
        # state is not yet registered thus shouldn't be known
        self.failUnlessRaises(KeyError, proper.__getitem__, 'newstate')

        proper._registerState("newstate", enabled=False)
        self.failUnlessEqual(proper.isStateEnabled("newstate"), True)

        # check if not mentioned in enable_states doesn't get enabled
        proper._registerState("newstate2", enabled=False)
        self.failUnlessEqual(proper.isStateEnabled("newstate2"), False)


    # TODO: make test for _copy_states_ or whatever comes as an alternative

    def _testStoredTemporarily(self):
        proper   = TestClassProper()
        properch = TestClassProperChild(enable_states=["state1"])

        self.failUnlessEqual(proper.enabledStates, ["state2"])
        proper._enableStatesTemporarily(["state1"], properch)
        self.failUnlessEqual(Set(proper.enabledStates),
                             Set(["state1", "state2"]))
        proper._resetEnabledTemporarily()
        self.failUnlessEqual(proper.enabledStates, ["state2"])

        # allow to enable disable without other instance
        proper._enableStatesTemporarily(["state1", "state2"])
        self.failUnlessEqual(Set(proper.enabledStates),
                             Set(["state1", "state2"]))
        proper._resetEnabledTemporarily()
        self.failUnlessEqual(proper.enabledStates, ["state2"])


    def _testProperStateChild(self):
        """
        Actually it would fail which makes it no sense to use
        _register_states class variables
        """
        proper = TestClassProperChild()
        self.failUnless(Set(proper.states) ==
            Set(TestClassProperChild._register_states).union(
            Set(TestClassProper._register_states)))


    def testStateVariables(self):
        """To test new states"""

        from mvpa.misc.state import StateVariable, Statefull

        class S1(Statefull):
            v1 = StateVariable(enabled=True, doc="values1 is ...")
            v1XXX = StateVariable(enabled=False, doc="values1 is ...")


        class S2(Statefull):
            v2 = StateVariable(enabled=True, doc="values12 is ...")

        class S1_(S1):
            pass

        class S1__(S1_):
            v1__ = StateVariable(enabled=False)

        class S12(S1__, S2):
            v12 = StateVariable()

        s1, s2, s1_, s1__, s12 = S1(), S2(), S1_(), S1__(), S12()

        self.failUnlessEqual(s1.states.isStateEnabled("v1"), True)
        s1.v1 = 12
        s12.v1 = 120
        s2.v2 = 100

        self.failUnlessEqual(s1.v1, 12)
        try:
            print s1__.v1__
            self.fail("Should have puked since values were not enabled yet")
        except:
            pass


def suite():
    return unittest.makeSuite(StateTests)


if __name__ == '__main__':
    import test_runner

