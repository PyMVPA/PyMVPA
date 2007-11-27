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

from mvpa.misc.state import State
from mvpa.misc.exceptions import UnknownStateError

class TestClassBlank(State):
    def __init__(self):
        State.__init__(self)

class TestClassProper(State):

    _register_states = { 'state1': False, 'state2': True }

    def __init__(self, **kargs):
        State.__init__(self, **kargs)


class StateTests(unittest.TestCase):

    def testBlankState(self):
        blank  = TestClassBlank()
        blank2 = TestClassBlank()

        self.failUnless(blank.registeredStates == [])
        self.failUnless(blank.enabledStates == [])
        self.failUnlessRaises(KeyError, blank.__getitem__, 'dummy')
        self.failUnlessRaises(KeyError, blank.__getitem__, '')

        # add some state variable
        blank._registerState('state1', False)
        self.failUnless(blank.registeredStates == ['state1'])

        self.failUnless(blank.isStateEnabled('state1') == False)
        self.failUnless(blank.enabledStates == [])
        self.failUnlessRaises(UnknownStateError, blank.__getitem__, 'state1')

        # assign value now
        blank['state1'] = 123
        self.failUnless(blank['state1'] == 123)

        # we should not share states across instances at the moment, so an arbitrary
        # object could carry some custom states
        self.failUnless(blank2.registeredStates == [])
        self.failUnlessRaises(KeyError, blank2.__getitem__, 'state1')


    def testProperState(self):
        proper   = TestClassProper()
        proper2  = TestClassProper(enable_states=['state1'])

        self.failUnless(Set(proper.registeredStates) == Set(proper._register_states.keys()))
        self.failUnless(proper.enabledStates == ['state2'])

        self.failUnless(Set(proper2.registeredStates) == Set(proper._register_states.keys()))
        self.failUnless(Set(proper2.enabledStates) == Set(['state1']))

        self.failUnlessRaises(KeyError, proper.__getitem__, 'state12')
        proper2._registerState('state3')
        # check default enabled
        self.failUnless(Set(proper2.enabledStates) == Set(['state1', 'state3']))
        # check if disable works
        proper2.disableState('state3')
        self.failUnless(Set(proper2.enabledStates) == Set(['state1']))


def suite():
    return unittest.makeSuite(StateTests)


if __name__ == '__main__':
    import test_runner

