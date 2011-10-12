# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA State parent class"""

import unittest
import mvpa2.support.copy as copy

import numpy as np

from mvpa2.base import externals

from mvpa2.base.state import ConditionalAttribute, ClassWithCollections, \
     ParameterCollection, _def_sep
from mvpa2.base.param import *
from mvpa2.misc.exceptions import UnknownStateError

if __debug__:
    from mvpa2.base import debug

class TestClassEmpty(ClassWithCollections):
    pass

class TestClassBlank(ClassWithCollections):
    # We can force to have 'ca' present even though we don't have
    # any ConditionalAttribute defined here -- it might be added later on at run time
    _ATTRIBUTE_COLLECTIONS = ['ca']
    pass

class TestClassBlankNoExplicitStates(ClassWithCollections):
    pass

class TestClassProper(ClassWithCollections):

    state1 = ConditionalAttribute(enabled=False, doc="state1 doc")
    state2 = ConditionalAttribute(enabled=True, doc="state2 doc")


class TestClassProperChild(TestClassProper):

    state4 = ConditionalAttribute(enabled=False, doc="state4 doc")

class TestClassReadOnlyParameter(ClassWithCollections):
    paramro = Parameter(0, doc="state4 doc", ro=True)


class TestClassParametrized(TestClassProper, ClassWithCollections):
    p1 = Parameter(0)
    state0 = ConditionalAttribute(enabled=False)

    def __init__(self, **kwargs):
        # XXX make such example when we actually need to invoke
        # constructor
        # TestClassProper.__init__(self, **kwargs)
        ClassWithCollections.__init__(self, **kwargs)


class StateTests(unittest.TestCase):

    def test_blank_state(self):
        empty  = TestClassEmpty()
        blank  = TestClassBlank()
        blank2 = TestClassBlank()

        self.failUnlessRaises(AttributeError, empty.__getattribute__, 'ca')

        self.failUnlessEqual(blank.ca.items(), [])
        self.failUnlessEqual(len(blank.ca), 0)
        self.failUnless(blank.ca.enabled == [])
        self.failUnlessRaises(AttributeError, blank.__getattribute__, 'dummy')
        self.failUnlessRaises(AttributeError, blank.__getattribute__, '_')

        # we shouldn't use _registerState now since metaclass statecollector wouldn't
        # update the ca... may be will be implemented in the future if necessity comes
        return

        # add some conditional attribute
        blank._registerState('state1', False)
        self.failUnless(blank.ca == ['state1'])

        self.failUnless(blank.ca.is_enabled('state1') == False)
        self.failUnless(blank.ca.enabled == [])
        self.failUnlessRaises(UnknownStateError, blank.__getattribute__, 'state1')

        # assign value now
        blank.state1 = 123
        # should have no effect since the conditional attribute wasn't enabled
        self.failUnlessRaises(UnknownStateError, blank.__getattribute__, 'state1')

        # lets enable and assign
        blank.ca.enable('state1')
        blank.state1 = 123
        self.failUnless(blank.state1 == 123)

        # we should not share ca across instances at the moment, so an arbitrary
        # object could carry some custom ca
        self.failUnless(blank2.ca == [])
        self.failUnlessRaises(AttributeError, blank2.__getattribute__, 'state1')


    def test_proper_state(self):
        proper   = TestClassProper()
        proper2  = TestClassProper(enable_ca=['state1'], disable_ca=['state2'])

        # disable_ca should override anything in enable_ca
        proper3 = TestClassProper(enable_ca=['all'], disable_ca='all')

        self.failUnlessEqual(len(proper3.ca.enabled), 0,
            msg="disable_ca should override anything in enable_ca")

        proper.ca.state2 = 1000
        value = proper.ca.state2
        self.failUnlessEqual(proper.ca.state2, 1000, msg="Simple assignment/retrieval")

        proper.ca.disable('state2')
        proper.ca.state2 = 10000
        self.failUnlessEqual(proper.ca.state2, 1000, msg="Simple assignment after being disabled")

        proper4 = copy.deepcopy(proper)

        proper.ca.reset('state2')
        self.failUnlessRaises(UnknownStateError, proper.ca.__getattribute__, 'state2')
        """Must be blank after being reset"""

        self.failUnlessEqual(proper4.ca.state2, 1000,
            msg="Simple assignment after being reset in original instance")


        proper.ca.enable(['state2'])
        self.failUnlessEqual(set(proper.ca.keys()), set(['state1', 'state2']))
        if __debug__ and 'ENFORCE_CA_ENABLED' in debug.active:
            # skip testing since all ca are on now
            return
        self.failUnless(proper.ca.enabled == ['state2'])

        self.failUnless(set(proper2.ca.enabled) == set(['state1']))

        self.failUnlessRaises(AttributeError, proper.__getattribute__, 'state12')

        # if documentary on the state is appropriate
        self.failUnlessEqual(proper2.ca.listing,
                             ['%sstate1+%s: state1 doc' % (_def_sep, _def_sep),
                              '%sstate2%s: state2 doc' % (_def_sep, _def_sep)])

        # if __str__ lists correct number of ca
        str_ = str(proper2)
        self.failUnless(str_.find('2 ca:') != -1)

        # check if disable works
        self.failUnless(set(proper2.ca.enabled), set(['state1']))

        proper2.ca.disable("all")
        self.failUnlessEqual(set(proper2.ca.enabled), set())

        proper2.ca.enable("all")
        self.failUnlessEqual(len(proper2.ca.enabled), 2)

        proper2.ca.state1, proper2.ca.state2 = 1,2
        self.failUnlessEqual(proper2.ca.state1, 1)
        self.failUnlessEqual(proper2.ca.state2, 2)

        # now reset them
        proper2.ca.reset('all')
        self.failUnlessRaises(UnknownStateError, proper2.ca.__getattribute__, 'state1')
        self.failUnlessRaises(UnknownStateError, proper2.ca.__getattribute__, 'state2')


    def test_get_save_enabled(self):
        """Check if we can store/restore set of enabled ca"""

        if __debug__ and 'ENFORCE_CA_ENABLED' in debug.active:
            # skip testing since all ca are on now
            return

        proper  = TestClassProper()
        enabled_ca = proper.ca.enabled
        proper.ca.enable('state1')

        self.failUnless(enabled_ca != proper.ca.enabled,
                        msg="New enabled ca should differ from previous")

        self.failUnless(set(proper.ca.enabled) == set(['state1', 'state2']),
                        msg="Making sure that we enabled all ca of interest")

        proper.ca.enabled = enabled_ca
        self.failUnless(enabled_ca == proper.ca.enabled,
                        msg="List of enabled ca should return to original one")


    # TODO: make test for _copy_ca_ or whatever comes as an alternative

    def test_stored_temporarily(self):
        proper   = TestClassProper()
        properch = TestClassProperChild(enable_ca=["state1"])

        if __debug__ and 'ENFORCE_CA_ENABLED' in debug.active:
            # skip testing since all ca are on now
            return

        self.failUnlessEqual(proper.ca.enabled, ["state2"])
        proper.ca.change_temporarily(
            enable_ca=["state1"], other=properch)
        self.failUnlessEqual(set(proper.ca.enabled),
                             set(["state1", "state2"]))
        proper.ca.reset_changed_temporarily()
        self.failUnlessEqual(proper.ca.enabled, ["state2"])

        # allow to enable disable without other instance
        proper.ca.change_temporarily(
            enable_ca=["state1", "state2"])
        self.failUnlessEqual(set(proper.ca.enabled),
                             set(["state1", "state2"]))
        proper.ca.reset_changed_temporarily()
        self.failUnlessEqual(proper.ca.enabled, ["state2"])


    def test_proper_state_child(self):
        """
        Simple test if child gets conditional attributes from the parent as well
        """
        proper = TestClassProperChild()
        self.failUnlessEqual(set(proper.ca.keys()),
                             set(['state1', 'state2', 'state4']))


    def test_state_variables(self):
        """To test new ca"""

        class S1(ClassWithCollections):
            v1 = ConditionalAttribute(enabled=True, doc="values1 is ...")
            v1XXX = ConditionalAttribute(enabled=False, doc="values1 is ...")


        class S2(ClassWithCollections):
            v2 = ConditionalAttribute(enabled=True, doc="values12 is ...")

        class S1_(S1):
            pass

        class S1__(S1_):
            v1__ = ConditionalAttribute(enabled=False)

        class S12(S1__, S2):
            v12 = ConditionalAttribute()

        s1, s2, s1_, s1__, s12 = S1(), S2(), S1_(), S1__(), S12()

        self.failUnlessEqual(s1.ca.is_enabled("v1"), True)
        s1.ca.v1 = 12
        s12.ca.v1 = 120
        s2.ca.v2 = 100

        self.failUnlessEqual(len(s2.ca.listing), 1)

        self.failUnlessEqual(s1.ca.v1, 12)
        try:
            tempvalue = s1__.ca.v1__
            self.fail("Should have puked since values were not enabled yet")
        except:
            pass


    def test_parametrized(self):

        self.failUnlessRaises(TypeError, TestClassParametrized,
            p2=34, enable_ca=['state1'],
            msg="Should raise an exception if argument doesn't correspond to"
                "any parameter")
        a = TestClassParametrized(p1=123, enable_ca=['state1'])
        self.failUnlessEqual(a.params.p1, 123, msg="We must have assigned value to instance")
        self.failUnless('state1' in a.ca.enabled,
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

    def test_read_only(self):
        # Should be able to assign in constructor
        cro = TestClassReadOnlyParameter(paramro=12)
        # but not run time
        self.failUnlessRaises(RuntimeError, cro.params['paramro']._set, 13)
        # Test if value wasn't actually changed
        self.failUnlessEqual(cro.params.paramro, 12)

    def test_value_in_constructor(self):
        param = Parameter(0, value=True)
        self.failUnless(param.value)

    def test_deep_copying_state_variable(self):
        for v in (True, False):
            sv = ConditionalAttribute(enabled=v,
                               doc="Testing")
            sv.enabled = not v
            sv_dc = copy.deepcopy(sv)
            self.failUnlessEqual(sv.enabled, sv_dc.enabled)
            self.failUnlessEqual(sv.name, sv_dc.name)
            self.failUnlessEqual(sv._instance_index, sv_dc._instance_index)

def suite():
    return unittest.makeSuite(StateTests)


if __name__ == '__main__':
    import runner

