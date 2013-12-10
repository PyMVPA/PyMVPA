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

        self.assertRaises(AttributeError, empty.__getattribute__, 'ca')

        self.assertEqual(list(blank.ca.items()), [])
        self.assertEqual(len(blank.ca), 0)
        self.assertTrue(blank.ca.enabled == [])
        self.assertRaises(AttributeError, blank.__getattribute__, 'dummy')
        self.assertRaises(AttributeError, blank.__getattribute__, '_')

        # we shouldn't use _registerState now since metaclass statecollector wouldn't
        # update the ca... may be will be implemented in the future if necessity comes
        return

        # add some conditional attribute
        blank._registerState('state1', False)
        self.assertTrue(blank.ca == ['state1'])

        self.assertTrue(blank.ca.is_enabled('state1') == False)
        self.assertTrue(blank.ca.enabled == [])
        self.assertRaises(UnknownStateError, blank.__getattribute__, 'state1')

        # assign value now
        blank.state1 = 123
        # should have no effect since the conditional attribute wasn't enabled
        self.assertRaises(UnknownStateError, blank.__getattribute__, 'state1')

        # lets enable and assign
        blank.ca.enable('state1')
        blank.state1 = 123
        self.assertTrue(blank.state1 == 123)

        # we should not share ca across instances at the moment, so an arbitrary
        # object could carry some custom ca
        self.assertTrue(blank2.ca == [])
        self.assertRaises(AttributeError, blank2.__getattribute__, 'state1')


    def test_proper_state(self):
        proper   = TestClassProper()
        proper2  = TestClassProper(enable_ca=['state1'], disable_ca=['state2'])

        # disable_ca should override anything in enable_ca
        proper3 = TestClassProper(enable_ca=['all'], disable_ca='all')

        self.assertEqual(len(proper3.ca.enabled), 0,
            msg="disable_ca should override anything in enable_ca")

        proper.ca.state2 = 1000
        value = proper.ca.state2
        self.assertEqual(proper.ca.state2, 1000, msg="Simple assignment/retrieval")

        proper.ca.disable('state2')
        proper.ca.state2 = 10000
        self.assertEqual(proper.ca.state2, 1000, msg="Simple assignment after being disabled")

        proper4 = copy.deepcopy(proper)

        proper.ca.reset('state2')
        self.assertRaises(UnknownStateError, proper.ca.__getattribute__, 'state2')
        """Must be blank after being reset"""

        self.assertEqual(proper4.ca.state2, 1000,
            msg="Simple assignment after being reset in original instance")


        proper.ca.enable(['state2'])
        self.assertEqual(set(proper.ca.keys()), set(['state1', 'state2']))
        if __debug__ and 'ENFORCE_CA_ENABLED' in debug.active:
            # skip testing since all ca are on now
            return
        self.assertTrue(proper.ca.enabled == ['state2'])

        self.assertTrue(set(proper2.ca.enabled) == set(['state1']))

        self.assertRaises(AttributeError, proper.__getattribute__, 'state12')

        # if documentary on the state is appropriate
        self.assertEqual(proper2.ca.listing,
                             ['%sstate1+%s: state1 doc' % (_def_sep, _def_sep),
                              '%sstate2%s: state2 doc' % (_def_sep, _def_sep)])

        # if __str__ lists correct number of ca
        str_ = str(proper2)
        self.assertTrue(str_.find('2 ca:') != -1)

        # check if disable works
        self.assertTrue(set(proper2.ca.enabled), set(['state1']))

        proper2.ca.disable("all")
        self.assertEqual(set(proper2.ca.enabled), set())

        proper2.ca.enable("all")
        self.assertEqual(len(proper2.ca.enabled), 2)

        proper2.ca.state1, proper2.ca.state2 = 1,2
        self.assertEqual(proper2.ca.state1, 1)
        self.assertEqual(proper2.ca.state2, 2)

        # now reset them
        proper2.ca.reset('all')
        self.assertRaises(UnknownStateError, proper2.ca.__getattribute__, 'state1')
        self.assertRaises(UnknownStateError, proper2.ca.__getattribute__, 'state2')


    def test_get_save_enabled(self):
        """Check if we can store/restore set of enabled ca"""

        if __debug__ and 'ENFORCE_CA_ENABLED' in debug.active:
            # skip testing since all ca are on now
            return

        proper  = TestClassProper()
        enabled_ca = proper.ca.enabled
        proper.ca.enable('state1')

        self.assertTrue(enabled_ca != proper.ca.enabled,
                        msg="New enabled ca should differ from previous")

        self.assertTrue(set(proper.ca.enabled) == set(['state1', 'state2']),
                        msg="Making sure that we enabled all ca of interest")

        proper.ca.enabled = enabled_ca
        self.assertTrue(enabled_ca == proper.ca.enabled,
                        msg="List of enabled ca should return to original one")


    # TODO: make test for _copy_ca_ or whatever comes as an alternative

    def test_stored_temporarily(self):
        proper   = TestClassProper()
        properch = TestClassProperChild(enable_ca=["state1"])

        if __debug__ and 'ENFORCE_CA_ENABLED' in debug.active:
            # skip testing since all ca are on now
            return

        self.assertEqual(proper.ca.enabled, ["state2"])
        proper.ca.change_temporarily(
            enable_ca=["state1"], other=properch)
        self.assertEqual(set(proper.ca.enabled),
                             set(["state1", "state2"]))
        proper.ca.reset_changed_temporarily()
        self.assertEqual(proper.ca.enabled, ["state2"])

        # allow to enable disable without other instance
        proper.ca.change_temporarily(
            enable_ca=["state1", "state2"])
        self.assertEqual(set(proper.ca.enabled),
                             set(["state1", "state2"]))
        proper.ca.reset_changed_temporarily()
        self.assertEqual(proper.ca.enabled, ["state2"])


    def test_proper_state_child(self):
        """
        Simple test if child gets conditional attributes from the parent as well
        """
        proper = TestClassProperChild()
        self.assertEqual(set(proper.ca.keys()),
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

        self.assertEqual(s1.ca.is_enabled("v1"), True)
        s1.ca.v1 = 12
        s12.ca.v1 = 120
        s2.ca.v2 = 100

        self.assertEqual(len(s2.ca.listing), 1)

        self.assertEqual(s1.ca.v1, 12)
        try:
            tempvalue = s1__.ca.v1__
            self.fail("Should have puked since values were not enabled yet")
        except:
            pass


    def test_parametrized(self):

        self.assertRaises(TypeError, TestClassParametrized,
            p2=34, enable_ca=['state1'],
            msg="Should raise an exception if argument doesn't correspond to"
                "any parameter")
        a = TestClassParametrized(p1=123, enable_ca=['state1'])
        self.assertEqual(a.params.p1, 123, msg="We must have assigned value to instance")
        self.assertTrue('state1' in a.ca.enabled,
                        msg="state1 must have been enabled")

        if (__debug__ and 'ID_IN_REPR' in debug.active):
            # next tests would fail due to ID in the tails
            return

        # validate that string representation of the object is valid and consistent
        a_str = repr(a)
        try:
            import mvpa2.tests.test_state as test_state
            exec("a2=%s" % a_str)
        except Exception as e:
            self.fail(msg="Failed to generate an instance out of "
                      "representation %s. Got exception: %s" % (a_str, e))

        # For specifics of difference in exec keyword from exec() function in
        # python3 see
        # http://stackoverflow.com/questions/6561482/why-did-python-3-changes-to-exec-break-this-code
        # which mandates us to use exec here around repr so it gets access to
        # above a2 placed into locals()
        exec('a2_str_=repr(a2)')
        a2_str = locals()['a2_str_']       # crazy ha?  it must not be a2_str either
        self.assertTrue(a2_str == a_str,
            msg="Generated object must have the same repr. Got %s and %s" %
            (a_str, a2_str))

        # Test at least that repr of collection is of correct syntax
        aparams_str = repr(a.params)
        try:
            import mvpa2.tests.test_state as test_state
            exec("aparams2=%s" % aparams_str)
        except Exception as e:
            self.fail(msg="Failed to generate an instance out of "
                      "representation  %s of params. Got exception: %s" % (aparams_str, e))

    def test_read_only(self):
        # Should be able to assign in constructor
        cro = TestClassReadOnlyParameter(paramro=12)
        # but not run time
        self.assertRaises(RuntimeError, cro.params['paramro']._set, 13)
        # Test if value wasn't actually changed
        self.assertEqual(cro.params.paramro, 12)

    def test_value_in_constructor(self):
        param = Parameter(0, value=True)
        self.assertTrue(param.value)

    def test_deep_copying_state_variable(self):
        for v in (True, False):
            sv = ConditionalAttribute(enabled=v,
                               doc="Testing")
            sv.enabled = not v
            sv_dc = copy.deepcopy(sv)
            if not (__debug__ and 'ENFORCE_CA_ENABLED' in debug.active):
                self.assertEqual(sv.enabled, sv_dc.enabled)
            self.assertEqual(sv.name, sv_dc.name)
            self.assertEqual(sv._instance_index, sv_dc._instance_index)

def suite():  # pragma: no cover
    return unittest.makeSuite(StateTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

