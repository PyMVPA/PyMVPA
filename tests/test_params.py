#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Parameter class."""

import unittest, copy

import numpy as N
from sets import Set

from mvpa.misc.state import Stateful
from mvpa.misc.param import Parameter, KernelParameter

class BlankClass(Stateful):
    pass

class SimpleClass(Stateful):
    C = Parameter(1.0, min=0, doc="C parameter")


class ParamsTests(unittest.TestCase):

    def testBlank(self):
        blank  = BlankClass()

        self.failUnlessRaises(AttributeError, blank.__getattribute__, 'states')
        self.failUnlessRaises(AttributeError, blank.__getattribute__, '')

        return

        blank  = BlankClass()

        self.failUnlessEqual(blank.states.items, {})
        self.failUnless(blank.states.enabled == [])
        self.failUnlessRaises(AttributeError, blank.__getattribute__, 'dummy')
        self.failUnlessRaises(AttributeError, blank.__getattribute__, '')

        # we shouldn't use _registerState now since metaclass statecollector wouldn't
        # update the states... may be will be implemented in the future if necessity comes
        return

    def testSimple(self):
        simple  = SimpleClass()

        self.failUnlessEqual(len(simple.params.items), 1)
        self.failUnlessRaises(AttributeError, simple.__getattribute__, 'dummy')
        self.failUnlessRaises(AttributeError, simple.__getattribute__, '')

        self.failUnlessEqual(simple.C, 1.0)
        simple.C = 10.0
        self.failUnlessEqual(simple.C, 10.0)
        simple.params["C"].resetvalue()
        self.failUnlessEqual(simple.C, 1.0)
        self.failUnlessRaises(AttributeError, simple.params.__getattribute__, 'B')

def suite():
    return unittest.makeSuite(ParamsTests)


if __name__ == '__main__':
    import runner

