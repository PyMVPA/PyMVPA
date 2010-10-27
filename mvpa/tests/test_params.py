# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Parameter class."""

import unittest, copy

import numpy as N

from mvpa.datasets import Dataset
from mvpa.misc.state import ClassWithCollections, StateVariable
from mvpa.misc.param import Parameter, KernelParameter

from tests_warehouse_clfs import SameSignClassifier

class ParametrizedClassifier(SameSignClassifier):
    p1 = Parameter(1.0)
    kp1 = KernelParameter(100.0)

class ParametrizedClassifierExtended(ParametrizedClassifier):
    def __init__(self):
        ParametrizedClassifier.__init__(self)
        self.kernel_params.add(KernelParameter(200.0, doc="Very useful param", name="kp2"))

class BlankClass(ClassWithCollections):
    pass

class SimpleClass(ClassWithCollections):
    C = Parameter(1.0, min=0, doc="C parameter")

class MixedClass(ClassWithCollections):
    C = Parameter(1.0, min=0, doc="C parameter")
    D = Parameter(3.0, min=0, doc="D parameter")
    state1 = StateVariable(doc="bogus")

class ParamsTests(unittest.TestCase):

    def testBlank(self):
        blank  = BlankClass()

        self.failUnlessRaises(AttributeError, blank.__getattribute__, 'states')
        self.failUnlessRaises(IndexError, blank.__getattribute__, '')

    def testSimple(self):
        simple  = SimpleClass()

        self.failUnlessEqual(len(simple.params.items), 1)
        self.failUnlessRaises(AttributeError, simple.__getattribute__, 'dummy')
        self.failUnlessRaises(IndexError, simple.__getattribute__, '')

        self.failUnlessEqual(simple.C, 1.0)
        self.failUnlessEqual(simple.params.isSet("C"), False)
        self.failUnlessEqual(simple.params.isSet(), False)
        self.failUnlessEqual(simple.params["C"].isDefault, True)
        self.failUnlessEqual(simple.params["C"].equalDefault, True)

        simple.C = 1.0
        # we are not actually setting the value if == default
        self.failUnlessEqual(simple.params["C"].isDefault, True)
        self.failUnlessEqual(simple.params["C"].equalDefault, True)

        simple.C = 10.0
        self.failUnlessEqual(simple.params.isSet("C"), True)
        self.failUnlessEqual(simple.params.isSet(), True)
        self.failUnlessEqual(simple.params["C"].isDefault, False)
        self.failUnlessEqual(simple.params["C"].equalDefault, False)

        self.failUnlessEqual(simple.C, 10.0)
        simple.params["C"].resetvalue()
        self.failUnlessEqual(simple.params.isSet("C"), True)
        # TODO: Test if we 'train' a classifier f we get isSet to false
        self.failUnlessEqual(simple.C, 1.0)
        self.failUnlessRaises(AttributeError, simple.params.__getattribute__, 'B')

    def testMixed(self):
        mixed  = MixedClass()

        self.failUnlessEqual(len(mixed.params.items), 2)
        self.failUnlessEqual(len(mixed.states.items), 1)
        self.failUnlessRaises(AttributeError, mixed.__getattribute__, 'kernel_params')

        self.failUnlessEqual(mixed.C, 1.0)
        self.failUnlessEqual(mixed.params.isSet("C"), False)
        self.failUnlessEqual(mixed.params.isSet(), False)
        mixed.C = 10.0
        self.failUnlessEqual(mixed.params.isSet("C"), True)
        self.failUnlessEqual(mixed.params.isSet("D"), False)
        self.failUnlessEqual(mixed.params.isSet(), True)
        self.failUnlessEqual(mixed.D, 3.0)


    def testClassifier(self):
        clf  = ParametrizedClassifier()
        self.failUnlessEqual(len(clf.params.items), 3) # + regression/retrainable
        self.failUnlessEqual(len(clf.kernel_params.items), 1)

        clfe  = ParametrizedClassifierExtended()
        self.failUnlessEqual(len(clfe.params.items), 3)
        self.failUnlessEqual(len(clfe.kernel_params.items), 2)
        self.failUnlessEqual(len(clfe.kernel_params.listing), 2)

        # check assignment once again
        self.failUnlessEqual(clfe.kp2, 200.0)
        clfe.kp2 = 201.0
        self.failUnlessEqual(clfe.kp2, 201.0)
        self.failUnlessEqual(clfe.kernel_params.isSet("kp2"), True)
        clfe.train(Dataset(samples=[[0,0]], labels=[1], chunks=[1]))
        self.failUnlessEqual(clfe.kernel_params.isSet("kp2"), False)
        self.failUnlessEqual(clfe.kernel_params.isSet(), False)
        self.failUnlessEqual(clfe.params.isSet(), False)

def suite():
    return unittest.makeSuite(ParamsTests)


if __name__ == '__main__':
    import runner

