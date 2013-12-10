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

import numpy as np

from mvpa2.datasets.base import dataset_wizard
from mvpa2.base.state import ClassWithCollections, ConditionalAttribute
from mvpa2.base.param import Parameter, KernelParameter

from mvpa2.testing.clfs import *

class ParametrizedClassifier(SameSignClassifier):
    p1 = Parameter(1.0)
    kp1 = KernelParameter(100.0)

class ParametrizedClassifierExtended(ParametrizedClassifier):
    def __init__(self):
        ParametrizedClassifier.__init__(self)
        self.kernel_params['kp2'] = \
            KernelParameter(200.0, doc="Very useful param")

class BlankClass(ClassWithCollections):
    pass

class SimpleClass(ClassWithCollections):
    C = Parameter(1.0, min=0, doc="C parameter")

class MixedClass(ClassWithCollections):
    C = Parameter(1.0, min=0, doc="C parameter")
    D = Parameter(3.0, min=0, doc="D parameter")
    state1 = ConditionalAttribute(doc="bogus")

class ParamsTests(unittest.TestCase):

    def test_blank(self):
        blank  = BlankClass()

        self.assertRaises(AttributeError, blank.__getattribute__, 'ca')
        self.assertRaises(AttributeError, blank.__getattribute__, '')

    def test_simple(self):
        simple  = SimpleClass()

        self.assertEqual(len(simple.params.items()), 1)
        self.assertRaises(AttributeError, simple.__getattribute__, 'dummy')
        self.assertRaises(AttributeError, simple.__getattribute__, '')

        self.assertEqual(simple.params.C, 1.0)
        self.assertEqual(simple.params.is_set("C"), False)
        self.assertEqual(simple.params.is_set(), False)
        self.assertEqual(simple.params["C"].is_default, True)
        self.assertEqual(simple.params["C"].equal_default, True)

        simple.params.C = 1.0
        # we are not actually setting the value if == default
        self.assertEqual(simple.params["C"].is_default, True)
        self.assertEqual(simple.params["C"].equal_default, True)

        simple.params.C = 10.0
        self.assertEqual(simple.params.is_set("C"), True)
        self.assertEqual(simple.params.is_set(), True)
        self.assertEqual(simple.params["C"].is_default, False)
        self.assertEqual(simple.params["C"].equal_default, False)

        self.assertEqual(simple.params.C, 10.0)
        simple.params["C"].reset_value()
        self.assertEqual(simple.params.is_set("C"), True)
        # TODO: Test if we 'train' a classifier f we get is_set to false
        self.assertEqual(simple.params.C, 1.0)
        self.assertRaises(AttributeError, simple.params.__getattribute__, 'B')

    def test_mixed(self):
        mixed  = MixedClass()

        self.assertEqual(len(mixed.params.items()), 2)
        self.assertEqual(len(mixed.ca.items()), 1)
        self.assertRaises(AttributeError, mixed.__getattribute__, 'kernel_params')

        self.assertEqual(mixed.params.C, 1.0)
        self.assertEqual(mixed.params.is_set("C"), False)
        self.assertEqual(mixed.params.is_set(), False)
        mixed.params.C = 10.0
        self.assertEqual(mixed.params.is_set("C"), True)
        self.assertEqual(mixed.params.is_set("D"), False)
        self.assertEqual(mixed.params.is_set(), True)
        self.assertEqual(mixed.params.D, 3.0)


    def test_classifier(self):
        clf  = ParametrizedClassifier()
        self.assertEqual(len(clf.params.items()), 2) # + retrainable
        self.assertEqual(len(clf.kernel_params.items()), 1)

        clfe  = ParametrizedClassifierExtended()
        self.assertEqual(len(clfe.params.items()), 2)
        self.assertEqual(len(clfe.kernel_params.items()), 2)
        self.assertEqual(len(clfe.kernel_params.listing), 2)

        # check assignment once again
        self.assertEqual(clfe.kernel_params.kp2, 200.0)
        clfe.kernel_params.kp2 = 201.0
        self.assertEqual(clfe.kernel_params.kp2, 201.0)
        self.assertEqual(clfe.kernel_params.is_set("kp2"), True)
        clfe.train(dataset_wizard(samples=[[0,0]], targets=[1], chunks=[1]))
        self.assertEqual(clfe.kernel_params.is_set("kp2"), False)
        self.assertEqual(clfe.kernel_params.is_set(), False)
        self.assertEqual(clfe.params.is_set(), False)

    def test_incorrect_parameter_error(self):
        # Just a sample class
        from mvpa2.generators.partition import NFoldPartitioner
        try:
            spl = NFoldPartitioner(1, incorrect=None)
            raise AssertionError("Must have failed with an exception here "
                                 "due to incorrect parameter")
        except Exception, e:
            estr = str(e)
        self.assertTrue(not "calling_time" in estr,
             msg="must give valid parameters for partitioner, "
                 "not .ca's. Got: \n\t%r" % estr)
        # sample parameters which should be present
        for p in 'count', 'disable_ca', 'postproc':
            self.assertTrue(p in estr)


def suite():
    return unittest.makeSuite(ParamsTests)


if __name__ == '__main__':
    import runner
    runner.run()

