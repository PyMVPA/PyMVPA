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
from mvpa2.base.constraints import *
from mvpa2.testing.clfs import *

class ParametrizedClassifier(SameSignClassifier):
    p1 = Parameter(1.0, constraints='float')
    kp1 = KernelParameter(100.0)

class ParametrizedClassifierExtended(ParametrizedClassifier):
    def __init__(self):
        ParametrizedClassifier.__init__(self)
        self.kernel_params['kp2'] = \
            KernelParameter(200.0, doc="Very useful param")


class ChoiceClass(ClassWithCollections):
    C = Parameter('choice1',
                  constraints=EnsureChoice('choice1', 'choice2'),
                  doc="documentation")

class BlankClass(ClassWithCollections):
    pass

class SimpleClass(ClassWithCollections):
    C = Parameter(1.0, 
                  constraints=Constraints(EnsureFloat(),
                                          EnsureRange(min=0.0, max=10.0)),
                  doc="C parameter")

class MixedClass(ClassWithCollections):
    C = Parameter(1.0, constraints=EnsureRange(min=0), doc="C parameter")
    D = Parameter(3.0, constraints=EnsureRange(min=0), doc="D parameter")
    state1 = ConditionalAttribute(doc="bogus")

class ParamsTests(unittest.TestCase):

    def test_blank(self):
        blank  = BlankClass()

        self.assertRaises(AttributeError, blank.__getattribute__, 'ca')
        self.assertRaises(AttributeError, blank.__getattribute__, '')

    def test_choice(self):
        c = ChoiceClass()
        self.assertRaises(ValueError, c.params.__setattr__, 'C', 'bu')

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

        # set int but get float
        simple.params.C = 10
        self.assertTrue(isinstance(simple.params.C, float))
        # wrong type causes exception
        self.assertRaises(ValueError, simple.params.__setattr__, 'C', 'somestr')
        # value < min causes exception
        self.assertRaises(ValueError, simple.params.__setattr__, 'C', -123.4)
        # value > max causes exception
        self.assertRaises(ValueError, simple.params.__setattr__, 'C', 123.4)


        # check for presence of the constraints description
        self.assertTrue(simple._paramsdoc[0][1].find('Constraints: ') > 0)

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

    def test_choices(self):
        # Test doc strings for parameters with choices
        class WithChoices(ClassWithCollections):
            C = Parameter('choice1',
                  constraints=EnsureChoice('choice1', 'choice2'),
                  doc="documentation")
            # We need __init__ to get 'custom' docstring
            def __init__(self, **kwargs):
                super(type(self), self).__init__(**kwargs)
  
        c = WithChoices()
        self.assertRaises(ValueError, c.params.__setattr__, 'C', 'bu')
        c__doc__ = c.__init__.__doc__.replace('"', "'")
        # Will currently fail due to unfixed _paramdoc of Parameter class 
        #self.assertTrue('choice2' in c__doc__)
        #self.assertTrue("(Default: 'choice1')" in c__doc__)

        # But we will not (at least for now) list choices if there are
        # non-strings
        class WithFuncChoices(ClassWithCollections):
            C = Parameter('choice1',
                          constraints=EnsureChoice('choice1', np.sum),
                          doc="documentation")
            # We need __init__ to get 'custom' docstring
            def __init__(self, **kwargs):
                super(type(self), self).__init__(**kwargs)

        cf = WithFuncChoices()
        self.assertRaises(ValueError, cf.params.__setattr__, 'C', 'bu')
        cf.params.C = np.sum
        cf__doc__ = cf.__init__.__doc__.replace('"', "'")
        # Will currently fail due to unfixed _paramdoc of Parameter class 
        #self.assertTrue('choice2' in c__doc__)
        #self.assertTrue("(Default: 'choice1')" in c__doc__)        
        #self.assertTrue("(Default: 'choice1')" in cf__doc__)

    def test_simple_specs(self):
        p = Parameter(1.0, constraints='int')
        self.assertTrue(p.value is 1)
        self.assertTrue(p.constraints is constraint_spec_map['int'])
        self.assertRaises(ValueError, Parameter, 'a', constraints='int')
        self.assertRaises(ValueError, Parameter, 1.0, constraints='str')


def suite():  # pragma: no cover
    return unittest.makeSuite(ParamsTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

