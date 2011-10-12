# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA args helpers"""

import unittest
from mvpa2.misc.args import *

if __debug__:
    from mvpa2.base import debug

class ArgsHelpersTest(unittest.TestCase):

    def test_basic(self):
        """Test if we are not missing basic parts"""
        kwargs = {'a':1, 'slave_a':3, 'slave_z':4, 'slave_slave_z':5, 'c':3}

        res = split_kwargs(kwargs, ['slave_'])
        self.failUnless(res.has_key('slave_') and res.has_key(''))
        self.failUnless(res['slave_'] == {'a':3, 'z':4, 'slave_z':5})
        self.failUnless(res[''] == {'a':1, 'c':3})

        res = split_kwargs(kwargs)
        self.failUnless(res.keys() == [''])
        self.failUnless(res[''] == kwargs)


    def test_decorator(self):
        """Test the group_kwargs decorator"""

        selftop = self

        class C1(object):

            @group_kwargs(prefixes=['slave_'], assign=True)
            def __init__(self, **kwargs):
                selftop.failUnless(hasattr(self, '_slave_kwargs'))
                self.method_passedempty()
                self.method_passed(1, custom_p1=144, bugax=1)
                self.method_filtered(1, custom_p1=123)

            @group_kwargs(prefixes=['custom_'], passthrough=True)
            def method_passedempty(self, **kwargs):
                # we must have it even though it is empty
                selftop.failUnless('custom_kwargs' in kwargs)

            @group_kwargs(prefixes=['custom_', 'buga'], passthrough=True)
            def method_passed(self, a, custom_kwargs, bugakwargs, **kwargs):
                # we must have it even though it is empty
                selftop.failUnless(custom_kwargs == {'p1':144})
                selftop.failUnless(bugakwargs == {'x':1})
                selftop.failUnless(not hasattr(self, '_custom_kwargs'))

            @group_kwargs(prefixes=['custom_'])
            def method_filtered(self, a, **kwargs):
                # we must have it even though it is empty
                selftop.failUnlessEqual(a, 1)
                selftop.failUnless(not 'custom_kwargs' in kwargs)

            def method(self):
                return 123

            @group_kwargs(prefixes=['xxx'])
            def method_decorated(self):
                return 124

        c1 = C1(slave_p1=1, p1=2)
        self.failUnless(c1.method() == 123)
        self.failUnless(c1.method_decorated() == 124)


def suite():
    return unittest.makeSuite(ArgsHelpersTest)


if __name__ == '__main__':
    import runner

