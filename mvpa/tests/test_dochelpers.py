# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA dochelpers"""

from mvpa.base.dochelpers import single_or_plural, borrowdoc

import unittest
import numpy as N

class DochelpersTests(unittest.TestCase):

    def testBasic(self):
        self.failUnlessEqual(single_or_plural('a', 'b', 1), 'a')
        self.failUnlessEqual(single_or_plural('a', 'b', 0), 'b')
        self.failUnlessEqual(single_or_plural('a', 'b', 123), 'b')

    def testBorrowDoc(self):

        class A(object):
            def met1(self):
                """met1doc"""
                pass
            def met2(self):
                """met2doc"""
                pass

        class B(object):
            @borrowdoc(A)
            def met1(self):
                pass
            @borrowdoc(A, 'met1')
            def met2(self):
                pass

        self.failUnlessEqual(B.met1.__doc__, A.met1.__doc__)
        self.failUnlessEqual(B.met2.__doc__, A.met1.__doc__)

# TODO: more unittests
def suite():
    return unittest.makeSuite(DochelpersTests)


if __name__ == '__main__':
    import runner

