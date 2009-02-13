# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA dochelpers"""

from mvpa.base.dochelpers import singleOrPlural

import unittest
import numpy as N

class DochelpersTests(unittest.TestCase):

    def testBasic(self):
        self.failUnlessEqual(singleOrPlural('a', 'b', 1), 'a')
        self.failUnlessEqual(singleOrPlural('a', 'b', 0), 'b')
        self.failUnlessEqual(singleOrPlural('a', 'b', 123), 'b')

    # TODO: more unittests
def suite():
    return unittest.makeSuite(DochelpersTests)


if __name__ == '__main__':
    import runner

