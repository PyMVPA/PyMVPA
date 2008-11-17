#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test some base functionality which did not make it into a separate unittests"""

import unittest
import os.path
import numpy as N

from mvpa.base import externals


class TestBases(unittest.TestCase):

    def testExternals(self):
        self.failUnlessRaises(ValueError, externals.exists, 'BoGuS')

def suite():
    return unittest.makeSuite(TestBases)


if __name__ == '__main__':
    import runner

