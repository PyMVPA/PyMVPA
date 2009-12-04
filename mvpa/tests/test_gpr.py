# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA GPR."""

from tests_warehouse import *

class GPRTests(unittest.TestCase):

    def testBasic(self):
        pass

    def testLinear(self):
        pass


def suite():
    return unittest.makeSuite(GPRTests)


if __name__ == '__main__':
    import runner
