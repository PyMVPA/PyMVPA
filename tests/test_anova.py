#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA univariate ANOVA sensitivity analyzer."""

from mvpa.measures.anova import OneWayAnova
from tests_warehouse import *


class ANOVATests(unittest.TestCase):

    def testANOVA(self):
        data = datasets['dumb']
        aov = OneWayAnova()

        # compute f-scores
        f = aov(data)

        self.failUnless(f.shape == (2,))
        self.failUnless(f[1] == 0.0)
        self.failUnless(f[0] > 0.0)


def suite():
    return unittest.makeSuite(ANOVATests)


if __name__ == '__main__':
    import runner

