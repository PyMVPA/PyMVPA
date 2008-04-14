#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA stats helpers"""

import unittest

import numpy as N

from mvpa.misc.stats import chisquare


class StatsTests(unittest.TestCase):

    def testChiSquare(self):
        # test equal distribution
        tbl = N.array([[5,5],[5,5]])
        chi, p = chisquare(tbl)
        self.failUnless( chi == 0.0 )
        self.failUnless( p == 1.0 )

        # test non-equal distribution
        tbl = N.array([[4,0],[0,4]])
        chi, p = chisquare(tbl)
        self.failUnless(chi == 8.0)
        self.failUnless(p < 0.05)


def suite():
    return unittest.makeSuite(StatsTests)


if __name__ == '__main__':
    import runner

