### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#
#    Unit tests for PyMVPA stats helpers
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the MIT License.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the COPYING
#    file that comes with this package for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

import mvpa.stats as ms
import unittest
import numpy as N

class StatsTests(unittest.TestCase):

    def testChiSquare(self):
        # test equal distribution
        tbl = N.array([[5,5],[5,5]])
        chi, p = ms.chisquare( tbl )
        self.failUnless( chi == 0.0 )
        self.failUnless( p == 1.0 )

        # test non-equal distribution
        tbl = N.array([[4,0],[0,4]])
        chi, p = ms.chisquare( tbl )
        self.failUnless( chi == 8.0 )
        self.failUnless( p < 0.05 )


def suite():
    return unittest.makeSuite(StatsTests)


if __name__ == '__main__':
    unittest.main()

