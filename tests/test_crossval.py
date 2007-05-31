### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#
#    Unit tests for PyMVPA pattern handling
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    version 2 of the License, or (at your option) any later version.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

import mvpa
import unittest
import numpy

class CrossValidationTests(unittest.TestCase):

    def testMofNCombinations(self):
        self.failUnlessEqual( 
            mvpa.getLengthNCombinations( range(3), 1 ), [[0],[1],[2]] )
        self.failUnlessEqual( 
            mvpa.getLengthNCombinations( 
                        range(4), 2 ), 
                        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]] 
                        )
        self.failUnlessEqual( 
            mvpa.getLengthNCombinations( 
                        range(4), 3 ), [[0, 1, 2], [0, 1, 3], [0, 2, 3]] )


def suite():
    return unittest.makeSuite(CrossValidationTests)


if __name__ == '__main__':
    unittest.main()

