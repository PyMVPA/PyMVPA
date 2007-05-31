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

class AlgorithmTests(unittest.TestCase):

    def testSphereGenerator(self):
        minimal = [ coord
            for coord in mvpa.SpheresInVolume(numpy.ones((1,1,1)), 1) ]

        # only one sphere possible
        self.failUnless( len(minimal) == 1 )
        # check sphere
        self.failUnless( (minimal[0][0] == 0).all() )
        self.failUnless( (minimal[0][1] == 0).all() )
        self.failUnless( (minimal[0][2] == 0).all() )


def suite():
    return unittest.makeSuite(AlgorithmTests)


if __name__ == '__main__':
    unittest.main()

