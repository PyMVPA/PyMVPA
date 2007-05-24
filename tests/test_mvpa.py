### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#
#    Unit tests for PyMVPA
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

class MVPATests(unittest.TestCase):

    def testSelectFeatures(self):
        # make random 4d data set
        data = numpy.random.standard_normal( (10,2,3,4) )
        # make full 3d mask
        mask = numpy.ones( (2,3,4) )
        # check 4d -> 2d with 3d mask
        selected = mvpa.selectFeatures( data, mask )
        self.failUnlessEqual( selected.shape, (10, 24) )

        # make random 2d data set
        data = numpy.random.standard_normal( (10,5) )
        # make full 1d mask
        mask = numpy.ones( (5) )
        # check 2d -> 2d with 1d mask
        selected = mvpa.selectFeatures( data, mask )
        self.failUnlessEqual( selected.shape, (10, 5) )


def suite():
    return unittest.makeSuite(MVPATests)


if __name__ == '__main__':
    unittest.main()

