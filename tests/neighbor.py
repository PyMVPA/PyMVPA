#emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Unit tests for PyMVPA finders
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
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from mvpa.neighbor import *
import unittest
import numpy as N

class NeighborFinderTests(unittest.TestCase):

    def testDistances(self):
        a = N.array([3,8])
        b = N.array([6,4])
        # test distances or yarik recalls unit testing ;)
        self.failUnless( cartesianDistance(a, b) == 5.0 )
        self.failUnless( manhattenDistance(a, b) == 7 )
        self.failUnless( absminDistance(a, b) == 4 )


    def testDescreteNeighborFinder(self):
		# who said that we will not use FSL's data
		# with negative dimensions? :-)
		elsize = [-2.5, 1.5]
        distance = 3

		# use default function
		finder = DescreteNeighborFinder(elsize)

        # simple check
        target = N.array([ [1,2], [2,1], [2,2], [2,3], [3,2] ])
        self.failUnless( (finder([2,2], 2.6) == target).all())

        # a bit longer one... not sure what for
        for point in finder([2,2], distance):
            self.failUnless( cartesianDistance(point, [2,2]) <= distance)

        # use manhattenDistance function
		finder = DescreteNeighborFinder(elsize, manhattenDistance)
        for point in finder([2,2], distance):
            self.failUnless( manhattenDistance(point, [2,2]) <= distance)


def suite():
    return unittest.makeSuite(NeighborFinderTests)


if __name__ == '__main__':
    unittest.main()

