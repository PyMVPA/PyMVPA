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
        minimal = [ coords
            for center, coords in mvpa.SpheresInMask(
                    numpy.ones((1,1,1)), 1) ]

        # only one sphere possible
        self.failUnless( len(minimal) == 1 )
        # check sphere
        self.failUnless( (minimal[0][0] == 0).all() )
        self.failUnless( (minimal[0][1] == 0).all() )
        self.failUnless( (minimal[0][2] == 0).all() )

        # make bigger 3d mask
        three_mask = numpy.ones((3,3,3))
        # disable a single voxel
        three_mask[1,1,1] = 0

        # get the spheres
        three = [ (center, coords)
            for center, coords in mvpa.SpheresInMask(three_mask, 1) ]
        # check we have all but one
        self.failUnless( len(three) == 26 )

        # first sphere contains 4 voxels
        self.failUnless( len(three[0][1][0]) == 4 )

        # middle suface sphere contains one less (center voxel)
        self.failUnless( len(three[4][1][0]) == 5 )


        s = [ (c,sc) for c,sc in \
            mvpa.algorithms.SpheresInMask( numpy.ones((2,2,2)),
                                           0.9,
                                           elementsize=(1,1,1),
                                           forcesphere=False ) ]
        # check for all possible sphere centers
        self.failUnless( len(s) == 8 )
        for coord in s:
            # center has to be in 3d
            self.failUnless( len(coord[0]) == 3 )
            # list of voxels as well
            self.failUnless( len(coord[1]) == 3 )
            for v in coord[1]:
                # only one voxel must be in sphere
                self.failUnless( len(v) == 1 )
            # the sphere center is the only voxel
            self.failUnless( ( coord[0] == ( coord[1][0][0],
                                             coord[1][1][0],
                                             coord[1][2][0] ) ).all() )



def suite():
    return unittest.makeSuite(AlgorithmTests)


if __name__ == '__main__':
    unittest.main()

