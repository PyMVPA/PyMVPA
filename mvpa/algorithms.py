### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Various algorithms
#
#    Copyright (C) 2006-2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2 of the License, or (at your option) any later version.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


import numpy

def SpheresInVolume(vol, radius, voxelsize=(1,1,1), forcesphere = False):
    """Generates all possible spheres with a fixed radius within a 3d volume.

    SearchSpheres( 3dmask, radius [, (x,y,z)] ) -> SearchSpheres object

    Each voxel with a non-zero value in 'vol' becomes the center of one sphere.
    Each sphere contains all voxels with a distance less-or-equal than that
    given by 'radius' and a non-zero value in 'vol'. 'vol' has to be a
    three-dimensional array.

    The radius can be specified in an arbitrary unit and 'voxelsize' is used
    to map it to <number of voxels>.

    Instead of returning all spheres as a list, this function return a
    generator object the can be used for looping. This is much more memory
    efficient (like range() and xrange()).

    Each call of the generator object returns a tuple of three arrays
    containing the coordinates of all voxels in the sphere (analog to the
    return value of numpy.nonzero()).

    By default each sphere will only contain voxels that are also part of the
    mask. By setting forcesphere to True the spheres will contain all voxel in
    range regardless of their status in the mask instead.
    """
    if not len(vol.shape) == 3:
        raise ValueError, "'vol' must be a 3d array."

    vs_trans = numpy.array([voxelsize[2], voxelsize[1], voxelsize[0]])

    # square the radius once here and later compare
    # squared distances instead of endless sqrt() calls
    sq_radius = radius**2

    # loop over all matrix elements (voxels)
    for z in xrange(vol.shape[0]):
        for y in xrange(vol.shape[1]):
            for x in xrange(vol.shape[2]):
                # sphere center
                coord = numpy.array([z,y,x])
                # consider real voxelsizes
                coord_trans = coord * vs_trans

                # list of voxel coords in sphere
                vlistx = []
                vlisty = []
                vlistz = []

                # only consider voxel if non-zero in mask
                if not vol[z,y,x] == 0:
                    # determine sphere elements
                    # only search in a minimal cube surrounding the sphere
                    # note: add one because xrange does not include the stop value itself
                    for sz in xrange( coord[0] - int( round( radius/voxelsize[0] ) ),
                                    coord[0] + int( round( radius/voxelsize[0] ) + 1 ) ):
                        for sy in xrange( coord[1] - int( round( radius/voxelsize[1] ) ),
                                        coord[1] + int( round( radius/voxelsize[1] ) + 1 ) ):
                            for sx in xrange( coord[2] - int( round( radius/voxelsize[2] ) ),
                                            coord[2] + int( round( radius/voxelsize[2] ) + 1 ) ):
                                # voxel coord potentially in sphere
                                scoord = numpy.array([sz,sy,sx])

                                # if this scoord is not outside the volume
                                if ( not numpy.sum( scoord < 0 )
                                    and not numpy.sum( scoord >= vol.shape ) ):
                                    # and only if the volume mask is not zero at this point
                                    if not vol[sz,sy,sx] == 0.0 or forcesphere:
                                        # and if distance from center is less than sphere radius
                                        # note: comparing squared values (using real voxelsizes)
                                        if sq_radius >= numpy.sum((coord_trans-scoord*vs_trans)**2):
                                            vlistx.append(sx)
                                            vlisty.append(sy)
                                            vlistz.append(sz)

                    yield ( numpy.array(vlistz),
                            numpy.array(vlisty),
                            numpy.array(vlistx) )

