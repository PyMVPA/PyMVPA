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

def SpheresInMask( mask, radius, elementsize=None, forcesphere = False):
    """Generates all possible spheres with a fixed radius within a mask.

    Each element with a non-zero value in 'mask' becomes the center of one
    sphere. Each sphere contains all elements with a distance less-or-equal
    than that given by 'radius' and a non-zero value in 'mask' (if
    'forcesphere' isn't enabled). 'mask' can have any number of dimensions.

    The radius can be specified in an arbitrary unit and 'elementsize' is used
    to map it to the mask elements. 'elementsize' must contain the extend of
    a mask element along all mask dimensions.

    Instead of returning all spheres as a list, this function return a
    generator object the can be used for looping (like xrange() instead of 
    range()).

    Each call of the generator object returns a tuple of the center coordinates
    and a tuple of three arrays containing the coordinates of all elements in
    the sphere (analog to the return value of numpy.nonzero()).

    By default each sphere will only contain elements that are also part of the
    mask. By setting 'forcesphere' to True the spheres will contain all element
    in range regardless of their status in the mask instead.
    """
    if elementsize == None:
        elementsize = numpy.ones( len( mask.shape ) )
    else:
        elementsize = numpy.array( elementsize )

    # compute radius in units of elementsize per axis
    elementradius_per_axis = float(radius) / elementsize

    # build prototype sphere
    filter = numpy.ones( ( numpy.ceil( elementradius_per_axis ) * 2 ) + 1 )
    filter_center = numpy.array( filter.shape ) / 2

    # now zero out all too distant elements
    f_coords = numpy.transpose( filter.nonzero() )
    # check all filter element
    for coord in f_coords:
        # scale coordinates by elementsize (and de-mean)
        trans_coord = (coord - filter_center) * elementsize

        # compare with radius
        if radius < numpy.linalg.norm( trans_coord ):
            # zero everything that is too distant
            filter[numpy.array(coord, ndmin=2).T.tolist()] = 0

    # convert spherical filter into releative coordinates
    filter = numpy.array( filter.nonzero() ).T - filter_center

    # get the nonzero mask coordinates
    coords = numpy.transpose( mask.nonzero() )

    # for all nonzero mask elements (a.k.a. sphere centers)
    for center in coords:
        # make abs sphere mask
        abs_sphere = center + filter

        # check if mask elements are outside of mask
        abs_sphere = [ v for v in abs_sphere \
                        if (v >= numpy.zeros(len( mask.shape ) ) ).all() \
                        and (v < mask.shape).all() ]

        # exclude nonzero mask elements if not requested otherwise
        if not forcesphere:
            abs_sphere = [ v for v in abs_sphere \
                        if mask[numpy.array(v, ndmin=2).T.tolist()] ]

        yield center, numpy.transpose( abs_sphere )
