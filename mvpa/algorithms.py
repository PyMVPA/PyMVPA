### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Various algorithms
#
#    Copyright (C) 2006-2007 by
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


import numpy as N

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
    and a boolean NumPy array matching the size of the mask with all sphere
    elements set to true (rest is false).

    ATTENTION: The return spheremask array is re-used in every iteration.
               If one only stores a reference of this array its content will
               change when the next iteration is performed! If one want to
               store the spheremask they have to be copied.

    By default each sphere will only contain elements that are also part of the
    mask. By setting 'forcesphere' to True the spheres will contain all element
    in range regardless of their status in the mask instead.
    """
    if elementsize == None:
        elementsize = N.ones( len( mask.shape ), dtype='uint' )
    else:
        elementsize = N.array( elementsize )

    # compute radius in units of elementsize per axis
    elementradius_per_axis = float(radius) / elementsize

    # build prototype sphere
    filter = N.ones( ( N.ceil( elementradius_per_axis ) * 2 ) + 1 )
    filter_center = N.array( filter.shape ) / 2

    # now zero out all too distant elements
    f_coords = N.transpose( filter.nonzero() )
    # check all filter element
    for coord in f_coords:
        # scale coordinates by elementsize (and de-mean)
        trans_coord = ((coord - filter_center) * elementsize) - elementsize/2

        # compare with radius
        if radius < N.linalg.norm( trans_coord ):
            # zero everything that is too distant
            filter[N.array(coord, ndmin=2).T.tolist()] = 0

    # convert spherical filter into releative coordinates
    filter = N.array( filter.nonzero() ).T - filter_center
    # get the nonzero mask coordinates
    coords = N.transpose( mask.nonzero() )

    # the absolute sphere mask (gets recycled everytime)
    spheremask = N.zeros(mask.shape, dtype='bool')
    # for all nonzero mask elements (a.k.a. sphere centers)
    for center in coords:
        # make abs sphere mask
        abs_sphere = center + filter

        # check if mask elements are outside of mask
        abs_sphere = [ v for v in abs_sphere \
                        if (v >= N.zeros(len( mask.shape ) ) ).all() \
                        and (v < mask.shape).all() ]

        # exclude nonzero mask elements if not requested otherwise
        if not forcesphere:
            abs_sphere = [ v for v in abs_sphere \
                        if mask[N.array(v, ndmin=2).T.tolist()] ]

        # reset the mask
        spheremask[:] = False

        # put current sphere into the mask
        spheremask[tuple( N.transpose( abs_sphere ) ) ] = True

        yield center, spheremask
