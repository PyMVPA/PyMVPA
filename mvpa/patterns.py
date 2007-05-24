### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#
#    Multivariate pattern analysis
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
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

import numpy


def packPatterns( patterns, regs, origins = None ):
    """ Pack information about pattern data, regressor value and pattern origin
    into a single datastructure.

    This function can be useful if a data pattern has to be associated with
    some properties (regressor value and origin of the pattern). This might be
    necessary if a function has to be applied to a list of patterns without
    loosing the association.

    This function takes sequences of patterns, regressor values and origin
    labels. All sequences have to be of equal length. The sequences have to be
    ordered in the way that patterns[i], regs[i] and origins[i] contain data,
    regressor and origin of pattern i.

    While the regressor associates patterns with certain conditions the origins
    can be used to mark patterns to be structural similiar e.g. recorded during
    the same session. If no origin value is specified each pattern will get a
    unique origin label.

    This function returns a sequence of 3-tupels (pattern, reg, origin).

    Please see the unpackPatterns() that reverts this procedure.
    """
    # if no pattern origins are specified let every pattern be in it's own
    if origins == None:
        origins = range( len( patterns ) )

    if not ( len( patterns ) == len( regs ) == len( origins ) ):
        raise ValueError, "All sequences have to be of equal length."

    return zip( patterns, regs, origins )


def unpackPatterns( pack ):
    """ Revert the procedure of packPatterns().
    """

    return [ p[0] for p in pack ], \
           [ p[1] for p in pack ], \
           [ p[2] for p in pack ]


def selectFeatures(source, mask = None):
    """Uses all non-zero elements of a 3d mask volume to select
    data elements in all volumes of a 4d timeseries ('input').

    Returns a 2d array ( volumes x <number of non-zeros in mask> ).
    """
    # make sure input is 5d
    data = source.reshape( tuple([ 1 for i in range( 5 - len(source.shape) ) ]) + source.shape )

    # use everything if there is no mask
    if mask == None:
        mask = numpy.ones(source.shape[-3:])

    if isinstance(mask, numpy.ndarray):
        if not len(mask.shape) == 3:
            raise ValueError, "'mask' array must be 3d."
        # tuple of arrays containing the indexes of all nonzero elements
        # of the mask
        nz = mask.nonzero()
    elif isinstance(mask, tuple):
        # mask already contains the nonzero coordinates
        nz = mask
    else:
        raise ValueError, "'mask' has to be either a 3d array or a 3-tuple of index array (like those returned by array.nonzero())"

    # how many nonzeros do we have
    nzsize = len(nz[0])

    # create the output array with desired size
    # make space for all non-zeros in each step in the 5th dimension
    # for all volumes (data.shape[1])
    selected = numpy.zeros( (data.shape[1], data.shape[0] * nzsize), dtype=source.dtype)

    # for all datasets (5th dim)
    for d in xrange(data.shape[0]):
        # for all volumes in the timeseries
        for v in xrange(data.shape[1]):
            # store the nonzero elements of all input volumes
            selected[v,d*nzsize:d*nzsize+nzsize] = data[d,v][nz]

    return selected
