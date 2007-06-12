### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Implementation of the Searchlight algorithm
#
#    Copyright (C) 2007 by
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

import algorithms


class Searchlight( object ):
    def __init__( self, pattern, mask, classifier, **kwargs ):
        self.__pattern = pattern
        self.__mask = mask
        self.__clf = classifier
        self.__clfargs = kwargs

        if not mask.shape == pattern.origshape:
            raise ValueError, 'Mask shape has to match the pattern origshape.'

    def run( classifier = None, **kwargs ):
        # accept new classifier if any
        if classifier:
            self.__clf = classifier
            self.__clfargs = kwargs

        for sphere in algorithms.SpheresInVolume( self.__mask,
                                                  self.__radius,
                                                  self.__voxelsize,
                                                  self.__forcesphere ):
            print sphere

