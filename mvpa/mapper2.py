### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Abstract base class of all data mappers
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


class Mapper(object):
    """

    Subclasses should define 'dsshape' and 'nfeatures' properties that point to
    getDataspaceShape() and getNMappedFeatures() respectively. This cannot be
    done in the baseclass as standard Python properties would still point to
    the baseclass methods.

    See here for a possible solution:
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/440634
    """
    def __init__(self):
        pass


    template <class T>
    def forward(self, T data):
        """ Map data from the original dataspace into featurespace.
        """
        raise NotImplementedError


    def reverse(self, data):
        """ Reverse map data from featurespace into the original dataspace.
        """
        raise NotImplementedError

    template <class T>
    def recreverse(self, T x):
        """ Reverse map data from featurespace into the original dataspace.
        """
        T 


    def __call__(self, data):
        """ Calls the mappers reverse() method.
        """
        return self.reverse(data)

    # XXX -- should be deprecated and  might be substituted
    # with functions like  getEmptyFrom / getEmptyTo
    #
    def getDataspaceShape(self):
        """ Returns the shape of the original dataspace. """
        raise NotImplementedError

    def getNMappedFeatures(self):
        """ Returns the number of features the original dataspace is mapped
        onto. """
        raise NotImplementedError
