#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Abstract base class of all data mappers
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
"""PyMVPA: Module that contains abstract base class of all data mappers"""

class Mapper(object):
    """
    Interface to provide mapping between two spaces: in and out.
    Methods are prefixed correspondingly. forward/reverse operate
    on the entire dataset. get(In|Out)Id[s] operate per element.
          forward
    in   ---------> out
         <--------/
           reverse

    Subclasses should define 'dsshape' and 'nfeatures' properties that point to
    getInShape() and getOutSize() respectively. This cannot be
    done in the baseclass as standard Python properties would still point to
    the baseclass methods.

    See here for a possible solution:
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/440634
    """
    def __init__(self):
        pass

    def forward(self, data):
        """ Map data from the original dataspace into featurespace.
        """
        raise NotImplementedError

    def __getitem__(self, data):
        """ Calls the mappers forward() method.
        """
        return self.reverse(data)

    def reverse(self, data):
        """ Reverse map data from featurespace into the original dataspace.
        """
        raise NotImplementedError

    def __call__(self, data):
        """ Calls the mappers reverse() method.
        """
        return self.reverse(data)

    def getInShape(self):
        """ Returns the dimensionality speicification of the original dataspace.

        XXX -- should be deprecated and  might be substituted
        with functions like  getEmptyFrom / getEmptyTo
        """
        raise NotImplementedError

    def getOutShape(self):
        """
        Returns the shape (or other dimensionality speicification)
        of the destination dataspace.
        """
        raise NotImplementedError


    def getInSize(self):
        """ Returns the size of the entity in input space """
        raise NotImplementedError

    def getOutSize(self):
        """ Returns the size of the entity in output space """
        raise NotImplementedError

# comment out for now... introduce when needed
#    def getInEmpty(self):
#        """ Returns empty instance of input object """
#        raise NotImplementedError
#
#    def getOutEmpty(self):
#        """ Returns empty instance of output object """
#        raise NotImplementedError

    def getInId(self, outId):
        """For a given Id in "out" returns corresponding "in" Id """
        raise NotImplementedError

    def getInIds(self):
        """Returns corresponding "in" Ids """
        raise NotImplementedError

    def getOutId(self, inId):
        """Returns corresponding "out" Id """
        raise NotImplementedError


### yoh: To think about generalization
##
## getMask... it might be more generic... so far seems to be
##            specific for featsel and rfe
## buildMaskFromFeatureIds ... used in ifs

