#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Data mapper"""

__docformat__ = 'restructuredtext'

from mvpa.misc.vproperty import VProperty
from mvpa.base.dochelpers import enhancedDocString


class Mapper(object):
    """Interface to provide mapping between two spaces: in and out.
    Methods are prefixed correspondingly. forward/reverse operate
    on the entire dataset. get(In|Out)Id[s] operate per element::

              forward
        in   ---------> out
             <--------/
               reverse

    Subclasses should define 'dsshape' and 'nfeatures' properties that point to
    `getInShape` and `getOutSize` respectively. This cannot be
    done in the baseclass as standard Python properties would still point to
    the baseclass methods.
    """
    def __init__(self):
        """Does nothing."""
        pass


    __doc__ = enhancedDocString('Mapper', locals())


    def forward(self, data):
        """Map data from the original dataspace into featurespace.
        """
        raise NotImplementedError


    def __call__(self, data):
        """Calls the mappers forward() method.
        """
        return self.forward(data)


    def reverse(self, data):
        """Reverse map data from featurespace into the original dataspace.
        """
        raise NotImplementedError


    def train(self, data):
        """Sub-classes have to override this method if the mapper need
        training.
        """
        pass


    def getInShape(self):
        """Returns the dimensionality specification of the original dataspace.

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
        """Returns the size of the entity in input space"""
        raise NotImplementedError


    def getOutSize(self):
        """Returns the size of the entity in output space"""
        raise NotImplementedError


    def selectOut(self, outIds):
        """Remove some elements and leave only ids in 'out'/feature space"""
        raise NotImplementedError


    nfeatures = VProperty(fget=getOutSize)
