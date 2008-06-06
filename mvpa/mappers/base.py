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

import numpy as N

from mvpa.datasets.metric import Metric

from mvpa.misc.vproperty import VProperty
from mvpa.base.dochelpers import enhancedDocString

if __debug__:
    from mvpa.misc import warning


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
    def __init__(self, metric=None):
        """Initialize a Mapper.

        :Parameters:
          metric : Metric
            Optional metric
        """
        self.__metric = None
        """Pylint happiness"""
        self.setMetric(metric)
        """Actually assign the metric"""


    __doc__ = enhancedDocString('Mapper', locals())


    def __repr__(self):
        if self.__metric is not None:
            s = "metric=%s" % `self.__metric`
        else:
            s = ''
        return "%s(%s)" % (self.__class__.__name__, s)

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


    def train(self, dataset):
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


    def getMetric(self):
        """To make pylint happy"""
        return self.__metric


    def isValidOutId(self, outId):
        """Validate if OutId is valid

        Override if Out space is not simly a 1D vector
        """
        return(outId>=0 and outId<self.getOutSize())


    def isValidInId(self, inId):
        """Validate if InId is valid

        Override if In space is not simly a 1D vector
        """
        return(inId>=0 and inId<self.getInSize())


    def setMetric(self, metric):
        """To make pylint happy"""
        if metric is not None and not isinstance(metric, Metric):
            raise ValueError, "metric for Mapper must be an " \
                              "instance of a Metric class . Got %s" \
                                % `type(metric)`
        self.__metric = metric


    def getNeighborIn(self, inId, *args, **kwargs):
        """Return the list of coordinates for the neighbors.

        :Parameters:
          inId
            id (index) of input element
          **kwargs : dict
            would be passed to assigned metric

        XXX See TODO below: what to return -- list of arrays or list
        of tuples?
        """
        if self.metric is None:
            raise RuntimeError, "No metric was assigned to %s, thus no " \
                  "neighboring information is present" % self

        isValidInId = self.isValidInId
        if isValidInId(inId):
            for neighbor in self.metric.getNeighbor(inId, *args, **kwargs):
                if isValidInId(neighbor):
                    yield neighbor


    def getNeighbor(self, outId, *args, **kwargs):
        """Return the list of Ids for the neighbors.

        Returns a list of outIds
        """
        if self.metric is None:
            raise RuntimeError, "No metric was assigned to %s, thus no " \
                  "neighboring information is present" % self

        if self.isValidOutId(outId):
            inId = self.getInId(outId)
            for inId in self.getNeighborIn(inId, *args, **kwargs):
                yield self.getOutId(inId)


    def getNeighbors(self, outId, *args, **kwargs):
        """Return the list of coordinates for the neighbors.

        By default it simply constracts the list based on
        the generator getNeighbor
        """
        return [ x for x in self.getNeighbor(outId, *args, **kwargs) ]


    metric = property(fget=getMetric, fset=setMetric)
    nfeatures = VProperty(fget=getOutSize)



