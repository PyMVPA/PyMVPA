#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Abstract base class of all data mappers"""

__docformat__ = 'restructuredtext'

from mvpa.datasets.metric import Metric

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

    See here for a possible solution:
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/440634
    """
    def __init__(self):
        pass

    def forward(self, data):
        """Map data from the original dataspace into featurespace.
        """
        raise NotImplementedError

    def __getitem__(self, data):
        """Calls the mappers forward() method.
        """
        return self.forward(data)

    def reverse(self, data):
        """Reverse map data from featurespace into the original dataspace.
        """
        raise NotImplementedError

    def __call__(self, data):
        """Calls the mappers reverse() method.
        """
        return self.reverse(data)

    def getInShape(self):
        """Returns the dimensionality speicification of the original dataspace.

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


# comment out for now... introduce when needed
#    def getInEmpty(self):
#        """Returns empty instance of input object"""
#        raise NotImplementedError
#
#    def getOutEmpty(self):
#        """Returns empty instance of output object"""
#        raise NotImplementedError

    def getInId(self, outId):
        """For a given Id in "out" returns corresponding "in" Id"""
        raise NotImplementedError

    def getInIds(self):
        """Returns corresponding "in" Ids"""
        raise NotImplementedError

    def getOutId(self, inId):
        """Returns corresponding "out" Id"""
        raise NotImplementedError


### yoh: To think about generalization
##
## getMask... it might be more generic... so far seems to be
##            specific for featsel and rfe
## buildMaskFromFeatureIds ... used in ifs



class MetricMapper(Mapper, Metric):
    """Mapper which has information about the metrics of the dataspace it is
    mapping.
    """
    def __init__(self, metric):
        """Cheap initialisation.

        'metric' is a subclass of Metric.
        """
        Mapper.__init__(self)
        Metric.__init__(self)

        if not isinstance(metric, Metric):
            raise ValueError, "MetricMapper has to be initialized with an " \
                              "instance of a 'Metric' object. Got %s" \
                                % `type(metric)`
        self.__metric = metric


    def getMetric(self):
        """To make pylint happy"""
        return self.__metric


    def setMetric(self, metric):
        """To make pylint happy"""
        self.__metric = metric


    metric = property(fget=getMetric, fset=setMetric)


