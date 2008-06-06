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


from mvpa.mappers.base import MaskMapper
from mvpa.datasets.metric import Metric
from mvpa.base.dochelpers import enhancedDocString



class MetricMapper(MaskMapper, Metric):
    """Mapper which has information about the metrics of the dataspace it is
    mapping.
    """
    def __init__(self, mask, metric):
        """Cheap initialisation.

        'metric' is a subclass of Metric.
        """
        MaskMapper.__init__(self, mask)
        Metric.__init__(self)

        if not isinstance(metric, Metric):
            raise ValueError, "MetricMapper has to be initialized with an " \
                              "instance of a 'Metric' object. Got %s" \
                                % `type(metric)`
        self.__metric = metric


    __doc__ = enhancedDocString('MetricMapper', locals(), MaskMapper, Metric)


    def getMetric(self):
        """To make pylint happy"""
        return self.__metric


    def setMetric(self, metric):
        """To make pylint happy"""
        self.__metric = metric


    metric = property(fget=getMetric, fset=setMetric)
