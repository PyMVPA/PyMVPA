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
    from mvpa.base import warning
    from mvpa.base import debug


class Mapper(object):
    """Interface to provide mapping between two spaces: in and out.
    Methods are prefixed correspondingly. forward/reverse operate
    on the entire dataset. get(In|Out)Id[s] operate per element::

              forward
        in   ---------> out
             <--------/
               reverse
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
            s = "metric=%s" % repr(self.__metric)
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
    dsshape = VProperty(fget=getInShape)



class ProjectionMapper(Mapper):
    """Mapper using a projection matrix to transform the data.

    This class cannot be used directly. Sub-classes have to implement
    the `_train()` method, which has to compute the projection matrix
    given a dataset (see `_train()` docstring for more information).

    Once the projection matrix is available, this class provides
    functionality to perform forward and backwards mapping of data, the
    latter using the hermitian (conjugate) transpose of the projection
    matrix. Additionally, `ProjectionMapper` supports optional (but done
    by default) demeaning of the data and selection of arbitrary
    component (i.e. columns of the projection matrix) of the projection.

    Forward and back-projection matrices (a.k.a. *projection* and
    *reconstruction*) are available via the `proj` and `recon`
    properties. the latter only after it has been computed (after first
    call to `reverse`).
    """

    def __init__(self, selector=None, demean=True):
        """Initialize the ProjectionMapper

        :Parameters:
            selector: None | list
                Which components (i.e. columns of the projection matrix)
                should be used for mapping. If `selector` is `None` all
                components are used. If a list is provided, all list
                elements are treated as component ids and the respective
                components are selected (all others are discarded).
            demean: bool
                Either data should be demeaned while computing
                projections and applied back while doing reverse()

        """
        Mapper.__init__(self)

        self._selector = selector
        self._proj = None
        """Forward projection matrix."""
        self._recon = None
        """Reverse projection (reconstruction) matrix."""
        self._demean = demean
        """Flag whether to demean the to be projected data, prior to projection.
        """
        self._mean = None
        """Data mean"""
        self._mean_out = None
        """Forward projected data mean."""

    __doc__ = enhancedDocString('ProjectionMapper', locals(), Mapper)


    def train(self, dataset):
        """Determine the projection matrix."""
        # store the feature wise mean
        self._mean = dataset.samples.mean(axis=0)
        # compute projection matrix with subclass logic
        self._train(dataset)

        # perform component selection
        if self._selector is not None:
            self.selectOut(self._selector)


    def _train(self, dataset):
        """Worker method. Needs to be implemented by subclass.

        This method has to train the mapper and store the resulting
        transformation matrix in `self._proj`.
        """
        raise NotImplementedError


    def forward(self, data, demean=None):
        """Perform forward projection.

        :Parameters:
          data: ndarray
            Data array to map
          demean: boolean | None
            Override demean setting for this method call.

        :Returns:
          NumPy array
        """
        # let arg overwrite instance flag
        if demean is None:
            demean = self._demean

        if self._proj is None:
            raise RuntimeError, "Mapper needs to be train before used."
        if demean and self._mean is not None:
            return ((N.asmatrix(data) - self._mean) * self._proj).A
        else:
            return (N.asmatrix(data) * self._proj).A


    def reverse(self, data):
        """Reproject (reconstruct) data into the original feature space.

        :Returns:
          NumPy array
        """
        if self._proj is None:
            raise RuntimeError, "Mapper needs to be trained before used."

        # (re)build reconstruction matrix
        if self._recon is None:
            self._recon = self._proj.H

            if self._demean:
                if self._mean_out is None:
                    # forward project mean and cache result
                    print 'FANCY'
                    print self._mean.shape
                    self._mean_out = self.forward(self._mean, demean=False)
                    if __debug__:
                        debug("MAP_",
                              "Mean of data in input space %s became %s in " \
                              "outspace" % (self._mean, self._mean_out))

        if self._demean:
            return ((N.asmatrix(data) + self._mean_out) * self._recon).A
        else:
            return ((N.asmatrix(data)) * self._recon).A


    def getInShape(self):
        """Returns a one-tuple with the number of original features."""
        return (self._proj.shape[0], )


    def getOutShape(self):
        """Returns a one-tuple with the number of projection components."""
        return (self._proj.shape[1], )


    def getInSize(self):
        """Returns the number of original features."""
        return self._proj.shape[0]


    def getOutSize(self):
        """Returns the number of components to project on."""
        return self._proj.shape[1]


    def selectOut(self, outIds):
        """Choose a subset of components (and remove all others)."""
        self._proj = self._proj[:, outIds]
        # invalidate reconstruction matrix
        self._recon = None
        self._mean_out = None


    proj  = property(fget=lambda self: self._proj, doc="Projection matrix")
    recon = property(fget=lambda self: self._recon, doc="Backprojection matrix")
