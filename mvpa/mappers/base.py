# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Data mapper"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.mappers.metric import Metric

from mvpa.misc.vproperty import VProperty
from mvpa.base.dochelpers import enhancedDocString

if __debug__:
    from mvpa.base import warning
    from mvpa.base import debug


class Mapper(object):
    """Interface to provide mapping between two spaces: IN and OUT.
    Methods are prefixed correspondingly. forward/reverse operate
    on the entire dataset. get(In|Out)Id[s] operate per element::

              forward
             --------->
         IN              OUT
             <--------/
               reverse
    """
    def __init__(self, metric=None):
        """
        :Parameters:
          metric : Metric
            Optional metric
        """
        self.__metric = None
        """Pylint happiness"""
        self.setMetric(metric)
        """Actually assign the metric"""

    #
    # The following methods are abstract and merely define the intended
    # interface of a mapper and have to be implemented in derived classes. See
    # the docstrings of the respective methods for details about what they
    # should do.
    #

    def forward(self, data):
        """Map data from the IN dataspace into OUT space.
        """
        raise NotImplementedError


    def reverse(self, data):
        """Reverse map data from OUT space into the IN space.
        """
        raise NotImplementedError


    def getInSize(self):
        """Returns the size of the entity in input space"""
        raise NotImplementedError


    def getOutSize(self):
        """Returns the size of the entity in output space"""
        raise NotImplementedError


    def selectOut(self, outIds):
        """Limit the OUT space to a certain set of features.

        :Parameters:
          outIds: sequence
            Subset of ids of the current feature in OUT space to keep.
        """
        raise NotImplementedError


    def getInId(self, outId):
        """Translate a feature id into a coordinate/index in input space.

        Such a translation might not be meaningful or even possible for a
        particular mapping algorithm and therefore cannot be relied upon.
        """
        raise NotImplementedError


    #
    # The following methods are candidates for reimplementation in derived
    # classes, in cases where the provided default behavior is not appropriate.
    #
    def isValidOutId(self, outId):
        """Validate feature id in OUT space.

        Override if OUT space is not simly a 1D vector
        """
        return(outId >= 0 and outId < self.getOutSize())


    def isValidInId(self, inId):
        """Validate id in IN space.

        Override if IN space is not simly a 1D vector
        """
        return(inId >= 0 and inId < self.getInSize())


    def train(self, dataset):
        """Perform training of the mapper.

        This method is called to put the mapper in a state that allows it to
        perform to intended mapping.

        :Parameter:
          dataset: Dataset or subclass

        .. note::
          The default behavior of this method is to do nothing.
        """
        pass


    def getNeighbor(self, outId, *args, **kwargs):
        """Get feature neighbors in input space, given an id in output space.

        This method has to be reimplemented whenever a derived class does not
        provide an implementation for :meth:`~mvpa.mappers.base.Mapper.getInId`.
        """
        if self.metric is None:
            raise RuntimeError, "No metric was assigned to %s, thus no " \
                  "neighboring information is present" % self

        if self.isValidOutId(outId):
            inId = self.getInId(outId)
            for inId in self.getNeighborIn(inId, *args, **kwargs):
                yield self.getOutId(inId)


    #
    # The following methods provide common functionality for all mappers
    # and there should be no immediate need to reimplement them
    #
    def getNeighborIn(self, inId, *args, **kwargs):
        """Return the list of coordinates for the neighbors.

        :Parameters:
          inId
            id (index) of an element in input dataspace.
          *args, **kwargs
            Any additional arguments are passed to the embedded metric of the
            mapper.

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


    def getNeighbors(self, outId, *args, **kwargs):
        """Return the list of coordinates for the neighbors.

        By default it simply constructs the list based on
        the generator returned by getNeighbor()
        """
        return [ x for x in self.getNeighbor(outId, *args, **kwargs) ]


    def __repr__(self):
        if self.__metric is not None:
            s = "metric=%s" % repr(self.__metric)
        else:
            s = ''
        return "%s(%s)" % (self.__class__.__name__, s)


    def __call__(self, data):
        """Calls the mappers forward() method.
        """
        return self.forward(data)


    def getMetric(self):
        """To make pylint happy"""
        return self.__metric


    def setMetric(self, metric):
        """To make pylint happy"""
        if metric is not None and not isinstance(metric, Metric):
            raise ValueError, "metric for Mapper must be an " \
                              "instance of a Metric class . Got %s" \
                                % `type(metric)`
        self.__metric = metric


    metric = property(fget=getMetric, fset=setMetric)
    nfeatures = VProperty(fget=getOutSize)



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

        # get feature-wise mean in out-space
        if self._demean and self._mean_out is None:
            # forward project mean and cache result
            self._mean_out = self.forward(self._mean, demean=False)
            if __debug__:
                debug("MAP_",
                      "Mean of data in input space %s became %s in " \
                      "outspace" % (self._mean, self._mean_out))


        # (re)build reconstruction matrix
        if self._recon is None:
            self._recon = self._proj.H

        if self._demean:
            return ((N.asmatrix(data) + self._mean_out) * self._recon).A
        else:
            return ((N.asmatrix(data)) * self._recon).A


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



class CombinedMapper(Mapper):
    """Meta mapper that combines several embedded mappers.

    This mapper can be used the map from several input dataspaces into a common
    output dataspace. When :meth:`~mvpa.mappers.base.CombinedMapper.forward`
    is called with a sequence of data, each element in that sequence is passed
    to the corresponding mapper, which in turned forward-maps the data. The
    output of all mappers is finally stacked (horizontally or column or
    feature-wise) into a single large 2D matrix (nsamples x nfeatures).

    .. note::
      This mapper can only embbed mappers that transform data into a 2D
      (nsamples x nfeatures) representation. For mappers not supporting this
      transformation, consider wrapping them in a
      :class:`~mvpa.mappers.base.ChainMapper` with an appropriate
      post-processing mapper.

    CombinedMapper fully supports forward and backward mapping, training,
    runtime selection of a feature subset (in output dataspace) and retrieval
    of neighborhood information.
    """
    def __init__(self, mappers, **kwargs):
        """
        :Parameters:
          mappers: list of Mapper instances
            The order of the mappers in the list is important, as it will define
            the order in which data snippets have to be passed to
            :meth:`~mvpa.mappers.base.CombinedMapper.forward`.
          **kwargs
            All additional arguments are passed to the base-class constructor.
        """
        Mapper.__init__(self, **kwargs)

        if not len(mappers):
            raise ValueError, \
                  'CombinedMapper needs at least one embedded mapper.'

        self._mappers = mappers


    def forward(self, data):
        """Map data from the IN spaces into to common OUT space.

        :Parameter:
          data: sequence
            Each element in the `data` sequence is passed to the corresponding
            embedded mapper and is mapped individually by it. The number of
            elements in `data` has to match the number of embedded mappers. Each
            element is `data` has to provide the same number of samples
            (first dimension).

        :Returns:
          array: nsamples x nfeatures
            Horizontally stacked array of all embedded mapper outputs.
        """
        if not len(data) == len(self._mappers):
            raise ValueError, \
                  "CombinedMapper needs a sequence with data for each " \
                  "Mapper"

        # return a big array for the result of the forward mapped data
        # of each embedded mapper
        try:
            return N.hstack(
                    [self._mappers[i].forward(d) for i, d in enumerate(data)])
        except ValueError:
            raise ValueError, \
                  "Embedded mappers do not generate same number of samples. " \
                  "Check input data."


    def reverse(self, data):
        """Reverse map data from OUT space into the IN spaces.

        :Parameter:
          data: array
            Single data array to be reverse mapped into a sequence of data
            snippets in their individual IN spaces.

        :Returns:
          list
        """
        # assure array and transpose
        # i.e. transpose of 1D does nothing, but of 2D puts features
        # along first dimension
        data = N.asanyarray(data).T

        if not len(data) == self.getOutSize():
            raise ValueError, \
                  "Data shape does match mapper reverse mapping properties."

        result = []
        fsum = 0
        for m in self._mappers:
            # calculate upper border
            fsum_new = fsum + m.getOutSize()

            result.append(m.reverse(data[fsum:fsum_new].T))

            fsum = fsum_new

        return result


    def train(self, dataset):
        """Trains all embedded mappers.

        The provided training dataset is splitted appropriately and the
        corresponding pieces are passed to the
        :meth:`~mvpa.mappers.base.Mapper.train` method of each embedded mapper.

        :Parameter:
          dataset: :class:`~mvpa.datasets.base.Dataset` or subclass
            A dataset with the number of features matching the `outSize` of the
            `CombinedMapper`.
        """
        if dataset.nfeatures != self.getOutSize():
            raise ValueError, "Training dataset does not match the mapper " \
                              "properties."

        fsum = 0
        for m in self._mappers:
            # need to split the dataset
            fsum_new = fsum + m.getOutSize()
            m.train(dataset.selectFeatures(range(fsum, fsum_new)))
            fsum = fsum_new


    def getInSize(self):
        """Returns the size of the entity in input space"""
        return N.sum(m.getInSize() for m in self._mappers)


    def getOutSize(self):
        """Returns the size of the entity in output space"""
        return N.sum(m.getOutSize() for m in self._mappers)


    def selectOut(self, outIds):
        """Remove some elements and leave only ids in 'out'/feature space.

        .. note::
          The subset selection is done inplace

        :Parameter:
          outIds: sequence
            All output feature ids to be selected/kept.
        """
        # determine which features belong to what mapper
        # and call its selectOut() accordingly
        ids = N.asanyarray(outIds)
        fsum = 0
        for m in self._mappers:
            # bool which meta feature ids belongs to this mapper
            selector = N.logical_and(ids < fsum + m.getOutSize(), ids >= fsum)
            # make feature ids relative to this dataset
            selected = ids[selector] - fsum
            fsum += m.getOutSize()
            # finally apply to mapper
            m.selectOut(selected)


    def getNeighbor(self, outId, *args, **kwargs):
        """Get the ids of the neighbors of a single feature in output dataspace.

        :Parameters:
          outId: int
            Single id of a feature in output space, whos neighbors should be
            determined.
          *args, **kwargs
            Additional arguments are passed to the metric of the embedded
            mapper, that is responsible for the corresponding feature.

        Returns a list of outIds
        """
        fsum = 0
        for m in self._mappers:
            fsum_new = fsum + m.getOutSize()
            if outId >= fsum and outId < fsum_new:
                return m.getNeighbor(outId - fsum, *args, **kwargs)
            fsum = fsum_new

        raise ValueError, "Invalid outId passed to CombinedMapper.getNeighbor()"


    def __repr__(self):
        s = Mapper.__repr__(self).rstrip(' )')
        # beautify
        if not s[-1] == '(':
            s += ' '
        s += 'mappers=[%s])' % ', '.join([m.__repr__() for m in self._mappers])
        return s



class ChainMapper(Mapper):
    """Meta mapper that embedded a chain of other mappers.

    Each mapper in the chain is called successively to perform forward or
    reverse mapping.

    .. note::

      In its current implementation the `ChainMapper` treats all but the last
      mapper as simple pre-processing (in forward()) or post-processing (in
      reverse()) steps. All other capabilities, e.g. training and neighbor
      metrics are provided by or affect *only the last mapper in the chain*.

      With respect to neighbor metrics this means that they are determined
      based on the input space of the *last mapper* in the chain and *not* on
      the input dataspace of the `ChainMapper` as a whole
    """
    def __init__(self, mappers, **kwargs):
        """
        :Parameters:
          mappers: list of Mapper instances
          **kwargs
            All additional arguments are passed to the base-class constructor.
        """
        Mapper.__init__(self, **kwargs)

        if not len(mappers):
            raise ValueError, 'ChainMapper needs at least one embedded mapper.'

        self._mappers = mappers


    def forward(self, data):
        """Calls all mappers in the chain successively.

        :Parameter:
          data
            data to be chain-mapped.
        """
        mp = data
        for m in self._mappers:
            mp = m.forward(mp)

        return mp


    def reverse(self, data):
        """Calls all mappers in the chain successively, in reversed order.

        :Parameter:
          data: array
            data array to be reverse mapped into the orginal dataspace.
        """
        mp = data
        for m in reversed(self._mappers):
            mp = m.reverse(mp)

        return mp


    def train(self, dataset):
        """Trains the *last* mapper in the chain.

        :Parameter:
          dataset: :class:`~mvpa.datasets.base.Dataset` or subclass
            A dataset with the number of features matching the `outSize` of the
            last mapper in the chain (which is identical to the one of the
            `ChainMapper` itself).
        """
        if dataset.nfeatures != self.getOutSize():
            raise ValueError, "Training dataset does not match the mapper " \
                              "properties."

        self._mappers[-1].train(dataset)


    def getInSize(self):
        """Returns the size of the entity in input space"""
        return self._mappers[0].getInSize()


    def getOutSize(self):
        """Returns the size of the entity in output space"""
        return self._mappers[-1].getOutSize()


    def selectOut(self, outIds):
        """Remove some elements from the *last* mapper in the chain.

        :Parameter:
          outIds: sequence
            All output feature ids to be selected/kept.
        """
        self._mappers[-1].selectOut(outIds)


    def getNeighbor(self, outId, *args, **kwargs):
        """Get the ids of the neighbors of a single feature in output dataspace.

        .. note::

          The neighbors are determined based on the input space of the *last
          mapper* in the chain and *not* on the input dataspace of the
          `ChainMapper` as a whole!

        :Parameters:
          outId: int
            Single id of a feature in output space, whos neighbors should be
            determined.
          *args, **kwargs
            Additional arguments are passed to the metric of the embedded
            mapper, that is responsible for the corresponding feature.

        Returns a list of outIds
        """
        return self._mappers[-1].getNeighbor(outId, *args, **kwargs)


    def __repr__(self):
        s = Mapper.__repr__(self).rstrip(' )')
        # beautify
        if not s[-1] == '(':
            s += ' '
        s += 'mappers=[%s])' % ', '.join([m.__repr__() for m in self._mappers])
        return s
