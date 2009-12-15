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
import copy

from mvpa.base.types import is_datasetlike, accepts_dataset_as_samples
from mvpa.misc.vproperty import VProperty

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
    def __init__(self, inspace=None):
        """
        :Parameters:
        """
        self.__inspace = None
        self.set_inspace(inspace)

    #
    # The following methods are abstract and merely define the intended
    # interface of a mapper and have to be implemented in derived classes. See
    # the docstrings of the respective methods for details about what they
    # should do.
    #

    def forward(self, data):
        """Map data from input to output space.

        Parameters
        ----------
        data: Dataset-like, (at least 2D)-array-like
          Typically this is a `Dataset`, but it might also be a plain data
          array, or even something completely different(TM) that is supported
          by a subclass' implementation. If such an object is Dataset-like it
          is handled by a dedicated method that also transforms dataset
          attributes if necessary. If an array-like is passed, it has to be
          at least two-dimensional, with the first axis separating samples
          or observations. For single samples `forward1()` might be more
          appropriate.
        """
        if is_datasetlike(data):
            return self._forward_dataset(data)
        else:
            if __debug__:
                if hasattr(data, 'ndim') and data.ndim < 2:
                    raise ValueError(
                        'Mapper.forward() only support mapping of data with '
                        'at least two dimensions, where the first axis '
                        'separates samples/observations. Consider using '
                        'Mapper.forward1() instead.')
            return self._forward_data(data)


    def forward1(self, data):
        """Wrapper method to map single samples.

        It is basically identical to `forward()`, but also accepts
        one-dimensional arguments. The map whole dataset this method cannot
        be used. but `forward()` handles them.
        """
        return self.forward(N.array([data]))[0]


    def _forward_data(self, data):
        """Forward-map some data.

        This is a private method that has to be implemented in derived
        classes.

        Parameters
        ----------
        data : anything (supported the derived class)
        """
        raise NotImplementedError


    def _forward_dataset(self, dataset):
        """Forward-map a dataset.

        This is a private method that can be reimplemented in derived
        classes. The default implementation forward-maps the dataset samples
        and returns a new dataset that is a shallow copy of the input with
        the mapped samples.

        Parameters
        ----------
        dataset : Dataset-like
        """
        msamples = self._forward_data(dataset.samples)
        mds = dataset.copy(deep=False)
        mds.samples = msamples
        return mds


    def reverse(self, data):
        """Reverse-map data from output back into input space.

        Parameters
        ----------
        data: Dataset-like, anything
          Typically this is a `Dataset`, but it might also be a plain data
          array, or even something completely different(TM) that is supported
          by a subclass' implementation. If such an object is Dataset-like it
          is handled by a dedicated method that also transforms dataset
          attributes if necessary.
        """
        if is_datasetlike(data):
            return self._reverse_dataset(data)
        else:
            return self._reverse_data(data)


    def reverse1(self, data):
        """Wrapper method to map single samples.

        It is basically identical to `reverse()`, but accepts one-dimensional
        arguments. To map whole dataset this method cannot be used. but
        `reverse()` handles them.
        """
        return self.reverse(N.atleast_2d(data))[0]


    def _reverse_data(self, data):
        """Reverse-map some data.

        This is a private method that has to be implemented in derived
        classes.

        Parameters
        ----------
        data : anything (supported the derived class)
        """
        raise NotImplementedError


    def _reverse_dataset(self, dataset):
        """Reverse-map a dataset.

        This is a private method that can be reimplemented in derived
        classes. The default implementation reverse-maps the dataset samples
        and returns a new dataset that is a shallow copy of the input with
        the mapped samples.

        Parameters
        ----------
        dataset : Dataset-like
        """
        msamples = self._reverse_data(dataset.samples)
        mds = dataset.copy(deep=False)
        mds.samples = msamples
        return mds


    def get_insize(self):
        """Returns the size of the entity in input space"""
        raise NotImplementedError


    def get_outsize(self):
        """Returns the size of the entity in output space"""
        raise NotImplementedError


    #
    # The following methods are candidates for reimplementation in derived
    # classes, in cases where the provided default behavior is not appropriate.
    #
    def is_valid_outid(self, outid):
        """Validate feature id in OUT space.

        Override if OUT space is not simly a 1D vector
        """
        return(outid >= 0 and outid < self.get_outsize())


    def is_valid_inid(self, inid):
        """Validate id in IN space.

        Override if IN space is not simly a 1D vector
        """
        return(inid >= 0 and inid < self.get_insize())


    def train(self, dataset):
        """Perform training of the mapper.

        This method is called to put the mapper in a state that allows it to
        perform the intended mapping. It takes care of running pre- and
        postprocessing that is potentially implemented in derived classes.

        Parameters
        ----------
        dataset: Dataset-like, anything
          Typically this is a `Dataset`, but it might also be a plain data
          array, or even something completely different(TM) that is supported
          by a subclass' implementation.

        Results
        -------
        whoknows
          Returns whatever is returned by the derived class.
        """
        # this mimics Classifier.train() -- we might merge them all at some
        # point
        self._pretrain(dataset)
        result = self._train(dataset)
        self._posttrain(dataset)
        return result


    def _train(self, dataset):
        """Worker method. Needs to be implemented by subclass."""
        raise NotImplementedError


    def _pretrain(self, dataset):
        """Preprocessing before actual mapper training.

        This method can be reimplemented in derived classes. By default it does
        nothing.

        Parameters
        ----------
        dataset: Dataset-like, anything
          Typically this is a `Dataset`, but it might also be a plain data
          array, or even something completely different(TM) that is supported
          by a subclass' implementation.
        """
        pass


    def _posttrain(self, dataset):
        """Postprocessing after actual mapper training.

        This method can be reimplemented in derived classes. By default it does
        nothing.

        Parameters
        ----------
        dataset: Dataset-like, anything
          Typically this is a `Dataset`, but it might also be a plain data
          array, or even something completely different(TM) that is supported
          by a subclass' implementation.
        """
        pass


    #
    # The following methods provide common functionality for all mappers
    # and there should be no immediate need to reimplement them
    #
    def __repr__(self):
        return "%s(inspace=%s)" \
                % (self.__class__.__name__,
                   repr(self.get_inspace()))


    def __call__(self, data):
        """Calls the mappers forward() method.
        """
        return self.forward(data)


    def get_inspace(self):
        """
        """
        return self.__inspace


    def set_inspace(self, name):
        """
        """
        self.__inspace = name


    nfeatures = VProperty(fget=get_outsize)



class FeatureSliceMapper(Mapper):
    """Mapper to select a subset of features.
    """
    def __init__(self, slicearg, dshape=None, **kwargs):
        """
        Parameters
        ----------
        slicearg : int, list(int), array(int), array(bool)
        dshape : tuple
        """
        Mapper.__init__(self, **kwargs)
        # store it here, might be modified later
        self.__dshape = dshape

        # convert int sliceargs into lists to prevent getting scalar values when
        # slicing
        if isinstance(slicearg, int):
            slicearg = [slicearg]
        self._slicearg = slicearg


    def __repr__(self):
        s = super(FeatureSliceMapper, self).__repr__()
        return s.replace("(",
                         "(slicearg=%s, dshape=%s, "
                          % (repr(self._slicearg), repr(self.__dshape)),
                         1)


    def _forward_data(self, data):
        """Map data from the original dataspace into featurespace.

        Parameters
        ----------
        data : array-like
          Either one-dimensional sample or two-dimensional samples matrix.
        """
        return data[:, self._slicearg]


    def _reverse_data(self, data):
        """Reverse map data from featurespace into the original dataspace.

        Parameters
        ----------
        data : array-like
          Either one-dimensional sample or two-dimensional samples matrix.
        """
        if self.__dshape is None:
            raise RuntimeError(
                "Cannot reverse-map data since the original data shape is "
                "unknown. Either set `dshape` in the constructor, or call "
                "train().")
        mapped = N.zeros(data.shape[:1] + self.__dshape,
                         dtype=data.dtype)
        mapped[:, self._slicearg] = data
        return mapped


    @accepts_dataset_as_samples
    def _train(self, data):
        if self.__dshape is None:
            # XXX what about arrays of generic objects???
            self.__dshape = data.shape[1:]



class FeatureSubsetMapper(Mapper):
    """Mapper to select a subset of features.

    This mapper only operates on one-dimensional samples or two-dimensional
    samples matrices. If necessary it can be combined for FlattenMapper to
    handle multidimensional data.
    """
    def __init__(self, mask, dshape=None, **kwargs):
        """
        Parameters
        ----------
        mask : array, int
          This is a one-dimensional array whos non-zero elements define both
          the feature subset and the data shape the mapper is going to handle.
          In case `mask` is an int, it is expaned into a boolean vector consisting
          of as many `True` elements.
        """
        Mapper.__init__(self, **kwargs)
        # store it here, might be modified later
        self.__dshape = dshape
        if isinstance(mask, int):
            self.__dshape = (mask, )
            mask = N.arange(mask)
        else:
            # must be an array (will convert e.g. list arguments)
            mask = N.asanyarray(mask)
            if not len(mask.shape) == 1:
                raise ValueError("The mask has to be a one-dimensional vector. "
                                 "For multidimensional data consider FlattenMapper "
                                 "before running SubsetMapper.")
        if mask.dtype == 'bool':
            self.__mask = mask.nonzero()[0]
            # we got a boolean mask in feature space, can set dshape from here
            if self.__dshape is None:
                self.__dshape = mask.shape
            elif mask.shape !=  self.__dshape:
                raise ValueError(
                    "Conflicting arguments: Given `dshape` (%s) does not match "
                    "the shape of the boolean mask array (%s)."
                    % (self.__dshape, mask.shape))
        elif mask.dtype.char in N.typecodes['AllInteger']:
            # assume we have indices already
            self.__mask = mask
        else:
            raise ValueError(
                "Only integer and bool are supported datatypes for array-based "
                "slicing arguments (got: %s)." % mask.dtype)


    def __repr__(self):
        s = super(FeatureSubsetMapper, self).__repr__()
        return s.replace("(", "(mask=%s," % repr(self.__mask), 1)


    def _forward_data(self, data):
        """Map data from the original dataspace into featurespace.

        Parameters
        ----------
        data : array-like
          Either one-dimensional sample or two-dimensional samples matrix.
        """
        return data[:, self.__mask]


    def _reverse_data(self, data):
        """Reverse map data from featurespace into the original dataspace.

        Parameters
        ----------
        data : array-like
          Either one-dimensional sample or two-dimensional samples matrix.
        """
        if self.__dshape is None:
            raise RuntimeError(
                "Cannot reverse-map data since the original data shape is "
                "unknown. Either set `dshape` in the constructor, or call "
                "train().")
        mapped = N.zeros(data.shape[:1] + self.__dshape,
                         dtype=data.dtype)
        mapped[:, self.__mask] = data
        return mapped


    @accepts_dataset_as_samples
    def _train(self, data):
        if self.__dshape is None:
            # XXX what about arrays of generic objects???
            self.__dshape = data.shape[1:]


    def get_insize(self):
        """Return the length of the input space vectors."""
        if not self.__dshape is None:
            # should be just length 1, but if this becomes generic slicing
            # mapper this might change
            return N.prod(self.__dshape)
        else:
            return None


    def get_outsize(self):
        """OutSize is a number of non-0 elements in the mask"""
        return len(self.__mask)


    def get_mask(self, copy=True):
        """By default returns a copy of the current mask.

        Parameters
        ----------
        copy : bool
          If False a reference to the mask is returned, a copy otherwise.
          This shared mask must not be modified!
        """
        if copy:
            return self.__mask.copy()
        else:
            return self.__mask


    def is_valid_outid(self, id):
        """Return whether a particular id is a valid output id/coordinate.

        Parameters
        ----------
        id : int
        """
        return id >=0 and id < self.get_outsize()


    def is_valid_inid(self, id):
        """Return whether a particular id is a valid input id/coordinate.

        Parameters
        ----------
        id : int
        """
        return id in self.__mask


    def select_out(self, slicearg, cow=True):
        """Limit the feature subset selection.

        Parameters
        ----------
        slicearg : array(bool), list, slice
          Any valid Numpy slicing argument defining a subset of the current
          feature set.
        cow: bool
          For internal use only!
          If `True`, it is safe to call the function on a shallow copy of
          another FeatureSubsetMapper instance without affecting the original
          mapper instance. If `False`, modifications done to one instance
          invalidate the other.

        Seealso
        -------
        `FeatureSubsetMapper.discard_out()`
        """
        self.__mask = N.atleast_1d(self.__mask[slicearg])



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

        if not len(data) == self.get_outsize():
            raise ValueError, \
                  "Data shape does match mapper reverse mapping properties."

        result = []
        fsum = 0
        for m in self._mappers:
            # calculate upper border
            fsum_new = fsum + m.get_outsize()

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
        if dataset.nfeatures != self.get_outsize():
            raise ValueError, "Training dataset does not match the mapper " \
                              "properties."

        fsum = 0
        for m in self._mappers:
            # need to split the dataset
            fsum_new = fsum + m.get_outsize()
            m.train(dataset[:, range(fsum, fsum_new)])
            fsum = fsum_new


    def get_insize(self):
        """Returns the size of the entity in input space"""
        return N.sum(m.get_insize() for m in self._mappers)


    def get_outsize(self):
        """Returns the size of the entity in output space"""
        return N.sum(m.get_outsize() for m in self._mappers)


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
            selector = N.logical_and(ids < fsum + m.get_outsize(), ids >= fsum)
            # make feature ids relative to this dataset
            selected = ids[selector] - fsum
            fsum += m.get_outsize()
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
            fsum_new = fsum + m.get_outsize()
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
    """Meta mapper that embeds a chain of other mappers.

    Each mapper in the chain is called successively to perform forward or
    reverse mapping. The class behaves to some degree like a list container.
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


    def __copy__(self):
        # XXX need to copy the base class stuff as well
        return self.__class__([copy.copy(m) for m in self],
                              inspace=self.get_inspace())


    def forward(self, data):
        """Forward data or datasets through the chain.

        See baseclass method for more information.
        """
        mp = data
        for m in self:
            mp = m.forward(mp)
        return mp


    def reverse(self, data):
        """Reverse-maps data or datasets through the chain (backwards).

        See baseclass method for more information.
        """
        mp = data
        for m in reversed(self):
            mp = m.reverse(mp)
        return mp


    def _train(self, dataset):
        """Trains the mapper chain.

        The training dataset is used to train the first mapper. Afterwards it is
        forward-mapped by this (now trained) mapper and the transformed dataset
        and then used to train the next mapper. This procedure is done till all
        mapper are trained.

        Parameters
        ----------
        dataset: `Dataset`
        """
        nmappers = len(self) - 1
        tdata = dataset
        for i, mapper in enumerate(self):
            mapper.train(tdata)
            # forward through all but the last mapper
            if i < nmappers:
                tdata = mapper.forward(tdata)


    def get_insize(self):
        """Returns the size of the entity in input space"""
        return self[0].get_insize()


    def get_outsize(self):
        """Returns the size of the entity in output space"""
        return self[-1].get_outsize()


    def is_valid_inid(self, id):
        """Queries the first mapper in the chain for this information."""
        return self[0].is_valid_inid(id)


    def is_valid_outid(self, id):
        """Queries the last mapper in the chain for this information."""
        return self[-1].is_valid_outid(id)


    def __ensure_selectable_tail(self):
        """Append a FeatureSubsetMapper to the chain if there is none yet."""
        if not isinstance(self[-1], FeatureSubsetMapper):
            self.append(FeatureSubsetMapper(self[-1].get_outsize()))


    def select_out(self, slicearg, cow=True):
        """Limit the feature subset selection.

        To achieve this a FeatureSubsetMapper is appended to the mapper chain
        (if necessary) and the arguments are passed to it.

        See baseclass method for more information.
        """
        self.__ensure_selectable_tail()
        self[-1].select_out(slicearg, cow)


    def discard_out(self, slicearg, cow=True):
        """Limit the feature subset selection.

        To achieve this a FeatureSubsetMapper is appended to the mapper chain
        (if necessary) and the arguments are passed to it.

        See baseclass method for more information.
        """
        self.__ensure_selectable_tail()
        self[-1].discard_out(slicearg, cow)


    def __repr__(self):
        s = Mapper.__repr__(self)
        m_repr = 'mappers=[%s]' % ', '.join([repr(m) for m in self])
        return s.replace("(", "(%s, " % m_repr, 1)

    #
    # Behave as a container
    #
    def append(self, mapper):
        """Append a mapper to the chain.

        The mapper's input size has to match the output size of the current
        chain.
        """
        # not checking, since get_outsize() is about to vanish
        #if not self.get_outsize() == mapper.get_insize():
        #    raise ValueError("To be appended mapper does not match the output "
        #                     "size of the current chain (%s vs. %s)."
        #                     % (mapper.get_insize(),  self.get_outsize()))
        self._mappers.append(mapper)


    def __len__(self):
        return len(self._mappers)


    def __iter__(self):
        for m in self._mappers:
            yield m


    def __reversed__(self):
        return reversed(self._mappers)


    def __getitem__(self, key):
        # if just one is requested return just one, otherwise return a
        # ChainMapper again
        if isinstance(key, int):
            return self._mappers[key]
        else:
         # operate on shallow copy of self
         sliced = copy.copy(self)
         sliced._mappers = self._mappers[key]
         return sliced
