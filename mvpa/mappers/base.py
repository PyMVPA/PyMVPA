# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Basic, general purpose and meta mappers."""

__docformat__ = 'restructuredtext'

import numpy as np
import copy

from mvpa.base.learner import Learner
from mvpa.base.types import is_datasetlike, accepts_dataset_as_samples
from mvpa.base.dochelpers import _str

if __debug__:
    from mvpa.base import debug


class Mapper(Learner):
    """Basic mapper interface definition.

    ::

              forward
             --------->
         IN              OUT
             <--------/
               reverse

    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
          All additional arguments are passed to the baseclass.
        """
        Learner.__init__(self, **kwargs)
        # internal settings that influence what should be done to the dataset
        # attributes in the default forward() and reverse() implementations.
        # they are passed to the Dataset.copy() method
        self._sa_filter = None
        self._fa_filter = None
        self._a_filter = None


    #
    # The following methods are abstract and merely define the intended
    # interface of a mapper and have to be implemented in derived classes. See
    # the docstrings of the respective methods for details about what they
    # should do.
    #
    def _forward_data(self, data):
        """Forward-map some data.

        This is a private method that has to be implemented in derived
        classes.

        Parameters
        ----------
        data : anything (supported the derived class)
        """
        raise NotImplementedError


    def _reverse_data(self, data):
        """Reverse-map some data.

        This is a private method that has to be implemented in derived
        classes.

        Parameters
        ----------
        data : anything (supported the derived class)
        """
        raise NotImplementedError


    #
    # The following methods are candidates for reimplementation in derived
    # classes, in cases where the provided default behavior is not appropriate.
    #
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
        mds = dataset.copy(deep=False,
                           sa=self._sa_filter,
                           fa=self._fa_filter,
                           a=self._a_filter)
        mds.samples = msamples
        return mds


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
        mds = dataset.copy(deep=False,
                           sa=self._sa_filter,
                           fa=self._fa_filter,
                           a=self._a_filter)
        mds.samples = msamples
        return mds


    #
    # The following methods provide common functionality for all mappers
    # and there should be no immediate need to reimplement them
    #
    def forward(self, data):
        """Map data from input to output space.

        Parameters
        ----------
        data : Dataset-like, (at least 2D)-array-like
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
        if isinstance(data, np.ndarray):
            return self.forward(data[np.newaxis])[0]
        else:
            return self.forward(np.array([data]))[0]



    def reverse(self, data):
        """Reverse-map data from output back into input space.

        Parameters
        ----------
        data : Dataset-like, anything
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
        if isinstance(data, np.ndarray):
            return self.reverse(data[np.newaxis])[0]
        else:
            return self.reverse(np.array([data]))[0]


    def __repr__(self):
        return "%s(space=%s)" \
                % (self.__class__.__name__,
                   repr(self.get_space()))



class CombinedMapper(Mapper):
    """Meta mapper that combines several embedded mappers.

    This mapper can be used the map from several input dataspaces into a common
    output dataspace. When :meth:`~mvpa.mappers.base.CombinedMapper.forward`
    is called with a sequence of data, each element in that sequence is passed
    to the corresponding mapper, which in turned forward-maps the data. The
    output of all mappers is finally stacked (horizontally or column or
    feature-wise) into a single large 2D matrix (nsamples x nfeatures).

    Notes
    -----
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
        Parameters
        ----------
        mappers : list of Mapper instances
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

        Parameters
        ----------
        data : sequence
          Each element in the `data` sequence is passed to the corresponding
          embedded mapper and is mapped individually by it. The number of
          elements in `data` has to match the number of embedded mappers. Each
          element is `data` has to provide the same number of samples
          (first dimension).

        Returns
        -------
        array : nsamples x nfeatures
          Horizontally stacked array of all embedded mapper outputs.
        """
        if not len(data) == len(self._mappers):
            raise ValueError, \
                  "CombinedMapper needs a sequence with data for each " \
                  "Mapper"

        # return a big array for the result of the forward mapped data
        # of each embedded mapper
        try:
            return np.hstack(
                    [self._mappers[i].forward(d) for i, d in enumerate(data)])
        except ValueError:
            raise ValueError, \
                  "Embedded mappers do not generate same number of samples. " \
                  "Check input data."


    def reverse(self, data):
        """Reverse map data from OUT space into the IN spaces.

        Parameters
        ----------
        data : array
          Single data array to be reverse mapped into a sequence of data
          snippets in their individual IN spaces.

        Returns
        -------
        list
        """
        # assure array and transpose
        # i.e. transpose of 1D does nothing, but of 2D puts features
        # along first dimension
        data = np.asanyarray(data).T

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

        Parameters
        ----------
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
        return np.sum(m.get_insize() for m in self._mappers)


    def get_outsize(self):
        """Returns the size of the entity in output space"""
        return np.sum(m.get_outsize() for m in self._mappers)


    ##REF: Name was automagically refactored
    ##TODO: yoh: is deprecated, right?
    def select_out(self, outIds):
        """Remove some elements and leave only ids in 'out'/feature space.

        Notes
        -----
        The subset selection is done inplace

        Parameters
        ----------
        outIds : sequence
          All output feature ids to be selected/kept.
        """
        # determine which features belong to what mapper
        # and call its select_out() accordingly
        ids = np.asanyarray(outIds)
        fsum = 0
        for m in self._mappers:
            # bool which meta feature ids belongs to this mapper
            selector = np.logical_and(ids < fsum + m.get_outsize(), ids >= fsum)
            # make feature ids relative to this dataset
            selected = ids[selector] - fsum
            fsum += m.get_outsize()
            # finally apply to mapper
            m.select_out(selected)


    ##REF: Name was automagically refactored
    def get_neighbor(self, outId, *args, **kwargs):
        """Get the ids of the neighbors of a single feature in output dataspace.

        Parameters
        ----------
        outId : int
          Single id of a feature in output space, whose neighbors should be
          determined.
        *args, **kwargs
          Additional arguments are passed to the metric of the embedded
          mapper, that is responsible for the corresponding feature.

        Returns
        -------
        list of outIds
        """
        fsum = 0
        for m in self._mappers:
            fsum_new = fsum + m.get_outsize()
            if outId >= fsum and outId < fsum_new:
                return m.get_neighbor(outId - fsum, *args, **kwargs)
            fsum = fsum_new

        raise ValueError, "Invalid outId passed to CombinedMapper.get_neighbor()"


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
        Parameters
        ----------
        mappers : list of Mapper instances
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
                              space=self.get_space())


    def forward(self, data):
        """Forward data or datasets through the chain.

        See baseclass method for more information.
        """
        mp = data
        for m in self:
            if __debug__:
                debug('MAP', "Forwarding input (%s) though '%s'."
                        % (mp.shape, str(m)))
            mp = m.forward(mp)
        return mp


    def forward1(self, data):
        """Forward data or datasets through the chain.

        See baseclass method for more information.
        """
        mp = data
        for m in self:
            if __debug__:
                debug('MAP', "Forwarding single input (%s) though '%s'."
                        % (mp.shape, str(m)))
            mp = m.forward1(mp)
        return mp


    def reverse(self, data):
        """Reverse-maps data or datasets through the chain (backwards).

        See baseclass method for more information.
        """
        mp = data
        for m in reversed(self):
            # we ignore mapper that do not have reverse mapping implemented
            # (e.g. detrending). That might cause problems if ignoring the
            # mapper make the data incompatible input for the next mapper in
            # the chain. If that pops up, we have to think about a proper
            # solution.
            try:
                if __debug__:
                    debug('MAP',
                          "Reversing %s-shaped input though '%s'."
                           % (mp.shape, str(m)))
                mp = m.reverse(mp)
            except NotImplementedError:
                if __debug__:
                    debug('MAP', "Ignoring %s on reverse mapping." % m)
        return mp


    def reverse1(self, data):
        """Reverse-maps data or datasets through the chain (backwards).

        See baseclass method for more information.
        """
        mp = data
        for i, m in enumerate(reversed(self)):
            # we ignore mapper that do not have reverse mapping implemented
            # (e.g. detrending). That might cause problems if ignoring the
            # mapper make the data incompatible input for the next mapper in
            # the chain. If that pops up, we have to think about a proper
            # solution.
            try:
                if __debug__:
                    debug('MAP',
                          "Reversing single %s-shaped input though '%s'."
                           % (mp.shape, str(m)))
                mp = m.reverse1(mp)
            except NotImplementedError:
                if __debug__:
                    debug('MAP', "Ignoring %s on reverse mapping." % m)
            except ValueError:
                if __debug__:
                    debug('MAP',
                          "Failed to reverse-map through chain at '%s'. Maybe "
                          "previous mapper return multiple samples. Trying to "
                          "switch to reverse() for the remainder of the chain."
                          % str(m))
                mp = self[:-1*i].reverse(mp)
                return mp
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


    def __repr__(self):
        s = Mapper.__repr__(self)
        m_repr = 'mappers=[%s]' % ', '.join([repr(m) for m in self])
        return s.replace("(", "(%s, " % m_repr, 1)


    def __str__(self):
        mapperlist = "%s" % "-".join([str(m) for m in self])
        return _str(self,
                    mapperlist.replace('Mapper', ''))

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
