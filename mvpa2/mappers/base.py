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
from mvpa.base.node import ChainNode
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
        if __debug__:
            debug('MAP_', "Forward-map %s-shaped samples in dataset with '%s'."
                        % (dataset.samples.shape, self))
        msamples = self._forward_data(dataset.samples)
        if __debug__:
            debug('MAP_', "Make shallow copy of to-be-forward-mapped dataset "
                    "and assigned forward-mapped samples ({sf}a_filters: "
                    "%s, %s, %s)." % (self._sa_filter, self._fa_filter,
                                      self._a_filter))
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
        if __debug__:
            debug('MAP_', "Reverse-map %s-shaped samples in dataset with '%s'."
                        % (dataset.samples.shape, self))
        msamples = self._reverse_data(dataset.samples)
        if __debug__:
            debug('MAP_', "Make shallow copy of to-be-reverse-mapped dataset "
                    "and assigned reverse-mapped samples ({sf}a_filters: "
                    "%s, %s, %s)." % (self._sa_filter, self._fa_filter,
                                      self._a_filter))
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
            if __debug__:
                debug('MAP', "Forward-map %s-shaped dataset through '%s'."
                        % (data.shape, self))
            return self._forward_dataset(data)
        else:
            if hasattr(data, 'ndim') and data.ndim < 2:
                raise ValueError(
                    'Mapper.forward() only support mapping of data with '
                    'at least two dimensions, where the first axis '
                    'separates samples/observations. Consider using '
                    'Mapper.forward1() instead.')
            if __debug__:
                debug('MAP', "Forward-map data through '%s'." % (self))
            return self._forward_data(data)


    def forward1(self, data):
        """Wrapper method to map single samples.

        It is basically identical to `forward()`, but also accepts
        one-dimensional arguments. The map whole dataset this method cannot
        be used. but `forward()` handles them.
        """
        if isinstance(data, np.ndarray):
            data = data[np.newaxis]
        else:
            data = np.array([data])
        if __debug__:
            debug('MAP', "Forward-map single %s-shaped sample through '%s'."
                    % (data.shape[1:], self))
        return self.forward(data)[0]



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
            if __debug__:
                debug('MAP', "Reverse-map %s-shaped dataset through '%s'."
                        % (data.shape, self))
            return self._reverse_dataset(data)
        else:
            if __debug__:
                debug('MAP', "Reverse-map data through '%s'." % (self))
            return self._reverse_data(data)


    def reverse1(self, data):
        """Wrapper method to map single samples.

        It is basically identical to `reverse()`, but accepts one-dimensional
        arguments. To map whole dataset this method cannot be used. but
        `reverse()` handles them.
        """
        if isinstance(data, np.ndarray):
            data = data[np.newaxis]
        else:
            data = np.array([data])
        if __debug__:
            debug('MAP', "Reverse-map single %s-shaped sample through '%s'."
                    % (data.shape[1:], self))
        mapped = self.reverse(data)[0]
        if __debug__:
            debug('MAP', "Mapped single %s-shaped sample to %s."
                    % (data.shape[1:], mapped.shape))
        return mapped


    def _call(self, ds):
        return self.forward(ds)



class ChainMapper(ChainNode):
    """Class that amends ChainNode with a mapper-like interface.

    ChainMapper supports sequential training of a mapper chain, as well as
    reverse-mapping and mapping of single samples.
    """
    def forward(self, ds):
        return self(ds)


    def forward1(self, data):
        """Forward data or datasets through the chain.

        See `Mapper` for more information.
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

        See `Mapper` for more information.
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

        See `Mapper` for more information.
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
                          "Reversing single %s-shaped input though chain node '%s'."
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


    def train(self, dataset):
        """Train the mapper chain sequentially.

        The training dataset is used to train the first mapper. Afterwards it is
        forward-mapped by this (now trained) mapper and the transformed dataset
        and then used to train the next mapper. This procedure is done till all
        mappers are trained.

        Parameters
        ----------
        dataset: `Dataset`
        """
        nmappers = len(self) - 1
        tdata = dataset
        for i, mapper in enumerate(self):
            if __debug__:
                debug('MAP',
                      "Training child mapper (%i/%i) %s with %s-shaped input."
                      % (i + 1, nmappers + 1, str(mapper), tdata.shape))
            mapper.train(tdata)
            # forward through all but the last mapper
            if i < nmappers:
                tdata = mapper.forward(tdata)


    def untrain(self):
        """Untrain all embedded mappers."""
        for m in self:
            m.untrain()


    def __str__(self):
        return super(ChainMapper, self).__str__().replace('Mapper', '')


# XXX implement a 'CombinedMapper' (analog to ex-CombinedFeatureSelection) that
# can a dataset to multiple child-mappers and hstacks the results into a single
# output dataset
