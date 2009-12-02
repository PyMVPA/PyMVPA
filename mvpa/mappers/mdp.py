# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""MDP interface module.

This module provides to mapper that allow embedding MDP nodes, or flows
into PyMVPA.
"""

__docformat__ = 'restructuredtext'

import numpy as N
import mdp

from mvpa.datasets.base import DatasetAttributeExtractor
from mvpa.mappers.base import Mapper, accepts_dataset_as_samples
from mvpa.misc.support import isInVolume



class MDPNodeMapper(Mapper):
    """Mapper encapsulating an arbitray MDP node.

    This mapper wraps an MDP node and uses it for forward and reverse data
    mapping (reverse is only available if the underlying MDP node supports
    it).  It is possible to specify arbitrary arguments for all processing
    steps of an MDP node (training, training stop, execution, and
    inverse).

    Because MDP does not allow to 'reset' a node and (re)train it from
    scratch the mapper uses a copy of the initially wrapped node for the
    actual processing. Upon subsequent training attempts a new copy of the
    original node is made and replaces the previous one.

    Note
    ----
    MDP nodes requiring multiple training phases are not supported.
    Moreover, it is not possible to perform incremental training of a
    node.
    """
    def __init__(self, node, nodeargs=None, inspace=None):
        """
        Parameters
        ----------
        node : mdp.Node instance
          This node instance is taken as the pristine source of which a
          copy is made for actual processing upon each training attempt.
        nodeargs : dict
          Dictionary for additional arguments for all call to the MDP
          node. The dictionary key's meaning is as follows:

            'train': Arguments for calls to `Node.train()`
            'stoptrain': Arguments for calls to `Node.stop_training()`
            'exec': Arguments for calls to `Node.execute()`
            'inv': Arguments for calls to `Node.inverse()`

          The value for each item is always a 2-tuple, consiting of a
          tuple (for the arguments), and a dictonary (for keyword
          arguments), i.e.  ((), {}). Both, tuple and dictonary have to be
          provided even if they are empty.
        """
        # Tiziano will check if there can be/is a public way to do it
        if not len(node._train_seq) == 1:
            raise ValueError("MDPNodeMapper does not support MDP nodes with "
                             "multiple training phases.")
        Mapper.__init__(self, inspace=inspace)
        self.__pristine_node = None
        self.node = node
        self.nodeargs = nodeargs


    def _expand_args(self, phase, ds=None):
        args = []
        kwargs = {}
        if not self.nodeargs is None and phase in self.nodeargs:
            sargs, skwargs = self.nodeargs[phase]
            for a in sargs:
                if isinstance(a, DatasetAttributeExtractor):
                    if ds is None:
                        raise RuntimeError('MDPNodeMapper does not (yet) '
                                           'support argument extraction from dataset on '
                                           'forward()')
                    args.append(a(ds))
                else:
                    args.append(a)
            for k in skwargs:
                if isinstance(skwargs[k], DatasetAttributeExtractor):
                    if ds is None:
                        raise RuntimeError('MDPNodeMapper does not (yet) '
                                           'support argument extraction from dataset on '
                                           'forward()')
                    kwargs[k] = skwargs[k](ds)
                else:
                    kwargs[k] = skwargs[k]
        return args, kwargs


    def _train(self, ds):
        if not self.node.is_trainable():
            return

        # whenever we have no cannonical node source, we assign the current
        # node -- this can only happen prior training and allows modifying
        # the node of having the MDPNodeMapper instance
        if self.__pristine_node is None:
            self.__pristine_node = self.node

        # training is done on a copy of the pristine node, because nodes cannot
        # be reset, but PyMVPA's mapper need to be able to be retrained from
        # scratch
        self.node = self.__pristine_node.copy()
        # train
        args, kwargs = self._expand_args('train', ds)
        self.node.train(ds.samples, *args, **kwargs)
        # stop train
        args, kwargs = self._expand_args('stoptrain', ds)
        self.node.stop_training(*args, **kwargs)


    def _forward_data(self, data):
        args, kwargs = self._expand_args('exec', data)
        return self.node.execute(data, *args, **kwargs)


    def _reverse_data(self, data):
        args, kwargs = self._expand_args('inv', data)
        return self.node.inverse(data, *args, **kwargs)


    def get_insize(self):
        """Returns the node's input dim."""
        return self.node.input_dim


    def get_outsize(self):
        """Returns the node's output dim."""
        return self.node.output_dim


    def _get_outids(self, in_ids):
        return []



class MDPFlowMapper(Mapper):
    def __init__(self, flow, data_iterables=None, inspace=None):
        if not data_iterables is None and len(data_iterables) != len(flow):
            raise ValueError("Length of data_iterables (%i) does not match the "
                             "number of nodes in the flow (%i)."
                             % (len(data_iterables), len(flow)))
        Mapper.__init__(self, inspace=inspace)
        self.__pristine_flow = None
        self.flow = flow
        self.data_iterables = data_iterables


    def _expand_nodeargs(self, ds, args):
        enal = []
        for a in args:
            if isinstance(a, DatasetAttributeExtractor):
                enal.append(a(ds))
            else:
                enal.append(a)
        return enal


    def _build_data_iterables(self, ds):
        if self.data_iterables is not None:
            data_iterables = [[ds.samples] + self._expand_nodeargs(ds, ndi)
                                    for ndi in self.data_iterables]
        else:
            data_iterables = ds.samples
        return data_iterables


    def _train(self, ds):
        # whenever we have no cannonical node source, we assign the current
        # node -- this can only happen prior training and allow modifying
        # the node of having the MDPNodeMapper instance
        if self.__pristine_flow is None:
            self.__pristine_flow = self.flow

        # training is done on a copy of the pristine node, because nodes cannot
        # be reset, but PyMVPA's mapper need to be able to be retrained from
        # scratch
        self.flow = self.__pristine_flow.copy()
        self.flow.train(self._build_data_iterables(ds))


    def _forward_data(self, data):
        return self.flow.execute(data)


    def _reverse_data(self, data):
        return self.flow.inverse(data)


    def is_valid_outid(self, id):
        # untrained -- all is invalid
        outdim = self.get_outsize()
        if outdim is None:
            kwargs[k] = skwargs[k](ds)
        else:
            kwargs[k] = skwargs[k]
        return args, kwargs


    def is_valid_inid(self, id):
        # untrained -- all is invalid
        indim = self.get_insize()
        if indim is None:
            return False
        return id >= 0 and id < indim


    def get_insize(self):
        """Return the (flattened) size of input space vectors."""
        return self.flow[0].input_dim


    def get_outsize(self):
        """Return the size of output space vectors."""
        return self.flow[-1].output_dim


    def get_outids(self, in_ids=None, **kwargs):
        ourspace = self.get_inspace()
        # first contrain the set of in_ids if a known space is given
        if not ourspace is None and ourspace in kwargs:
            # XXX don't do anything for now -- we claim that we cannot
            # track features through the MDP node
            # remove the space contraint, since it has been processed
            del kwargs[ourspace]

        return ([], kwargs)
