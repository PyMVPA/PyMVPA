# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wrapper to use MDP nodes and flows as PyMVPA mappers.

This module provides to mapper that allow embedding MDP nodes, or flows
into PyMVPA.
"""

__docformat__ = 'restructuredtext'

from mvpa2.base import externals
if externals.exists('mdp', raise_=True):
    import mdp

import numpy as np

from mvpa2.base.dataset import DatasetAttributeExtractor
from mvpa2.mappers.base import Mapper, accepts_dataset_as_samples
from mvpa2.misc.support import is_in_volume


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

    Notes
    -----
    MDP nodes requiring multiple training phases are not supported. Use a
    MDPFlowWrapper for that. Moreover, it is not possible to perform
    incremental training of a node.
    """
    def __init__(self, node, nodeargs=None, **kwargs):
        """
        Parameters
        ----------
        node : mdp.Node instance
          This node instance is taken as the pristine source of which a
          copy is made for actual processing upon each training attempt.
        nodeargs : dict
          Dictionary for additional arguments for all calls to the MDP
          node. The dictionary key's meaning is as follows:
          
          'train'
            Arguments for calls to `Node.train()`
          'stoptrain'
            Arguments for calls to `Node.stop_training()`
          'exec'
            Arguments for calls to `Node.execute()`
          'inv'
            Arguments for calls to `Node.inverse()`
          
          The value for each item is always a 2-tuple, consisting of a
          tuple (for the arguments), and a dictionary (for keyword
          arguments), i.e.  ((), {}). Both, tuple and dictionary have to be
          provided even if they are empty.
        space : see base class
        """
        # NOTE: trailing spaces in above docstring must not be pruned
        # for correct parsing

        if (externals.versions['mdp'] >= (2, 5) \
                and node.has_multiple_training_phases()) \
            or not len(node._train_seq) == 1:
            raise ValueError("MDPNodeMapper does not support MDP nodes with "
                             "multiple training phases.")
        Mapper.__init__(self, **kwargs)
        self.__pristine_node = None
        self.node = node
        self.nodeargs = nodeargs


    def __repr__(self):
        s = super(MDPNodeMapper, self).__repr__()
        return s.replace("(", "(node=%s, nodeargs=%s, "
                              % (repr(self.node),
                                 repr(self.nodeargs)), 1)


    def _expand_args(self, phase, ds=None):
        args = []
        kwargs = {}
        if self.nodeargs is not None and phase in self.nodeargs:
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
        return self.node.execute(np.atleast_2d(data), *args, **kwargs).squeeze()


    def _reverse_data(self, data):
        args, kwargs = self._expand_args('inv', data)
        return self.node.inverse(np.atleast_2d(data), *args, **kwargs).squeeze()



class PCAMapper(MDPNodeMapper):
    """Convenience wrapper to perform PCA using MDP's Mapper
    """

    def __init__(self, alg='PCA', nodeargs=None, **kwargs):
        """
        Parameters
        ----------
        alg : {'PCA', 'NIPALS'}
          Which MDP implementation of a PCA to use.
        nodeargs : None or dict
          Arguments passed to the MDP node in various stages of its lifetime.
          See the :class:`MDPNodeMapper` for more details.
        **kwargs
          Additional constructor arguments for the MDP node.
        """
        if alg == 'PCA':
            node = mdp.nodes.PCANode(**kwargs)
        elif alg == 'NIPALS':
            node = mdp.nodes.NIPALSNode(**kwargs)
        else:
            raise ValueError("Unkown algorithm '%s' for PCAMapper."
                             % alg)
        MDPNodeMapper.__init__(self, node, nodeargs=nodeargs)


    proj = property(fget=lambda self: self.node.get_projmatrix(),
                    doc="Projection matrix (as an array)")
    recon = property(fget=lambda self: self.node.get_projmatrix(),
                     doc="Backprojection matrix (as an array)")
    var = property(fget=lambda self: self.node.d, doc="Variances per component")
    centroid = property(fget=lambda self: self.node.avg,
                        doc="Mean of the training data")


class ICAMapper(MDPNodeMapper):
    """Convenience wrapper to perform ICA using MDP nodes.
    """
    def __init__(self, alg='FastICA', nodeargs=None, **kwargs):
        """
        Parameters
        ----------
        alg : {'FastICA', 'CuBICA'}
          Which MDP implementation of an ICA to use.
        nodeargs : None or dict
          Arguments passed to the MDP node in various stages of its lifetime.
          See the baseclass for more details.
        **kwargs
          Additional constructor arguments for the MDP node.
        """
        if alg == 'FastICA':
            node = mdp.nodes.FastICANode(**kwargs)
        elif alg == 'CuBICA':
            node = mdp.nodes.CuBICANode(*kwargs)
        else:
            raise ValueError("Unkown algorithm '%s' for ICAMapper."
                             % alg)
        MDPNodeMapper.__init__(self, node, nodeargs=nodeargs)


    proj = property(fget=lambda self: self.node.get_projmatrix(),
                    doc="Projection matrix (as an array)")
    recon = property(fget=lambda self: self.node.get_projmatrix(),
                     doc="Backprojection matrix (as an array)")



class MDPFlowMapper(Mapper):
    """Mapper encapsulating an arbitray MDP flow.

    This mapper wraps an MDP flow and uses it for forward and reverse data
    mapping (reverse is only available if the underlying MDP flow supports
    it).  It is possible to specify arbitrary arguments for the training of
    the MDP flow.

    Because MDP does not allow to 'reset' a flow and (re)train it from
    scratch the mapper uses a copy of the initially wrapped flow for the
    actual processing. Upon subsequent training attempts a new copy of the
    original flow is made and replaces the previous one.

    Examples
    --------
    >>> import mdp
    >>> from mvpa2.mappers.mdp_adaptor import MDPFlowMapper
    >>> from mvpa2.base.dataset import DAE
    >>> flow = (mdp.nodes.PCANode() + mdp.nodes.IdentityNode() +
    ...         mdp.nodes.FDANode())
    >>> mapper = MDPFlowMapper(flow,
    ...                        node_arguments=(None, None,
    ...                        [DAE('sa', 'targets')]))

    Notes
    -----
    It is not possible to perform incremental training of the MDP flow. 
    """
    def __init__(self, flow, node_arguments=None, **kwargs):
        """
        Parameters
        ----------
        flow : mdp.Flow instance
          This flow instance is taken as the pristine source of which a
          copy is made for actual processing upon each training attempt.
        node_arguments : tuple, list
          A tuple or a list the same length as the flow. Each item is a
          list of arguments for the training of the corresponding node in
          the flow. If a node does not require additional arguments, None
          can be provided instead. Keyword arguments are currently not
          supported by mdp.Flow.
        """
        if node_arguments is not None and len(node_arguments) != len(flow):
            raise ValueError("Length of node_arguments (%i) does not match the "
                             "number of nodes in the flow (%i)."
                             % (len(node_arguments), len(flow)))
        Mapper.__init__(self, **kwargs)
        self.__pristine_flow = None
        self.flow = flow
        self.node_arguments = node_arguments


    def __repr__(self):
        s = super(MDPFlowMapper, self).__repr__()
        return s.replace("(", "(flow=%s, node_arguments=%s, "
                              % (repr(self.flow),
                                 repr(self.node_arguments)), 1)


    def _expand_nodeargs(self, ds, args):
        enal = []
        for a in args:
            if isinstance(a, DatasetAttributeExtractor):
                enal.append(a(ds))
            else:
                enal.append(a)
        return enal


    def _build_node_arguments(self, ds):
        if self.node_arguments is not None:
            node_arguments = []
            for ndi in self.node_arguments:
                l = [ds.samples]
                if ndi is not None:
                    l = [ds.samples]
                    l.extend(self._expand_nodeargs(ds, ndi))
                node_arguments.append([l])
        else:
            node_arguments = ds.samples
        return node_arguments


    def _train(self, ds):
        # whenever we have no cannonical flow source, we assign the current
        # flow -- this can only happen prior training and allow modifying
        # the flow of having the MDPNodeMapper instance
        if self.__pristine_flow is None:
            self.__pristine_flow = self.flow

        # training is done on a copy of the pristine flow, because flows cannot
        # be reset, but PyMVPA's mapper need to be able to be retrained from
        # scratch
        self.flow = self.__pristine_flow.copy()
        self.flow.train(self._build_node_arguments(ds))


    def _forward_data(self, data):
        return self.flow.execute(np.atleast_2d(data)).squeeze()


    def _reverse_data(self, data):
        return self.flow.inverse(np.atleast_2d(data)).squeeze()
