#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Local Linear Embedding Data mapper.

This is a wrapper class around the corresponding MDP nodes LLE and HLLE
(since MDP 2.4).
"""

__docformat__ = 'restructuredtext'

from mvpa.base import externals

import numpy as N

from mvpa.mappers.base import Mapper

if externals.exists('mdp >= 2.4', raiseException=True):
    from mdp.nodes import LLENode, HLLENode


class LLEMapper(Mapper):
    """Locally linear embbeding Mapper.

    This mapper performs dimensionality reduction. It wraps two algorithms
    provided by the Modular Data Processing (MDP) framework.

    Locally linear embedding (LLE) approximates the input data with a
    low-dimensional surface and reduces its dimensionality by learning a
    mapping to the surface.

    This wrapper class provides access to two different LLE algorithms (i.e.
    the corresponding MDP processing nodes). 1) An algorithm outlined in *An
    Introduction to Locally Linear Embedding* by L. Saul and S. Roweis, using
    improvements suggested in *Locally Linear Embedding for Classification* by
    D. deRidder and R.P.W. Duin (aka `LLENode`) and 2) Hessian Locally Linear
    Embedding analysis based on algorithm outlined in *Hessian Eigenmaps: new
    locally linear embedding techniques for high-dimensional data* by C. Grimes
    and D. Donoho, 2003.

    .. note::
      This mapper only provides forward-mapping functionality -- no reverse
      mapping is available.

    .. seealso::
      http://mdp-toolkit.sourceforge.net
    """
    def __init__(self, k, algorithm='lle', **kwargs):
        """
        :Parameters:
          k: int
            Number of nearest neighbor to be used by the algorithm.
          algorithm: 'lle' | 'hlle'
            Either use the standard LLE algorithm or Hessian Linear Local
            Embedding (HLLE).
          **kwargs:
            Additional arguments are passed to the underlying MDP node.
            Most importantly this is the `output_dim` argument, that determines
            the number of dimensions to mapper is using as output space.
        """
        # no meaningful metric
        Mapper.__init__(self, metric=None)

        self._algorithm = algorithm
        self._node_kwargs = kwargs
        self._k = k
        self._node = None


    def train(self, ds):
        """Train the mapper.
        """
        if self._algorithm == 'lle':
            self._node = LLENode(self._k, dtype=ds.samples.dtype,
                                 **self._node_kwargs)
        elif self._algorithm == 'hlle':
            self._node = HLLENode(self._k, dtype=ds.samples.dtype,
                                  **self._node_kwargs)
        else:
            raise NotImplementedError

        self._node.train(ds.samples)
        self._node.stop_training()


    def forward(self, data):
        """Map data from the IN dataspace into OUT space.
        """
        # experience the beauty of MDP -- just call the beast and be done ;-)
        return self.node(data)


    def reverse(self, data):
        """Reverse map data from OUT space into the IN space.
        """
        raise NotImplementedError


    def getInSize(self):
        """Returns the size of the entity in input space"""
        return self.node.input_dim


    def getOutSize(self):
        """Returns the size of the entity in output space"""
        return self.node.output_dim


    def _accessNode(self):
        """Provide access to the underlying MDP processing node.

        With some care.
        """
        if self._node is None:
            raise RuntimeError, \
                  'The LLEMapper needs to be trained before access to the ' \
                  'processing node is possible.'

        return self._node


    node = property(fget=_accessNode)
