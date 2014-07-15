# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Local Linear Embedding.

This is a wrapper class around the corresponding MDP nodes LLE and HLLE
(since MDP 2.4).
"""

__docformat__ = 'restructuredtext'

from mvpa2.base import externals

import numpy as np

from mvpa2.mappers.mdp_adaptor import MDPNodeMapper

if externals.exists('mdp ge 2.4', raise_=True):
    import mdp


class LLEMapper(MDPNodeMapper):
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
    D. deRidder and R.pl.W. Duin (aka `LLENode`) and 2) Hessian Locally Linear
    Embedding analysis based on algorithm outlined in *Hessian Eigenmaps: new
    locally linear embedding techniques for high-dimensional data* by C. Grimes
    and D. Donoho, 2003.

    For more information see the MDP website at
    http://mdp-toolkit.sourceforge.net

    Notes
    -----
    This mapper only provides forward-mapping functionality -- no reverse
    mapping is available.
    """
    def __init__(self, k, alg='LLE', nodeargs=None, **kwargs):
        """
        Parameters
        ----------
        k : int
          Number of nearest neighbors to be used by the algorithm.
        algorithm : {'LLE', 'HLLE'}
          Either use the standard LLE algorithm or Hessian Linear Local
          Embedding (HLLE).
        nodeargs : None or dict
          Arguments passed to the MDP node in various stages of its lifetime.
          See the baseclass for more details.
        **kwargs
          Additional constructor arguments for the MDP node.
        """
        if alg == 'LLE':
            node = mdp.nodes.LLENode(k, **kwargs)
        elif alg == 'HLLE':
            node = mdp.nodes.HLLENode(k, **kwargs)
        else:
            raise ValueError("Unkown algorithm '%s' for LLEMapper.")

        MDPNodeMapper.__init__(self, node, nodeargs=nodeargs)
