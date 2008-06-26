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

from mvpa.base.dochelpers import enhancedDocString
from mvpa.mappers.base import ProjectionMapper

from mdp.nodes import NIPALSNode


class PCAMapper(ProjectionMapper):
    """Mapper to project data onto PCA components estimated from some dataset.

    After the mapper has been instantiated, it has to be train first. The PCA
    mapper only handles 2D data matrices.
    """
    def __init__(self, transpose=False, **kwargs):
        ProjectionMapper.__init__(self, **kwargs)

        self._var = None


    __doc__ = enhancedDocString('PCAMapper', locals(), ProjectionMapper)


    def _train(self, dataset):
        """Determine the projection matrix onto the components from
        a 2D samples x feature data matrix.
        """
        node = NIPALSNode(dtype=dataset.samples.dtype)
        node.train(dataset.samples)
        self._proj = N.asmatrix(node.get_projmatrix())
        self._recon = N.asmatrix(node.get_recmatrix())

        # store variance per PCA component
        self._var = node.d


    var = property(fget=lambda self: self._var, doc='Variances per component')
