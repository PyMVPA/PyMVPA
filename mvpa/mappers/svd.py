# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Singular-value decomposition mapper"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.base.dochelpers import enhancedDocString
from mvpa.mappers.base import ProjectionMapper
from mvpa.featsel.helpers import ElementSelector

if __debug__:
    from mvpa.base import debug


class SVDMapper(ProjectionMapper):
    """Mapper to project data onto SVD components estimated from some dataset.
    """
    def __init__(self, **kwargs):
        """Initialize the SVDMapper

        :Parameters:
            **kwargs:
                All keyword arguments are passed to the ProjectionMapper
                constructor.

                Note, that for the 'selector' argument this class also supports
                passing a `ElementSelector` instance, which will be used to
                determine the to be selected features, based on the singular
                values of each component.
        """
        ProjectionMapper.__init__(self, **kwargs)

        self._sv = None
        """Singular values of the training matrix."""

    __doc__ = enhancedDocString('SVDMapper', locals(), ProjectionMapper)


    def _train(self, dataset):
        """Determine the projection matrix onto the SVD components from
        a 2D samples x feature data matrix.
        """
        X = N.asmatrix(dataset.samples)
        X = self._demeanData(X)

        # singular value decomposition
        U, SV, Vh = N.linalg.svd(X, full_matrices=0)

        # store the final matrix with the new basis vectors to project the
        # features onto the SVD components. And store its .H right away to
        # avoid computing it in forward()
        self._proj = Vh.H

        # also store singular values of all components
        self._sv = SV

        if __debug__:
            debug("MAP", "SVD was done on %s and obtained %d SVs " %
                  (dataset, len(SV)) + " (%d non-0, max=%f)" %
                  (len(SV.nonzero()), SV[0]))

            debug("MAP_", "Mixing matrix has %s shape and norm=%f" %
                  (self._proj.shape, N.linalg.norm(self._proj)))


    def selectOut(self, outIds):
        """Choose a subset of SVD components (and remove all others)."""
        # handle ElementSelector operating on SV (base class has no idea about)
        if isinstance(self._selector, ElementSelector):
            ProjectionMapper.selectOut(self, self._selector(self._sv))
        else:
            ProjectionMapper.selectOut(self, outIds)


    sv = property(fget=lambda self: self._sv, doc="Singular values")
