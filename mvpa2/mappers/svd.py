# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Singular-value decomposition"""

__docformat__ = 'restructuredtext'

import numpy as np
#import scipy.linalg as spl

from mvpa2.base.dochelpers import borrowdoc
from mvpa2.mappers.base import accepts_dataset_as_samples
from mvpa2.mappers.projection import ProjectionMapper
from mvpa2.featsel.helpers import ElementSelector

if __debug__:
    from mvpa2.base import debug


class SVDMapper(ProjectionMapper):
    """Mapper to project data onto SVD components estimated from some dataset.
    """

    @borrowdoc(ProjectionMapper)
    def __init__(self, **kwargs):
        """Initialize the SVDMapper

        Parameters
        ----------
        **kwargs:
          All keyword arguments are passed to the ProjectionMapper
          constructor.

        """
        ProjectionMapper.__init__(self, **kwargs)

        self._sv = None
        """Singular values of the training matrix."""


    @accepts_dataset_as_samples
    def _train(self, samples):
        """Determine the projection matrix onto the SVD components from
        a 2D samples x feature data matrix.
        """
        X = np.asmatrix(samples)
        X = self._demean_data(X)

        # singular value decomposition
        U, SV, Vh = np.linalg.svd(X, full_matrices=0)
        #U, SV, Vh = spl.svd(X, full_matrices=0)

        # store the final matrix with the new basis vectors to project the
        # features onto the SVD components. And store its .H right away to
        # avoid computing it in forward()
        self._proj = Vh.H

        # also store singular values of all components
        self._sv = SV

        if __debug__:
            debug("MAP", "SVD was done on %s and obtained %d SVs " %
                  (samples, len(SV)) + " (%d non-0, max=%f)" %
                  (len(SV.nonzero()), SV[0]))
            # .norm might be somewhat expensive to compute
            if "MAP_" in debug.active:
                debug("MAP_", "Mixing matrix has %s shape and norm=%f" %
                      (self._proj.shape, np.linalg.norm(self._proj)))


    ##REF: Name was automagically refactored
    def _compute_recon(self):
        """Since singular vectors are orthonormal, sufficient to take hermitian
        """
        if issubclass(self._proj.dtype.type, np.complexfloating):
            return self._proj.T.conjugate()
        else:
            return self._proj.T


    sv = property(fget=lambda self: self._sv, doc="Singular values")
