# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Static Projection mapper"""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.base.dochelpers import borrowdoc
from mvpa2.mappers.projection import ProjectionMapper

if __debug__:
    from mvpa2.base import debug


class StaticProjectionMapper(ProjectionMapper):
    """Mapper to project data onto arbitrary space using transformation given as input.
    """

    @borrowdoc(ProjectionMapper)
    def __init__(self, proj, **kwargs):
        """Initialize the StaticProjectionMapper

        Parameters
        ----------
        **kwargs:
          All keyword arguments are passed to the ProjectionMapper
          constructor.

        """
        ProjectionMapper.__init__(self, **kwargs)

        self._proj = proj

    def _train(self):
        """Do Nothing
        """
        if __debug__:
            if "MAP_" in debug.active:
                debug("MAP_", "Mixing matrix has %s shape and norm=%f" %
                      (self._proj.shape, np.linalg.norm(self._proj)))



    def _compute_recon(self):
        """Computing the inverse of the projection matrix for reverse
        """
        return np.linalg.pinv(self._proj)


