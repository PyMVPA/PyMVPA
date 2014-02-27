# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Transform data via static projection matrices"""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.base.dochelpers import borrowdoc
from mvpa2.mappers.projection import ProjectionMapper

if __debug__:
    from mvpa2.base import debug


class StaticProjectionMapper(ProjectionMapper):
    """Mapper to project data onto arbitrary space using transformation given as input.
       Both forward and reverse projections can be provided.
    """

    def __init__(self, proj, recon=None, **kwargs):
        """Initialize the StaticProjectionMapper

        Parameters
        ----------
        proj : 2-D array
          Projection matrix to be used for forward projection.
        recon: 2-D array
          Projection matrix to be used for reverse projection.
          If this is not given, `numpy.linalg.pinv` of proj
          will be used by default.
        **kwargs:
          All keyword arguments are passed to the ProjectionMapper
          constructor.
        """
        ProjectionMapper.__init__(self, auto_train=True, **kwargs)
        self._proj = proj
        self._recon = recon

    def _train(self, dummyds):
        """Do Nothing
        """
        if __debug__:
            debug("MAP_", "Mixing matrix has %s shape and norm=%f" %
                  (self._proj.shape, np.linalg.norm(self._proj)))



