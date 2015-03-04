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
from mvpa2.base.types import is_datasetlike

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
        ProjectionMapper.__init__(self, auto_train=kwargs.pop('auto_train', True), **kwargs)
        self._proj = proj
        self._recon = recon

    def _train(self, dummyds):
        """Do Nothing
        """
        if __debug__:
            debug("MAP_", "Mixing matrix has %s shape and norm=%f" %
                  (self._proj.shape, np.linalg.norm(self._proj)))



class StaticProjectionMapperWithAttr(StaticProjectionMapper):
    """
    Extends StaticProjectionMapper with the ability to add
    feature attributes during forward mapping.

    Parameters
    ----------
    add_fa : Dictionary of features attributes to be added
      when forwarding a dataset.
    **kwargs:
      All keyword arguments are passed to the StaticProjectionMapper
      constructor.
    """
    @borrowdoc(StaticProjectionMapper)
    def __init__(self, proj, recon=None, add_fa=None, **kwargs):
        StaticProjectionMapper.__init__(self, proj=proj, recon=recon, **kwargs)
        self._add_fa = add_fa

    def __train(self, dummyds):
        if __debug__:
            if self.add_fa is None:
                 debug("MAP_", "Mixing matrix has no additional feature attributes")
            else:
                 debug("MAP_", "Mixing matrix has additional attributes to apply %s" %
                      (self._add_fa.keys()))

    def _forward_dataset(self, data):
        res = StaticProjectionMapper._forward_dataset(self, data)
        if is_datasetlike(res) and self._add_fa is not None:
            for key in self._add_fa:
                res.fa[key] = self._add_fa[key]
        return res



