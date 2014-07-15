# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Project data onto a space defined by prototypes via a similarity function"""

import numpy as np

from mvpa2.mappers.base import accepts_dataset_as_samples
from mvpa2.mappers.projection import ProjectionMapper

if __debug__:
    from mvpa2.base import debug


class PrototypeMapper(ProjectionMapper):
    """Mapper to project data onto a space defined by prototypes from
    the same space via a similarity function.
    """
    def __init__(self,
                 similarities,
                 prototypes=None,
                 **kwargs):
        """
        Parameters
        ----------
        similarities : list
          A list of similarity functions.
        prototypes : Dataset or list
          A dataset or a list of instances (e.g., streamlines)?
        **kwargs:
          All keyword arguments are passed to the ProjectionMapper
          constructor
        """
        ProjectionMapper.__init__(self, **kwargs)

        self.similarities = similarities
        self.prototypes = prototypes


    @accepts_dataset_as_samples
    def _train(self, samples):
        """Train PrototypeMapper
        """

        self._proj = np.hstack([similarity.computed(samples, self.prototypes)
                               for similarity in self.similarities])
        if __debug__:
            debug("MAP", "projected data of shape %s: %s "
                  % (self._proj.shape, self._proj))
