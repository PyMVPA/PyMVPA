# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Prototype-based Mapper."""

import numpy as N

from mvpa.base import warning
from mvpa.base.dochelpers import enhancedDocString
from mvpa.mappers.base import accepts_dataset_as_samples
from mvpa.mappers.projection import ProjectionMapper

if __debug__:
    from mvpa.base import debug


class PrototypeMapper(ProjectionMapper):
    """Mapper to project data onto a space defined by prototypes from
    the same space via a similarity function.
    """
    def __init__(self,
                 similarities,
                 prototypes=None,
                 **kwargs):
        """Initialize the ProjectionMapper

        :Parameters:

          similarities : a list of similarity functions.
        
          prototypes : a dataset or a list of instances (e.g.,
            streamlines)?

          **kwargs:
            All keyword arguments are passed to the ProjectionMapper
            constructor
        """
        ProjectionMapper.__init__(self, **kwargs)

        self.similarities = similarities
        self.prototypes = prototypes


    __doc__ = enhancedDocString('PrototypeMapper', locals(), ProjectionMapper)


    @accepts_dataset_as_samples
    def _train(self, samples):
        """Compute similarities between instances in dataset and
        prototypes using the provided similarity functions.
        
        :Parameters:

          dataset : the Dataset to project over prototypes.

          fraction : when prototypes are not explicitely given use a
            random subset of dataset whose size is a fraction of it.
        """
        
        self._proj = N.hstack([similarity.compute(samples,self.prototypes) for similarity in self.similarities])
        debug("MAP","projected data: "+str(self._proj))
        debug("MAP","Projected data is"+str(self._proj.shape))
        
