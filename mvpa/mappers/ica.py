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
from mvpa.mappers.base import Mapper

from mdp.nodes import FastICANode


class ICAMapper(Mapper):
    """Mapper to project data onto ICA components estimated from some dataset.

    After the mapper has been instantiated, it has to be train first. The ICA
    mapper only handles 2D data matrices.
    """
    def __init__(self, selector=None):
        Mapper.__init__(self)

        self.__selector = selector
        self.proj = None
        self.recon = None

    __doc__ = enhancedDocString('ICAMapper', locals(), Mapper)


    def train(self, dataset):
        """Determine the projection matrix onto the components from
        a 2D samples x feature data matrix.
        """
        node = FastICANode(dtype='float')

        # more features than samples?
        if dataset.samples.shape[1] > dataset.samples.shape[0]:
            node.train(dataset.samples.T)
            raise NotImplementedError
            #self.proj = (N.asmatrix(node.get_projmatrix()) * dataset.samples).T
            #self.recon = N.asmatrix(node.get_recmatrix()) * dataset.samples
        else:
            node.train(dataset.samples)
            self.proj = N.asmatrix(node.get_projmatrix())
            self.recon = N.asmatrix(node.get_recmatrix())


    def forward(self, data):
        """Project a 2D samples x features matrix onto the PCA components.

        :Returns:
          NumPy array
        """
        if self.proj is None:
            raise RuntimeError, \
                  "ProjectionMapper needs to be train before used."

        return (N.asmatrix(data) * self.proj).A


    def reverse(self, data):
        """Projects feature vectors or matrices with feature vectors back
        onto the original features.

        :Returns:
          NumPy array
        """
        return (N.asmatrix(data) * self.recon).A


    def getInShape(self):
        """Returns a one-tuple with the number of original features."""
        return (self.mix.shape[0], )


    def getOutShape(self):
        """Returns a one-tuple with the number of components."""
        return (self.mix.shape[1], )


    def getInSize(self):
        """Returns the number of original features."""
        return self.mix.shape[0]


    def getOutSize(self):
        """Returns the number of components."""
        return self.mix.shape[1]


    def selectOut(self, outIds):
        """Choose a subset of PCA components (and remove all others)."""
        self.proj = self.proj[:, outIds]
        self.recon = self.recon[outIds]
