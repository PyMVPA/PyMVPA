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
from mvpa.featsel.helpers import ElementSelector


class SVDMapper(Mapper):
    """Mapper to project data onto SVD components estimated from some dataset.
    """
    def __init__(self, selector=None):
        """Initialize the PCAMapper

        :Parameters:
            selector: None, list of ElementSelector
                Which SVD components should be used for mapping. If `selector`
                is `None` all components are used. If a list is provided, all
                list elements are treated as component ids and the respective
                components are selected (all others are discarded).
                Alternatively an `ElementSelector` instance can be provided
                which chooses components based on the corresponding eigenvalues
                of each component.
        """
        Mapper.__init__(self)

        self.__selector = selector
        self.mix = None
        """Transformation matrix from orginal features onto SVD-components."""
        self.unmix = None
        """Un-mixing matrix for projecting from the SVD space back onto the
        original features."""
        self.sv = None
        """Singular values of the training matrix."""


    __doc__ = enhancedDocString('SVDMapper', locals(), Mapper)


    def __deepcopy__(self, memo=None):
        """Yes, this is it."""
        if memo is None:
            memo = {}
        out = SVDMapper()
        if self.mix is not None:
            out.mix = self.mix.copy()
            out.sv = self.sv.copy()

        return out


    def train(self, dataset):
        """Determine the projection matrix onto the SVD components from
        a 2D samples x feature data matrix.
        """
        X = N.asmatrix(dataset.samples)

        # demean the training data
        X = X - X.mean(axis=0)

        # singular value decomposition
        U, SV, Vh = N.linalg.svd(X, full_matrices=0)

        # store the final matrix with the new basis vectors to project the
        # features onto the PCA components
        self.mix = Vh

        # also store singular values of all components
        self.sv = SV

        if not self.__selector == None:
            if isinstance(self.__selector, list):
                self.selectOut(self.__selector)
            elif isinstance(self.__selector, ElementSelector):
                self.selectOut(self.__selector(SV))
            else:
                raise ValueError, 'Unknown type of selector.'


    def forward(self, data):
        """Project a 2D samples x features matrix onto the PCA components.

        :Returns:
          NumPy array
        """
        if self.mix is None:
            raise RuntimeError, "PCAMapper needs to be train before used."

        return N.asarray(self.mix * N.asmatrix(data).T).T


    def reverse(self, data):
        """Projects feature vectors or matrices with feature vectors back
        onto the original features.

        :Returns:
          NumPy array
        """
        if self.mix is None:
            raise RuntimeError, "PCAMapper needs to be train before used."

        if self.unmix is None:
            # XXX yoh: should be simply H instead of I
            self.unmix = self.mix.H

        return (self.unmix * N.asmatrix(data).T).T.A


    def getInShape(self):
        """Returns a one-tuple with the number of original features."""
        return (self.mix.shape[1], )


    def getOutShape(self):
        """Returns a one-tuple with the number of PCA components."""
        return (self.mix.shape[0], )


    def getInSize(self):
        """Returns the number of original features."""
        return self.mix.shape[1]


    def getOutSize(self):
        """Returns the number of PCA components."""
        return self.mix.shape[0]


    def selectOut(self, outIds):
        """Choose a subset of PCA components (and remove all others)."""
        self.mix = self.mix[outIds]
        # invalidate unmixing matrix
        self.unmix = None
