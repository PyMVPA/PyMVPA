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


class PCAMapper(Mapper):
    """Mapper to project data onto PCA components estimated from some dataset.

    After the mapper has been instantiated, it has to be train first. When
    `train()` is called with a 2D (samples x features) matrix the PCA
    components are determined by performing singular value decomposition
    on the covariance matrix.

    The PCA mapper only handle 2D data matrices.
    """
    def __init__(self):
        """Does nothing."""
        Mapper.__init__(self)

        self.mix = None
        """Transformation matrix from orginal features onto PCA-components."""
        self.unmix = None
        """Un-mixing matrix for projecting from the PCA space back onto the
        original features."""
        self.sv = None
        """Eigenvalues of the covariance matrix."""


    __doc__ = enhancedDocString('PCAMapper', locals(), Mapper)


    def __deepcopy__(self, memo=None):
        """Yes, this is it."""
        if memo is None:
            memo = {}
        out = PCAMapper()
        out.mix = self.mix.copy()
        out.sv = self.sv.copy()

        return out


    def train(self, data):
        """Determine the projection matrix onto the PCA components from
        a 2D samples x feature data matrix.
        """
        # transpose the data to minimize the number of columns and therefore
        # reduce the size of the covariance matrix!
        transposed_data = False
        if data.shape[0] < data.shape[1]:
            transposed_data = True
            X = N.matrix(data).T
        else:
            X = N.matrix(data)

        # compute covariance matrix
        R = X.T * X / X.shape[0]

        # singular value decomposition
        # note: U and V are equal in this case, as R is a covanriance matrix
        U, SV, V = N.linalg.svd(R)

        # store the final matrix with the new basis vextors to project the
        # features onto the PCA components
        if transposed_data:
            self.mix = (X * U).T
        else:
            self.mix = U.T

        # also store eigenvalues of all components
        self.sv = SV


    def forward(self, data):
        """Project a 2D samples x features matrix onto the PCA components.

        :Returns:
          NumPy array
        """
        return N.array(self.mix * N.matrix(data).T).T


    def reverse(self, data):
        """Projects feature vectors or matrices with feature vectors back
        onto the original features.

        :Returns:
          NumPy array
        """
        if self.unmix == None:
            self.unmix = self.mix.I
        return (self.unmix * N.matrix(data).T).T.A


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
