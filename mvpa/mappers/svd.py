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

if __debug__:
    from mvpa.misc import debug

class SVDMapper(Mapper):
    """Mapper to project data onto SVD components estimated from some dataset.
    """
    def __init__(self, selector=None, demean=True):
        """Initialize the SVDMapper

        :Parameters:
            selector: None, list of ElementSelector
                Which SVD components should be used for mapping. If `selector`
                is `None` all components are used. If a list is provided, all
                list elements are treated as component ids and the respective
                components are selected (all others are discarded).
                Alternatively an `ElementSelector` instance can be provided
                which chooses components based on the corresponding eigenvalues
                of each component.
            demean: bool
                Either data should be demeaned while computing projections and
                applied back while doing reverse()
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
        self.__demean = demean
        self.mean = None
        """Data mean"""

    __doc__ = enhancedDocString('SVDMapper', locals(), Mapper)


    def __deepcopy__(self, memo=None):
        """Yes, this is it.
        XXX But do we need it really? copy.deepcopy wouldn't have a problem copying stuff
        """
        if memo is None:
            memo = {}
        out = SVDMapper()
        if self.mix is not None:
            out.mix = self.mix.copy()
            out.sv = self.sv.copy()
        if self.mean is not None:
            out.mean = self.mean.copy()

        return out


    def train(self, dataset):
        """Determine the projection matrix onto the SVD components from
        a 2D samples x feature data matrix.
        """
        X = N.asmatrix(dataset.samples)

        if self.__demean:
            # demean the training data
            self.mean = X.mean(axis=0)
            X = X - self.mean

            if __debug__:
                debug("MAP_",
                      "Mean of data in input space %s was subtracted" %
                      (self.mean))


        # singular value decomposition
        U, SV, Vh = N.linalg.svd(X, full_matrices=0)

        # store the final matrix with the new basis vectors to project the
        # features onto the SVD components. And store its .H right away to
        # avoid computing it in forward()
        self.mix = Vh.H

        # also store singular values of all components
        self.sv = SV

        if __debug__:
            debug("MAP", "SVD was done on %s and obtained %d SVs " %
                  (dataset, len(SV)) + " (%d non-0, max=%f)" %
                  (len(SV.nonzero()), SV[0]))

            debug("MAP_", "Mixing matrix has %s shape and norm=%f" %
                  (self.mix.shape, N.linalg.norm(self.mix)))

        if not self.__selector == None:
            if isinstance(self.__selector, list):
                self.selectOut(self.__selector)
            elif isinstance(self.__selector, ElementSelector):
                self.selectOut(self.__selector(SV))
            else:
                raise ValueError, \
                      'Unknown type of selector %s' % self.__selector


    def forward(self, data, demean=True):
        """Project a 2D samples x features matrix onto the SVD components.

        :Parameters:
            data: array
                Data arry to map
            demean: bool
                Flag whether to substract the training data mean before mapping.
                XXX: Not sure if this is the right place. Maybe better move to
                     constructor as it would be difficult to set this flag.
        :Returns:
          NumPy array
        """
        if self.mix is None:
            raise RuntimeError, "SVDMapper needs to be train before used."
        if demean and self.mean is not None:
            return ((N.asmatrix(data) - self.mean)*self.mix).A
        else:
            return (N.asmatrix(data) * self.mix).A


    def reverse(self, data):
        """Projects feature vectors or matrices with feature vectors back
        onto the original features.

        :Returns:
          NumPy array
        """
        if self.mix is None:
            raise RuntimeError, "SVDMapper needs to be train before used."

        if self.unmix is None:
            self.unmix = self.mix.H

            if self.__demean:
                # XXX: why store in object if computed all the time?
                self.mean_out = self.forward(self.mean, demean=False)
                if __debug__:
                    debug("MAP_",
                          "Mean of data in input space %s bacame %s in " \
                          "outspace" % (self.mean, self.mean_out))

        if self.__demean:
            return ((N.asmatrix(data) + self.mean_out) * self.unmix).A
        else:
            return ((N.asmatrix(data)) * self.unmix).A


    def getInShape(self):
        """Returns a one-tuple with the number of original features."""
        return (self.mix.shape[0], )


    def getOutShape(self):
        """Returns a one-tuple with the number of SVD components."""
        return (self.mix.shape[1], )


    def getInSize(self):
        """Returns the number of original features."""
        return self.mix.shape[0]


    def getOutSize(self):
        """Returns the number of SVD components."""
        return self.mix.shape[1]


    def selectOut(self, outIds):
        """Choose a subset of SVD components (and remove all others)."""
        self.mix = self.mix[:, outIds]
        # invalidate unmixing matrix
        self.unmix = None
