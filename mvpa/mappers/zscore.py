# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Simple mapper to perform zscoring"""

__docformat__ = 'restructuredtext'

from mvpa.base import warning, externals

import numpy as N
from mvpa.base.dochelpers import enhancedDocString
from mvpa.mappers.base import ProjectionMapper

if externals.exists('scipy', raiseException=True):
    import scipy.sparse
    if externals.versions['scipy'] >= (0, 7, 0):
        _identity = scipy.sparse.identity
    else:
        _identity = scipy.sparse.spidentity


__all__ = [ 'ZScoreMapper' ]            # just to don't leak all the evil ;)

class ZScoreMapper(ProjectionMapper):
    """Mapper to project data into standardized values (z-scores).

    After the mapper has been instantiated, it has to be train first.

    Since it tries to reuse ProjectionMapper, invariant features will
    simply be assigned a std == 1, which would be equivalent to not
    standardizing them at all.  This is similar to not touching them
    at all, so similar to what zscore function currently does
    """
    def __init__(self,
                 baselinelabels=None,
                 **kwargs):
        """Initialize ZScoreMapper

        :Parameters:
          baselinelabels : None or list of int or string
            Used to compute mean and variance
            TODO: not in effect now and need to be adherent to all
            `ProjectionMapper`s
        """

        ProjectionMapper.__init__(self, **kwargs)
        if baselinelabels is not None:
            raise NotImplementedError, "Support for baseline labels " \
                  "is not yet implemented in ZScoreMapper"
        self.baselinelabels = baselinelabels
        #self._var = None


    __doc__ = enhancedDocString('ZScoreMapper', locals(), ProjectionMapper)


    def _train(self, dataset):
        """Determine the diagonal matrix with coefficients for standartization
        """
        samples = dataset.samples
        X = self._demeanData(samples)
        std = X.std(axis=0)

        # ??? equivalent to not touching values at all, but we don't
        #     have such ability in ProjectionMapper atm afaik
        std[std == 0] = 1.0
        n = len(std)

        # scipy or numpy manages to screw up:
        # or YOH is too tired?:
        # (Pydb) zsm._proj
        # <1x1 sparse matrix of type '<type 'numpy.float64'>'
        #         with 1 stored elements (space for 1)
        #         in Compressed Sparse Column format>
        # *(Pydb) (N.asmatrix(ds1.samples) - zsm._mean).shape
        # (120, 1)
        # *(Pydb) (N.asmatrix(ds1.samples) - zsm._mean) * zsm._proj
        # matrix([[-0.13047326]])
        #
        # so we will handle case with n = 1 with regular non-sparse
        # matrices
        if n > 1:
            # format='csr' to avoid getting dia_matrix on scipy 0.12.0
            # which has non-functional dia_matrix.setdiag()
            proj = _identity(n, format='csr')
            proj.setdiag(1.0/std)
            recon = _identity(n, format='csr')
            recon.setdiag(std)
            self._proj = proj
            self._recon = recon
        else:
            self._proj = N.matrix([[1.0/std[0]]])
            self._recon = N.matrix([std])
