# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Procrustean rotation mapper"""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.base import externals
from mvpa2.mappers.projection import ProjectionMapper
from mvpa2.datasets import Dataset
from mvpa2.featsel.helpers import ElementSelector

from mvpa2.base import warning
if __debug__:
    from mvpa2.base import debug



class ProcrusteanMapper(ProjectionMapper):
    """Mapper to project from one space to another using Procrustean
    transformation (shift + scaling + rotation)
    """

    _DEV__doc__ = """Possibly revert back to inherit from ProjectionMapper"""

    def __init__(self, scaling=True, reflection=True, reduction=True,
                 oblique=False, oblique_rcond=-1, svd='numpy', **kwargs):
        """Initialize the ProcrusteanMapper

        Parameters
        ----------
        scaling : bool
          Scale data for the transformation (no longer rigid body
          transformation)
        reflection : bool
          Allow for the data to be reflected (so it might not be a rotation).
          Effective only for non-oblique transformations
        reduction : bool
          If true, it is allowed to map into lower-dimensional
          space. Forward transformation might be suboptimal then and reverse
          transformation might not recover all original variance
        oblique : bool
          Either to allow non-orthogonal transformation -- might heavily overfit
          the data if there is less samples than dimensions. Use `oblique_rcond`.
        oblique_rcond : float
          Cutoff for 'small' singular values to regularize the inverse. See
          :class:`~numpy.linalg.lstsq` for more information.
        svd : string (numpy, scipy, dgesvd), optional
          Implementation of SVD to use.  dgesvd requires ctypes to be available.
        **kwargs
          To be passed to ProjectionMapper
        """
        ProjectionMapper.__init__(self, **kwargs)

        self._scaling = scaling
        """Either to determine the scaling factor"""

        self._reduction = reduction
        self._reflection = reflection
        self._oblique = oblique
        self._oblique_rcond = oblique_rcond
        self._scale = None
        """Estimated scale"""
        if svd == 'dgesvd' and not externals.exists('liblapack.so'):
            warning("Reverting choice of svd for ProcrusteanMapper to be default "
                    "'numpy' since liblapack.so seems not to be available for "
                    "'dgesvd'")
            svd = 'numpy'
        self._svd = svd

    # XXX we should just use beautiful ClassWithCollections everywhere... makes
    # life so easier... for now -- manual
    def __repr__(self):
        s = ProjectionMapper.__repr__(self).rstrip(' )')
        if not s[-1] == '(': s += ', '
        s += "scaling=%r, reflection=%r, reduction=%r, " \
             "oblique=%r, oblique_rcond=%r)" % \
             (self._scaling, self._reflection, self._reduction,
              self._oblique, self._oblique_rcond)
        return s

    # XXX we have to override train since now we have multiple datasets
    #     alternative way is to assign target to the labels of the source
    #     dataset
    def _train(self, source, target=None):
        """Train Procrustean transformation

        Parameters
        ----------
        source : dataset or ndarray
          Source space for determining the transformation. If target
          is None, then labels of 'source' dataset are taken as the target
        target : dataset or ndarray or Null
          Target space for determining the transformation
        """

        # Since it is unsupervised, we don't care about labels
        datas = ()
        odatas = ()
        means = ()
        shapes = ()

        assess_residuals = __debug__ and 'MAP_' in debug.active

        if target is None:
            target = source.targets

        for i, ds in enumerate((source, target)):
            if isinstance(ds, Dataset):
                data = np.asarray(ds.samples)
            else:
                data = ds
            if assess_residuals:
                odatas += (data,)
            if i == 0:
                mean = self._offset_in
            else:
                mean = data.mean(axis=0)
            data = data - mean
            means += (mean,)
            datas += (data,)
            shapes += (data.shape,)

        # shortcuts for sizes
        sn, sm = shapes[0]
        tn, tm = shapes[1]

        # Check the sizes
        if sn != tn:
            raise ValueError, "Data for both spaces should have the same " \
                  "number of samples. Got %d in source and %d in target space" \
                  % (sn, tn)

        # Sums of squares
        ssqs = [np.sum(d**2, axis=0) for d in datas]

        # XXX check for being invariant?
        #     needs to be tuned up properly and not raise but handle
        for i in xrange(2):
            if np.all(ssqs[i] <= np.abs((np.finfo(datas[i].dtype).eps
                                       * sn * means[i] )**2)):
                raise ValueError, "For now do not handle invariant in time datasets"

        norms = [ np.sqrt(np.sum(ssq)) for ssq in ssqs ]
        normed = [ data/norm for (data, norm) in zip(datas, norms) ]

        # add new blank dimensions to source space if needed
        if sm < tm:
            normed[0] = np.hstack( (normed[0], np.zeros((sn, tm-sm))) )

        if sm > tm:
            if self._reduction:
                normed[1] = np.hstack( (normed[1], np.zeros((sn, sm-tm))) )
            else:
                raise ValueError, "reduction=False, so mapping from " \
                      "higher dimensionality " \
                      "source space is not supported. Source space had %d " \
                      "while target %d dimensions (features)" % (sm, tm)

        source, target = normed
        if self._oblique:
            # Just do silly linear system of equations ;) or naive
            # inverse problem
            if sn == sm and tm == 1:
                T = np.linalg.solve(source, target)
            else:
                T = np.linalg.lstsq(source, target, rcond=self._oblique_rcond)[0]
            ss = 1.0
        else:
            # Orthogonal transformation
            # figure out optimal rotation
            if self._svd == 'numpy':
                U, s, Vh = np.linalg.svd(np.dot(target.T, source),
                               full_matrices=False)
            elif self._svd == 'scipy':
                # would raise exception if not present
                externals.exists('scipy', raise_=True)
                import scipy
                U, s, Vh = scipy.linalg.svd(np.dot(target.T, source),
                               full_matrices=False)
            elif self._svd == 'dgesvd':
                from mvpa2.support.lapack_svd import svd as dgesvd
                U, s, Vh = dgesvd(np.dot(target.T, source),
                                    full_matrices=True, algo='svd')
            else:
                raise ValueError('Unknown type of svd %r'%(self._svd))
            T = np.dot(Vh.T, U.T)

            if not self._reflection:
                # then we need to assure that it is only rotation
                # "recipe" from
                # http://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
                # for more and info and original references, see
                # http://dx.doi.org/10.1007%2FBF02289451
                nsv = len(s)
                s[:-1] = 1
                s[-1] = np.linalg.det(T)
                T = np.dot(U[:, :nsv] * s, Vh)

            # figure out scale and final translation
            # XXX with reflection False -- not sure if here or there or anywhere...
            ss = sum(s)

        # if we were to collect standardized distance
        # std_d = 1 - sD**2

        # select out only relevant dimensions
        if sm != tm:
            T = T[:sm, :tm]

        self._scale = scale = ss * norms[1] / norms[0]
        # Assign projection
        if self._scaling:
            proj = scale * T
        else:
            proj = T
        self._proj = proj

        if self._demean:
            self._offset_out = means[1]

        if __debug__ and 'MAP_' in debug.active:
            # compute the residuals
            res_f = self.forward(odatas[0])
            d_f = np.linalg.norm(odatas[1] - res_f)/np.linalg.norm(odatas[1])
            res_r = self.reverse(odatas[1])
            d_r = np.linalg.norm(odatas[0] - res_r)/np.linalg.norm(odatas[0])
            debug('MAP_', "%s, residuals are forward: %g,"
                  " reverse: %g" % (repr(self), d_f, d_r))
