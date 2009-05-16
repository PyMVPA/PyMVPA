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

import numpy as N

from mvpa.base.dochelpers import enhancedDocString
from mvpa.mappers.base import Mapper
from mvpa.datasets import Dataset
from mvpa.featsel.helpers import ElementSelector

if __debug__:
    from mvpa.base import debug


class ProcrusteanMapper(Mapper):
    """Mapper to project from one space to another using Procrustean
    transformation (shift + scaling + rotation)
    """

    _DEV__doc__ = """Possibly revert back to inherit from ProjectionMapper"""

    def __init__(self, scaling=True, reflection=True, reduction=True, **kwargs):
        """Initialize the ProcrusteanMapper

        :Parameters:
          scaling: bool
            Scale data for the transformation (no longer rigid body
            transformation)
          reflection: bool
            Allow for the data to be reflected (so it might not be a rotation)
          reduction: bool
            If true, it is allowed to map into lower-dimensional
            space. Forward transformation might be suboptimal then and reverse
            transformation might not recover all original variance
        """
        Mapper.__init__(self, **kwargs)

        self._scaling = scaling
        """Either to determine the scaling factor"""

        self._reduction = reduction
        self._reflection = reflection
        self._T = None
        """Rotation matrix"""

    __doc__ = enhancedDocString('ProcrusteanMapper', locals(), Mapper)

    # XXX we should just use beautiful ClassWithCollections everywhere... makes
    # life so easier... for now -- manual
    def __repr__(self):
        s = Mapper.__repr__(self).rstrip(' )')
        if not s[-1] == '(': s += ', '
        s += "scaling=%d, reflection=%d, reduction=%d)" % \
             (self._scaling, self._reflection, self._reduction)
        return s

    # XXX we have to override train since now we have multiple datasets
    #     alternative way is to assign target to the labels of the source
    #     dataset
    def train(self, source, target=None):
        """Train Procrustean transformation

        :Parameters:
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
            target = source.labels

        for i,ds in enumerate((source, target)):
            if isinstance(ds, Dataset): data = N.asarray(ds.samples)
            else: data = ds
            if assess_residuals:
                odatas += (data,)
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

        ## if tm < sm:

        # Sums of squares
        ssqs = [N.sum(d**2, axis=0) for d in datas]

        # XXX check for being invariant?
        #     needs to be tuned up properly and not raise but handle
        for i in xrange(2):
            if N.all(ssqs[i] <= N.abs((N.finfo(datas[i].dtype).eps
                                       * sn * means[i] )**2)):
                raise ValueError, "For now do not handle invariant in time datasets"

        norms = [ N.sqrt(N.sum(ssq)) for ssq in ssqs ]
        normed = [ data/norm for (data, norm) in zip(datas, norms) ]

        # add new blank dimensions to source space if needed
        if sm < tm:
            normed[0] = N.hstack( (normed[0], N.zeros((sn, tm-sm))) )

        if sm > tm:
            if self._reduction:
                normed[1] = N.hstack( (normed[1], N.zeros((sn, sm-tm))) )
            else:
                raise ValueError, "reduction=False, so mapping from " \
                      "higher dimensionality " \
                      "source space is not supported. Source space had %d " \
                      "while target %d dimensions (features)" % (sm, tm)


        # figure out optimal rotation
        U, s, Vh = N.linalg.svd(N.dot(normed[1].T, normed[0]),
                                full_matrices=False)
        T = N.dot(Vh.T, U.T)

        if not self._reflection:
            # then we need to assure that it is only rotation
            # "recipe" from
            # http://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
            # for more and info and original references, see
            # http://dx.doi.org/10.1007%2FBF02289451
            nsv = len(s)
            s[:-1] = 1
            s[-1] = N.linalg.det(T)
            T = N.dot(U[:, :nsv] * s, Vh)

        # figure out scale and final translation
        # XXX with reflection False -- not sure if here or there or anywhere...
        ss = sum(s)

        # if we were to collect standardized distance
        # std_d = 1 - sD**2

        # select out only relevant dimensions
        if sm != tm:
            T = T[:sm, :tm]

        self._T = T
        self._scale = scale = ss * norms[1] / norms[0]
        mT = N.dot(means[0], T)
        self._trans =  means[1] - scale * mT
        self._trans_unscaled =  means[1] - mT

        if __debug__ and 'MAP_' in debug.active:
            # compute the residuals
            res_f = self.forward(odatas[0])
            d_f = N.linalg.norm(odatas[1] - res_f)/N.linalg.norm(odatas[1])
            res_r = self.reverse(odatas[1])
            d_r = N.linalg.norm(odatas[0] - res_r)/N.linalg.norm(odatas[0])
            debug('MAP_', "%s, residuals are forward: %g,"
                  " reverse: %g" % (`self`, d_f, d_r))
        # Combine rotation + scale into _T
        #self._T = self._scale * R

    ## def __getT(self):
    ##     """A little helper function to return proper translation
    ##     """
    ##     return (self._trans_unscaled, self._trans)[int(self._scaling)]]


    def forward(self, data):
        """Project data using precomputed Procrustean

        :Parameters:
           data: ndarray
             Data array to map
        """
        if self._T is None:
            raise RuntimeError, "Mapper needs to be trained before used."
        if self._scaling:
            return self._scale * N.dot(data, self._T) + self._trans
        else:
            return N.dot(data, self._T) + self._trans_unscaled


    def reverse(self, data):
        """Project data back using precomputed Procrustean
        """
        if self._scaling:
            return N.dot((data - self._trans)/self._scale, self._T.T)
        else:
            return N.dot((data - self._trans_unscaled), self._T.T)


    def getInSize(self):
        """Returns the number of original features."""
        return self._T.shape[0]


    def getOutSize(self):
        """Returns the number of components to project on."""
        return self._T.shape[1]

