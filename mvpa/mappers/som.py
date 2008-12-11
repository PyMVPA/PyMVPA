#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Self-organizing map (SOM) mapper."""

__docformat__ = 'restructuredtext'


import numpy as N
from mvpa.mappers.base import Mapper


class SimpleSOMMapper(Mapper):
    """Mapper using a self-organizing map (SOM) for dimensionality reduction.

    This mapper provides a simple, but pretty fast implementation of a
    self-organizing map using an unsupervised training algorithm. It performs a
    ND -> 2D mapping, which can for, example, be used for visualization of
    high-dimensional data.

    This SOM implementation uses squared euclidean distance to determine
    the best matching Kohonen unit and a gaussian neighborhood influence
    kernel.
    """
    def __init__(self, nrows, ncolumns, niter, learning_rate=0.005,
                 iradius=None):
        """
        :Parameters:
          nrows: int
            Number of rows in the Kohonen layer.
          ncolumns: int
            Number of columns in the Kohonen layer.
          niter: int
            Number of iteration during network training.
          learning_rate: float
            Initial learning rate, which will continously decreased during
            network training.
          iradius: float | None
            Initial radius of the gaussian neighborhood kernel radius, which
            will continously decreased during network training. If `None`
            (default) the radius is set equal to the longest edge of the
            Kohonen layer.
        """
        # init base class
        Mapper.__init__(self)

        self.nrows = int(nrows)
        self.ncolumns = int(ncolumns)

        if iradius is None:
            self.radius = max(self.nrows, self.ncolumns)
        else:
            self.radius = iradius

        # learning rate
        self.lrate = learning_rate

        # number of training iterations
        self.niter = niter

        # precompute whatever can be done
        # scalar for decay of learning rate and radius across all iterations
        self.iter_scale = self.niter / N.log(self.radius)




    def train(self, ds):
        """Perform network training.

        :Parameter:
          ds: Dataset
            All samples in the dataset will be used for unsupervised training of
            the SOM.
        """
        # XXX initialize with clever default, e.g. plain of first two PCA
        # components
        self.units = N.random.randn(self.nrows, self.ncolumns, ds.nfeatures)

        # units weight vector deltas for batch training
        # (height x width x #features)
        unit_deltas = N.zeros(self.units.shape, dtype='float')

        # precompute distance kernel between elements in the kohonen layer
        # that will remain constant troughout the training
        # (just compute one quadrant, as the distances are symetric)
        # XXX maybe do other than squared euclidean?
        dqd = N.fromfunction(lambda x, y: (x**2 + y**2)**0.5,
                             (self.nrows, self.ncolumns),
                             dtype='float')

        # for all iterations
        for it in xrange(1, self.niter + 1):
            # compute the neighborhood impact kernel for this iteration
            # has to be recomputed since kernel shrinks over time
            k = self._computeInfluenceKernel(it, dqd)

            # for all training vectors
            for s in ds.samples:
                # determine closest unit (as element coordinate)
                b = self._getBMU(s)

                # train all units at once by unfolding the kernel (from the
                # single quadrant that is precomputed), cutting it to the
                # right shape and simply multiply it to the difference of target
                # and all unit weights....
                infl = N.vstack((
                        N.hstack((
                            # upper left
                            k[b[0]:0:-1, b[1]:0:-1],
                            # upper right
                            k[b[0]:0:-1, :self.ncolumns - b[1]])),
                        N.hstack((
                            # lower left
                            k[:self.nrows - b[0], b[1]:0:-1],
                            # lower right
                            k[:self.nrows - b[0], :self.ncolumns - b[1]]))
                               ))
                unit_deltas += infl[:,:,N.newaxis] * (s - self.units)

            # apply cumulative unit deltas
            self.units += unit_deltas

            # reset unit deltas
            unit_deltas.fill(0.)


    def _computeInfluenceKernel(self, iter, dqd):
        """Compute the neighborhood kernel for some iteration.

        :Parameters:
          iter: int
            The iteration for which to compute the kernel.
          dqd: array (nrows x ncolumns)
            This is one quadrant of euclidean distances between Kohonen unit
            locations.
        """
        # compute radius decay for this iteration
        curr_max_radius = self.radius * N.exp(-1.0 * iter / self.iter_scale)

        # same for learning rate
        curr_lrate = self.lrate * N.exp(-1.0 * iter / self.iter_scale)

        # compute gaussian influence kernel
        infl = N.exp((-1.0 * dqd) / (2 * curr_max_radius * iter))
        infl *= curr_lrate

        # hard-limit kernel to max radius
        # XXX is this really necessary?
        infl[dqd > curr_max_radius] = 0.

        return infl


    def _getBMU(self, sample):
        """Returns the ID of the best matching unit.

        'best' is determined as minimal squared euclidean distance between
        any units weight vector and some given target `sample`

        :Parameters:
          sample: array
            Target sample.

        :Returns:
          tuple: (row, column)
        """
        # TODO expose distance function as parameter
        loc = N.argmin(((self.units - sample) ** 2).sum(axis=2))

        return (N.divide(loc, self.ncolumns), loc % self.ncolumns)
