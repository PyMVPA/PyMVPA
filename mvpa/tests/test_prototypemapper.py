# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA prototype mapper."""


import unittest
import numpy as N
from mvpa.mappers.prototype import PrototypeMapper
from mvpa.kernels.np import ExponentialKernel, SquaredExponentialKernel

from mvpa.datasets import Dataset
from mvpa.clfs.similarity import StreamlineSimilarity
from mvpa.clfs.distance import corouge

from mvpa.testing.tools import assert_array_equal, assert_array_almost_equal

import random

if __debug__:
    from mvpa.base import debug

class PrototypeMapperTests(unittest.TestCase):

    def setUp(self):
        pass

    def buildVectorBasedPM(self):
        # samples: 40 samples in 20d space (40x20; samples x features)
        self.samples = N.random.rand(40,20)

        # initial prototypes are samples itself:
        self.prototypes = self.samples.copy()

        # using just two similarities for now:
        self.similarities = [ExponentialKernel(), SquaredExponentialKernel()]
        # set up prototype mapper with prototypes identical to samples.
        self.pm = PrototypeMapper(similarities=self.similarities,
                                  prototypes=self.prototypes)
        # train Prototype
        self.pm.train(self.samples)


    def test_size(self):
        self.buildVectorBasedPM()
        assert_array_equal(self.pm.proj.shape,
                           (self.samples.shape[0],
                            self.prototypes.shape[0] * len(self.similarities)))


    def test_symmetry(self):
        self.buildVectorBasedPM()
        assert_array_almost_equal(self.pm.proj[:,self.samples.shape[0]],
                                  self.pm.proj.T[self.samples.shape[0],:])
        assert_array_equal(self.pm.proj[:,self.samples.shape[0]],
                           self.pm.proj.T[self.samples.shape[0],:])


    def test_size_random_prototypes(self):
        self.buildVectorBasedPM()
        fraction = 0.5
        prototype_number = max(int(len(self.samples)*fraction),1)
        ## debug("MAP","Generating "+str(prototype_number)+" random prototypes.")
        self.prototypes2 = N.array(random.sample(self.samples, prototype_number))
        self.pm2 = PrototypeMapper(similarities=self.similarities, prototypes=self.prototypes2)
        self.pm2.train(self.samples)
        assert_array_equal(self.pm2.proj.shape, (self.samples.shape[0], self.pm2.prototypes.shape[0]*len(self.similarities)))

    def buildStreamlineThings(self):
        # Build a dataset having samples of different lengths. This is
        # trying to mimic a possible interface for streamlines
        # datasets, i.e., an iterable container of Mx3 points, where M
        # depends on each single streamline.

        # trying to pack it into an 'object' array to prevent conversion in the
        # Dataset
        self.streamline_samples = N.array([
                                   N.random.rand(3,3),
                                   N.random.rand(5,3),
                                   N.random.rand(7,3)],
                                   dtype='object')
        self.dataset = Dataset(self.streamline_samples)
        self.similarities = [StreamlineSimilarity(distance=corouge)]


    def test_streamline_equal_mapper(self):
        self.buildStreamlineThings()

        self.prototypes_equal = self.dataset.samples
        self.pm = PrototypeMapper(similarities=self.similarities,
                                  prototypes=self.prototypes_equal,
                                  demean=False)
        self.pm.train(self.dataset.samples)
        ## debug("MAP","projected data: "+str(self.pm.proj))
        # check size:
        assert_array_equal(self.pm.proj.shape, (len(self.dataset.samples), len(self.prototypes_equal)*len(self.similarities)))
        # test symmetry
        assert_array_almost_equal(self.pm.proj, self.pm.proj.T)


    def test_streamline_random_mapper(self):
        self.buildStreamlineThings()

        # Adding one more similarity to test multiple similarities in the streamline case:
        self.similarities.append(StreamlineSimilarity(distance=corouge))

        fraction = 0.5
        prototype_number = max(int(len(self.dataset.samples)*fraction),1)
        ## debug("MAP","Generating "+str(prototype_number)+" random prototypes.")
        self.prototypes_random = random.sample(self.dataset.samples, prototype_number)
        ## debug("MAP","prototypes: "+str(self.prototypes_random))

        self.pm = PrototypeMapper(similarities=self.similarities, prototypes=self.prototypes_random, demean=False)
        self.pm.train(self.dataset.samples) # , fraction=1.0)
        # test size:
        assert_array_equal(self.pm.proj.shape, (len(self.dataset.samples), len(self.prototypes_random)*len(self.similarities)))


def suite():
    return unittest.makeSuite(PrototypeMapperTests)


if __name__ == '__main__':
    import runner

