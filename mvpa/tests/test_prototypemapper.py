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
from mvpa.clfs.kernel import KernelExponential, KernelSquaredExponential

from numpy.testing import assert_array_equal, assert_array_almost_equal

from mvpa.datasets.base import dataset


class PrototypeMapperTests(unittest.TestCase):

    def setUp(self):
        # samples: 40 sample feature line in 20d space (40x20; samples x features)
        self.samples = N.random.rand(40,20)

        # initial prototypes are samples itself:
        self.prototypes = self.samples.copy()

        # using just two similarities for now:
        self.similarities = [KernelExponential(), KernelSquaredExponential()]

        # set up prototype mapper with prototypes identical to samples.
        self.pm = PrototypeMapper(similarities=self.similarities, prototypes=self.prototypes)
        # train Prototype
        self.pm.train(self.samples)

        # set up prototype mapper without specifying prototypes
        self.pm2 = PrototypeMapper(similarities=self.similarities)
        self.pm2.train(self.samples)
        

    def testSize(self):
        assert_array_equal(self.pm.proj.shape, (self.samples.shape[0], self.prototypes.shape[0]*len(self.similarities)))


    def testSimmetry(self):
        assert_array_almost_equal(self.pm.proj[:,self.samples.shape[0]], self.pm.proj.T[self.samples.shape[0],:])
        assert_array_equal(self.pm.proj[:,self.samples.shape[0]], self.pm.proj.T[self.samples.shape[0],:])


    def testNoExplicitPrototypes(self):
        assert_array_equal(self.pm2.proj.shape, (self.samples.shape[0], self.pm2.prototypes.shape[0]*len(self.similarities)))


def suite():
    return unittest.makeSuite(PrototypeMapperTests)


if __name__ == '__main__':
    import runner

