# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SampleGroup mapper"""


import unittest
from mvpa.support.copy import deepcopy
import numpy as N

from mvpa.mappers.samplegroup import SampleGroupMapper
from mvpa.datasets import Dataset


class SampleGroupMapperTests(unittest.TestCase):

    def testSimple(self):
        data = N.arange(24).reshape(8,3)
        labels = [0, 1] * 4
        chunks = N.repeat(N.array((0,1)),4)

        # correct results
        csamples = [[3, 4, 5], [6, 7, 8], [15, 16, 17], [18, 19, 20]]
        clabels = [0, 1, 0, 1]
        cchunks = [0, 0, 1, 1]

        ds = Dataset(samples=data, labels=labels, chunks=chunks)

        # default behavior
        m = SampleGroupMapper()

        # error if not trained
        self.failUnlessRaises(RuntimeError, m, data)

        # train mapper first
        m.train(ds)

        self.failUnless((m.forward(ds.samples) == csamples).all())
        self.failUnless((m.forward(ds.labels) == clabels).all())
        self.failUnless((m.forward(ds.chunks) == cchunks).all())


        # directly apply to dataset
        # using untrained mapper!
        mapped = ds.applyMapper(samplesmapper=SampleGroupMapper())

        self.failUnless(mapped.nsamples == 4)
        self.failUnless(mapped.nfeatures == 3)
        self.failUnless((mapped.samples == csamples).all())
        self.failUnless((mapped.labels == clabels).all())
        self.failUnless((mapped.chunks == cchunks).all())
        # make sure origids get regenerated
        self.failUnless((mapped.origids == range(4)).all())


def suite():
    return unittest.makeSuite(SampleGroupMapperTests)


if __name__ == '__main__':
    import runner

