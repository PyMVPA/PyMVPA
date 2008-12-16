#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SOM mapper"""


import unittest
import numpy as N
from mvpa import cfg
from mvpa.mappers.som import SimpleSOMMapper
from mvpa.datasets import Dataset

class SOMMapperTests(unittest.TestCase):

    def testSimpleSOM(self):
        colors = [[0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
                  [1., 0., 0.], [0., 1., 1.], [1., 0., 1.],
                  [1., 1., 0.], [1., 1., 1.]]
        ds = Dataset(samples=colors, labels=1)

        # only small SOM for speed reasons
        som = SimpleSOMMapper((10, 5), 200, learning_rate=0.05)

        # no acces when nothing is there
        self.failUnlessRaises(RuntimeError, som._accessKohonen)
        self.failUnlessRaises(RuntimeError, som.getInSize)
        self.failUnlessRaises(RuntimeError, som.getOutSize)

        som.train(ds)

        self.failUnless(som.getInSize() == 3)
        self.failUnless(som.getOutSize() == (10,5))

        fmapped = som(colors)
        self.failUnless(fmapped.shape == (8, 2))
        for fm in fmapped:
            self.failUnless(som.isValidOutId(fm))

        # reverse mapping
        rmapped = som.reverse(fmapped)

        if cfg.getboolean('tests', 'labile', default='yes'):
            # should approximately restore the input, but could fail
            # with bas initialisation
            self.failUnless((N.round(rmapped) == ds.samples).all())


def suite():
    return unittest.makeSuite(SOMMapperTests)


if __name__ == '__main__':
    import runner

