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
from mvpa.datasets.base import dataset_wizard

class SOMMapperTests(unittest.TestCase):

    def test_simple_som(self):
        colors = N.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
                          [1., 0., 0.], [0., 1., 1.], [1., 0., 1.],
                          [1., 1., 0.], [1., 1., 1.]])

        # only small SOM for speed reasons
        som = SimpleSOMMapper((10, 5), 200, learning_rate=0.05)

        # no acces when nothing is there
        self.failUnlessRaises(RuntimeError, som._accessKohonen)

        som.train(colors)

        fmapped = som(colors)
        self.failUnless(fmapped.shape == (8, 2))

        # reverse mapping
        rmapped = som.reverse(fmapped)

        if cfg.getboolean('tests', 'labile', default='yes'):
            # should approximately restore the input, but could fail
            # with bad initialisation
            self.failUnless((N.round(rmapped) == colors).all())


def suite():
    return unittest.makeSuite(SOMMapperTests)


if __name__ == '__main__':
    import runner

