# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SOM mapper"""


import unittest
import numpy as np
from mvpa2 import cfg
from mvpa2.mappers.som import SimpleSOMMapper
from mvpa2.datasets.base import dataset_wizard

class SOMMapperTests(unittest.TestCase):
    def test_periodic_boundaries(self):
    
        som = SimpleSOMMapper((10, 5), 200, learning_rate=0.05) 
        
        test_dqdshape = np.array([5,2,5,3])

        # som._dqdshape only defined in newer version
        # this is not explicitly linked to the periodic boundary conditions,
        # but had trouble coming up with a simple test for them
        self.assertTrue((som._dqdshape == test_dqdshape).all())
    
    def test_kohonen_update(self):
        # before update error occured when learning_rate*number of samples > 1
        # here use extreme learning_rate to force bad behaviour  
        som = SimpleSOMMapper((10, 5), 200, learning_rate=1.0)
        
        trainer = np.ones([8,3])
        
        som.train(trainer)
        
        # use 10 instead of 4 to allow for some randomness in the training
        # fail values tend to be closer to 10^30
        self.assertTrue((np.abs(som.K) <= 10).all())

    def test_simple_som(self):
        colors = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
                          [1., 0., 0.], [0., 1., 1.], [1., 0., 1.],
                          [1., 1., 0.], [1., 1., 1.]])

        distance_measures = (None, lambda x, y:(x ** 3 + y ** 3) ** (1. / 3))

        for distance_measure in distance_measures:
            # only small SOM for speed reasons
            som = SimpleSOMMapper((10, 5), 200, learning_rate=0.05)

            # no acces when nothing is there
            self.assertRaises(RuntimeError, som._access_kohonen)

            som.train(colors)

            fmapped = som.forward(colors)
            self.assertTrue(fmapped.shape == (8, 2))

            # reverse mapping
            rmapped = som.reverse(fmapped)

            if cfg.getboolean('tests', 'labile', default='yes'):
                # should approximately restore the input, but could fail
                # with bad initialisation
                self.assertTrue((np.round(rmapped) == colors).all())


def suite():  # pragma: no cover
    return unittest.makeSuite(SOMMapperTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

