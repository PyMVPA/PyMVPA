# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA ZScore mapper"""


import unittest

from mvpa.base import externals
externals.exists('scipy', raiseException=True)

from mvpa.support.copy import deepcopy
import numpy as N

from mvpa.datasets import Dataset
from mvpa.mappers.zscore import ZScoreMapper
from mvpa.datasets.miscfx import zscore

from tests_warehouse import datasets

class ZScoreMapperTests(unittest.TestCase):
    """Test simple ZScoreMapper
    """

    def setUp(self):
        """Setup sample datasets
        """
        # data: 40 sample feature line in 20d space (40x20; samples x features)
        self.dss = [
            Dataset(samples=N.concatenate(
                [N.arange(40) for i in range(20)]).reshape(20,-1).T,
                    labels=1, chunks=1),
            ] + datasets.values()


    def testCompareToZscore(self):
        """Test by comparing to results of elderly z-score function
        """
        for ds in self.dss:
            ds1 = deepcopy(ds)
            ds2 = deepcopy(ds)

            zsm = ZScoreMapper()
            zsm.train(ds1)
            ds1z = zsm.forward(ds1.samples)

            zscore(ds2, perchunk=False)
            self.failUnless(N.linalg.norm(ds1z - ds2.samples) < 1e-12)
            self.failUnless((ds1.samples == ds.samples).all(),
                            msg="It seems we modified original dataset!")

            ds0 = zsm.reverse(ds1z)
            self.failUnless(N.linalg.norm(ds0 - ds.samples) < 1e-12,
                            msg="Can't reconstruct from z-scores")


def suite():
    return unittest.makeSuite(ZScoreMapperTests)


if __name__ == '__main__':
    import runner

