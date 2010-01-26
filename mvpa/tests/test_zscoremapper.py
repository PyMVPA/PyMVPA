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
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mvpa.datasets.base import dataset_wizard
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
            dataset_wizard(N.concatenate(
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
            assert_array_almost_equal(ds1z, ds2.samples)
            assert_array_equal(ds1.samples, ds.samples)

            ds0 = zsm.reverse(ds1z)
            assert_array_almost_equal(ds0, ds.samples)


def suite():
    return unittest.makeSuite(ZScoreMapperTests)


if __name__ == '__main__':
    import runner

