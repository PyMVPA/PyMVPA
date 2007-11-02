#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Unit tests for PyMVPA nifti dataset"""

import unittest
import os.path
import numpy as N

from mvpa.datasets.niftidataset import *

class NiftiDatasetTests(unittest.TestCase):

    def testNiftiDataset(self):
        data = NiftiDataset(os.path.join('data','example4d'), [1,2], None)
        self.failUnless(data.nfeatures == 294912)
        self.failUnless(data.nsamples == 2)
        self.failUnless(data.mapper.dsshape == (24,96,128))

        self.failUnless((data.mapper.metric.elementsize \
                         == data.niftihdr['pixdim'][3:0:-1]).all())

        #check that mapper honours elementsize
        nb22=N.array([i for i in data.mapper.getNeighborIn((1,1,1), 2.2)])
        nb20=N.array([i for i in data.mapper.getNeighborIn((1,1,1), 2.0)])
        self.failUnless(nb22.shape[0] == 7)
        self.failUnless(nb20.shape[0] == 5)

        merged = data + data

        self.failUnless(merged.nfeatures == 294912)
        self.failUnless(merged.nsamples == 4)
        self.failUnless(merged.mapper.dsshape == (24,96,128))

        # check that the header survives
        #self.failUnless(merged.niftihdr == data.niftihdr)
        for k in merged.niftihdr.keys():
            self.failUnless(N.mean(merged.niftihdr[k] == data.niftihdr[k]) == 1)

        # throw away old dataset and see if new one survives
        del data
        self.failUnless(merged.samples[3, 120000] == merged.samples[1, 120000])


def suite():
    return unittest.makeSuite(NiftiDatasetTests)


if __name__ == '__main__':
    import test_runner

