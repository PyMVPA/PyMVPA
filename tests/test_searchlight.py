#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA searchlight algorithm"""

import unittest

import numpy as N

from mvpa.datasets.maskeddataset import MaskedDataset
from mvpa.algorithms.searchlight import Searchlight
from mvpa.clfs.knn import kNN
from mvpa.datasets.splitter import NFoldSplitter
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.clfs.transerror import TransferError


class SearchlightTests(unittest.TestCase):

    def setUp(self):
        data = N.random.standard_normal(( 100, 3, 6, 6 ))
        labels = N.concatenate( ( N.repeat( 0, 50 ),
                                  N.repeat( 1, 50 ) ) )
        chunks = N.repeat( range(5), 10 )
        chunks = N.concatenate( (chunks, chunks) )
        mask = N.ones( (3, 6, 6) )
        mask[0,0,0] = 0
        mask[1,3,2] = 0
        self.dataset = MaskedDataset(samples=data, labels=labels,
                                     chunks=chunks, mask=mask)


    def testSearchlight(self):
        # compute N-1 cross-validation for each sphere
        transerror = TransferError(kNN(k=5))
        cv = CrossValidatedTransferError(
                transerror,
                NFoldSplitter(cvtype=1))
        # contruct radius 1 searchlight
        sl = Searchlight( cv, radius=1.0 )

        # run searchlight
        results = sl(self.dataset)

        # check for correct number of spheres
        self.failUnless(len(results) == 106)

        # check for chance-level performance across all spheres
        self.failUnless(0.4 < results.mean() < 0.6)

        # check resonable sphere sizes
        self.failUnless(len(sl.spheresizes) == 106)
        self.failUnless(max(sl.spheresizes) == 7)
        self.failUnless(min(sl.spheresizes) == 4)



def suite():
    return unittest.makeSuite(SearchlightTests)


if __name__ == '__main__':
    import test_runner

