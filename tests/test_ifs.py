#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA incremental feature search."""

import unittest
import numpy as N

from mvpa.datasets.dataset import Dataset
from mvpa.datasets.maskeddataset import MaskedDataset
from mvpa.algorithms.ifs import IFS
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.clfs.svm import LinearNuSVMC
from mvpa.clfs.transerror import TransferError
from mvpa.datasets.splitter import NFoldSplitter
from mvpa.algorithms.featsel import FixedNElementTailSelector


def dumbFeatureDataset():
    data = [[0,1],[1,1],[0,2],[1,2],[0,3],[1,3],[0,4],[1,4],
            [0,5],[1,5],[0,6],[1,6],[0,7],[1,7],[0,8],[1,8],
            [0,9],[1,9],[0,10],[1,10],[0,11],[1,11],[0,12],[1,12]]
    regs = [1 for i in range(8)] \
         + [2 for i in range(8)] \
         + [3 for i in range(8)]

    return Dataset(samples=data, labels=regs)



class IFSTests(unittest.TestCase):

    def getData(self):
        data = N.random.standard_normal(( 100, 2, 2, 2 ))
        labels = N.concatenate( ( N.repeat( 0, 50 ),
                                  N.repeat( 1, 50 ) ) )
        chunks = N.repeat( range(5), 10 )
        chunks = N.concatenate( (chunks, chunks) )
        return MaskedDataset(samples=data, labels=labels, chunks=chunks)


    def testIFS(self):
        svm = LinearNuSVMC()

        # data measure and transfer error quantifier use the SAME clf!
        trans_error = TransferError(svm)
        data_measure = CrossValidatedTransferError(trans_error,
                                                   NFoldSplitter(1))

        ifs = IFS(data_measure,
                  trans_error,
                  feature_selector=\
                    # go for lower tail selection as data_measure will return
                    # errors -> low is good
                    FixedNElementTailSelector(1, tail='lower', mode='select'),
                  )
        wdata = self.getData()
        wdata_nfeatures = wdata.nfeatures
        tdata = self.getData()
        tdata_nfeatures = tdata.nfeatures

        sdata, stdata = ifs(wdata, tdata)

        # fail if orig datasets are changed
        self.failUnless(wdata.nfeatures == wdata_nfeatures)
        self.failUnless(tdata.nfeatures == tdata_nfeatures)

        # check that the features set with the least error is selected
        self.failUnless(len(ifs.errors))
        e = N.array(ifs.errors)
        self.failUnless(sdata.nfeatures == e.argmin() + 1)


        # repeat with dataset where selection order is known
        signal = dumbFeatureDataset()
        sdata, stdata = ifs(signal, signal)
        self.failUnless((sdata.samples[:,0] == signal.samples[:,1]).all())


def suite():
    return unittest.makeSuite(IFSTests)


if __name__ == '__main__':
    import test_runner

