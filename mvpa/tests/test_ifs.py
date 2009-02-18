# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA incremental feature search."""

from mvpa.datasets.masked import MaskedDataset
from mvpa.featsel.ifs import IFS
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.clfs.transerror import TransferError
from mvpa.datasets.splitters import NFoldSplitter
from mvpa.featsel.helpers import FixedNElementTailSelector

from tests_warehouse import *
from tests_warehouse_clfs import *


class IFSTests(unittest.TestCase):

    def getData(self):
        data = N.random.standard_normal(( 100, 2, 2, 2 ))
        labels = N.concatenate( ( N.repeat( 0, 50 ),
                                  N.repeat( 1, 50 ) ) )
        chunks = N.repeat( range(5), 10 )
        chunks = N.concatenate( (chunks, chunks) )
        return MaskedDataset(samples=data, labels=labels, chunks=chunks)


    # XXX just testing based on a single classifier. Not sure if
    # should test for every known classifier since we are simply
    # testing IFS algorithm - not sensitivities
    @sweepargs(svm=clfswh['has_sensitivity', '!meta'][:1])
    def testIFS(self, svm):

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
        signal = datasets['dumb2']
        sdata, stdata = ifs(signal, signal)
        self.failUnless((sdata.samples[:,0] == signal.samples[:,0]).all())


def suite():
    return unittest.makeSuite(IFSTests)


if __name__ == '__main__':
    import runner

