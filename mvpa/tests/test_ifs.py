# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA incremental feature search."""

from mvpa.testing import *
from mvpa.testing.clfs import *
from mvpa.testing.datasets import datasets

from mvpa.datasets.base import Dataset
from mvpa.featsel.ifs import IFS
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.clfs.transerror import TransferError
from mvpa.datasets.splitters import NFoldSplitter
from mvpa.featsel.helpers import FixedNElementTailSelector
from mvpa.mappers.fx import mean_sample



class IFSTests(unittest.TestCase):

    ##REF: Name was automagically refactored
    def get_data(self):
        data = np.random.standard_normal(( 100, 2, 2, 2 ))
        labels = np.concatenate( ( np.repeat( 0, 50 ),
                                  np.repeat( 1, 50 ) ) )
        chunks = np.repeat( range(5), 10 )
        chunks = np.concatenate( (chunks, chunks) )
        return Dataset.from_wizard(samples=data, targets=labels, chunks=chunks)


    # XXX just testing based on a single classifier. Not sure if
    # should test for every known classifier since we are simply
    # testing IFS algorithm - not sensitivities
    @sweepargs(svm=clfswh['has_sensitivity', '!meta'][:1])
    def test_ifs(self, svm):

        # data measure and transfer error quantifier use the SAME clf!
        trans_error = TransferError(svm)
        data_measure = CrossValidatedTransferError(trans_error,
                                                   NFoldSplitter(1),
                                                   postproc=mean_sample())

        ifs = IFS(data_measure,
                  trans_error,
                  feature_selector=\
                    # go for lower tail selection as data_measure will return
                    # errors -> low is good
                    FixedNElementTailSelector(1, tail='lower', mode='select'),
                  )
        wdata = self.get_data()
        wdata_nfeatures = wdata.nfeatures
        tdata = self.get_data()
        tdata_nfeatures = tdata.nfeatures

        sdata, stdata = ifs(wdata, tdata)

        # fail if orig datasets are changed
        self.failUnless(wdata.nfeatures == wdata_nfeatures)
        self.failUnless(tdata.nfeatures == tdata_nfeatures)

        # check that the features set with the least error is selected
        self.failUnless(len(ifs.ca.errors))
        e = np.array(ifs.ca.errors)
        self.failUnless(sdata.nfeatures == e.argmin() + 1)


        # repeat with dataset where selection order is known
        signal = datasets['dumb2']
        sdata, stdata = ifs(signal, signal)
        self.failUnless((sdata.samples[:,0] == signal.samples[:,0]).all())


def suite():
    return unittest.makeSuite(IFSTests)


if __name__ == '__main__':
    import runner

