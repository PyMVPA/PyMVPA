# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA perturbation sensitivity analyzer."""

from mvpa.datasets.masked import MaskedDataset
from mvpa.measures.noiseperturbation import NoisePerturbationSensitivity
from mvpa.datasets.splitters import NFoldSplitter
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.clfs.transerror import TransferError

from tests_warehouse import *
from tests_warehouse_clfs import *

class PerturbationSensitivityAnalyzerTests(unittest.TestCase):

    def setUp(self):
        data = N.random.standard_normal(( 100, 3, 4, 2 ))
        labels = N.concatenate( ( N.repeat( 0, 50 ),
                                  N.repeat( 1, 50 ) ) )
        chunks = N.repeat( range(5), 10 )
        chunks = N.concatenate( (chunks, chunks) )
        mask = N.ones( (3, 4, 2) )
        mask[0,0,0] = 0
        mask[1,3,1] = 0
        self.dataset = MaskedDataset(samples=data, labels=labels,
                                     chunks=chunks, mask=mask)


    def testPerturbationSensitivityAnalyzer(self):
        # compute N-1 cross-validation as datameasure
        cv = CrossValidatedTransferError(
                TransferError(sample_clf_lin),
                NFoldSplitter(cvtype=1))
        # do perturbation analysis using gaussian noise
        pa = NoisePerturbationSensitivity(cv, noise=N.random.normal)

        # run analysis
        map = pa(self.dataset)

        # check for correct size of map
        self.failUnless(len(map) == 22)

        # dataset is noise -> mean sensitivity should be zero
        self.failUnless(-0.2 < map.mean() < 0.2)


def suite():
    return unittest.makeSuite(PerturbationSensitivityAnalyzerTests)


if __name__ == '__main__':
    import runner

