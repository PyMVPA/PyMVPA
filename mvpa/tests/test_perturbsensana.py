# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA perturbation sensitivity analyzer."""

import numpy as np
from mvpa.testing import *
from mvpa.testing.clfs import *

from mvpa.datasets.base import Dataset
from mvpa.measures.noiseperturbation import NoisePerturbationSensitivity
from mvpa.datasets.splitters import NFoldSplitter
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.clfs.transerror import TransferError


class PerturbationSensitivityAnalyzerTests(unittest.TestCase):

    def setUp(self):
        data = np.random.standard_normal(( 100, 3, 4, 2 ))
        labels = np.concatenate( ( np.repeat( 0, 50 ),
                                  np.repeat( 1, 50 ) ) )
        chunks = np.repeat( range(5), 10 )
        chunks = np.concatenate( (chunks, chunks) )
        mask = np.ones( (3, 4, 2), dtype='bool')
        mask[0,0,0] = 0
        mask[1,3,1] = 0
        self.dataset = Dataset.from_wizard(samples=data, targets=labels,
                                           chunks=chunks, mask=mask)


    def test_perturbation_sensitivity_analyzer(self):
        # compute N-1 cross-validation as datameasure
        cv = CrossValidatedTransferError(
                TransferError(sample_clf_lin),
                NFoldSplitter(cvtype=1))
        # do perturbation analysis using gaussian noise
        pa = NoisePerturbationSensitivity(cv, noise=np.random.normal)

        # run analysis
        map = pa(self.dataset)

        # check for correct size of map
        self.failUnless(map.nfeatures == self.dataset.nfeatures)

        # dataset is noise -> mean sensitivity should be zero
        self.failUnless(-0.2 < np.mean(map) < 0.2)


def suite():
    return unittest.makeSuite(PerturbationSensitivityAnalyzerTests)


if __name__ == '__main__':
    import runner

