# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Sensitivity Analyzers"""

import numpy as np

from mvpa2.testing import *
from mvpa2.testing import _ENFORCE_CA_ENABLED

from mvpa2.datasets.base import dataset_wizard
from mvpa2.misc.data_generators import normal_feature_dataset
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.clfs.meta import SplitClassifier
from mvpa2.clfs.smlr import SMLR

class SensitivityTests(unittest.TestCase):

    def setUp(self):
        self.dataset = normal_feature_dataset(perlabel=100, nlabels=2,
                                              nfeatures=10,
                                              nonbogus_features=[0,1],
                                              snr=0.3, nchunks=2)
        #zscore(dataset, chunks_attr='chunks')

    def test_split_clf(self):
        # set up the classifier
        sclf = SplitClassifier(SMLR(),
                               NFoldPartitioner())

        analyzer = sclf.get_sensitivity_analyzer()

        senses = analyzer(self.dataset)

        # This should be False when comparing two folds
        assert_false(np.allclose(senses.samples[0],senses.samples[2]))


if __name__ == '__main__':
    import runner
