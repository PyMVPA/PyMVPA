# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Shrinkage LDA classifier"""
# based on test_gnb.py

import numpy as np
import sklearn as skl

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as sklLDA
from mvpa2.clfs.skl.base import SKLLearnerAdapter

from mvpa2.misc.data_generators import normal_feature_dataset
from mvpa2.clfs.shrinklda import ShrinkageLDA
from mvpa2.measures.base import CrossValidation
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.testing import *

class ShrinkageLDATests(unittest.TestCase):

    def test_shrinkage_lda_faster_than_skl(self):
        # Compare PyMVPA's shrinkage LDA implementation with scikit-learn's
        # implementation when used through SKLLearnerAdapter. Are prediction
        # accuracies next to identical? Is the PyMVPA implementation faster?

        # Dataset properties: should be of type "high p, low n" as is typical
        # for fMRI data
        n_features = 500
        n_chunks = 10
        n_classes = 3
        n_samples_per_class = 60 # should be multiple of n_chunks
        snr = 3.0

        # Shrinkage estimators to use. OAS not an available option for sklLDA.
        shrink_type = 'ledoit-wolf'

        # The competitors
        pymvpa_clf = ShrinkageLDA(shrinkage_estimator='ledoit-wolf')
        # Using 'lsqr' because it is claimed to be more efficient than 'svd'
        # see sklearn doc.
        skl_clf = SKLLearnerAdapter(sklLDA(solver='lsqr', shrinkage='auto'))

        ds = normal_feature_dataset(perlabel=n_samples_per_class,
        nlabels=n_classes, nfeatures=n_features, nchunks=n_chunks,
        nonbogus_features=range(n_classes), snr=snr)

        partitioner = NFoldPartitioner(attr='chunks')

        delta_t = []
        for clf in [pymvpa_clf, skl_clf]:
            cv = CrossValidation(clf, partitioner)
            _ = cv(ds)
            delta_t.append(cv.ca.calling_time)

        # Was the PyMVPA version faster than the wrapped sklearn version?
        self.assertTrue(delta_t[0] < delta_t[1])

def suite():  # pragma: no cover
    return unittest.makeSuite(ShrinkageLDATests)

if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()
