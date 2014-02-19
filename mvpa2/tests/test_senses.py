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
from mvpa2.misc.data_generators import normal_feature_dataset
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.clfs.meta import SplitClassifier
from mvpa2.clfs.smlr import SMLR

@reseed_rng()
@labile(5, 1)
def test_splitclf_sensitivities():
    datasets = [normal_feature_dataset(perlabel=100, nlabels=2,
                                       nfeatures=4,
                                       nonbogus_features=[0, i + 1],
                                       snr=1, nchunks=2)
                for i in xrange(2)]

    sclf = SplitClassifier(SMLR(),
                           NFoldPartitioner())
    analyzer = sclf.get_sensitivity_analyzer()

    senses1 = analyzer(datasets[0])
    senses2 = analyzer(datasets[1])

    for senses in senses1, senses2:
        # This should be False when comparing two folds
        assert_false(np.allclose(senses.samples[0],
                                 senses.samples[2]))
        assert_false(np.allclose(senses.samples[1],
                                 senses.samples[3]))
    # Moreover with new data we should have got different results
    # (i.e. it must retrained correctly)
    for s1, s2 in zip(senses1, senses2):
        assert_false(np.allclose(s1, s2))

    # and we should have "selected" "correct" voxels
    for i, senses in enumerate((senses1, senses2)):
        assert_equal(set(np.argsort(np.max(np.abs(senses), axis=0))[-2:]),
                     set((0, i + 1)))


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()
