# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA GNB classifier"""

import numpy as np

from mvpa2.testing import *
from mvpa2.testing.datasets import *

from mvpa2.clfs.gnb import GNB
from mvpa2.measures.base import TransferMeasure
from mvpa2.generators.splitters import Splitter
from mvpa2.misc.data_generators import normal_feature_dataset

class GNBTests(unittest.TestCase):

    def test_gnb(self):
        gnb = GNB()
        gnb_nc = GNB(common_variance=False)
        gnb_n = GNB(normalize=True)
        gnb_n_nc = GNB(normalize=True, common_variance=False)
        gnb_lin = GNB(common_variance=True)

        ds = datasets['uni2medium']

        # Generic silly coverage just to assure that it works in all
        # possible scenarios:
        bools = (True, False)
        # There should be better way... heh
        for cv in bools:                # common_variance?
          for prior in ('uniform', 'laplacian_smoothing', 'ratio'):
            tp = None                   # predictions -- all above should
                                        # result in the same predictions
            for n in bools:             # normalized?
              for ls in bools:          # logspace?
                for es in ((), ('estimates')):
                    gnb_ = GNB(common_variance=cv,
                               prior=prior,
                               normalize=n,
                               logprob=ls,
                               enable_ca=es)
                    tm = TransferMeasure(gnb_, Splitter('train'))
                    predictions = tm(ds).samples[:,0]
                    if tp is None:
                        tp = predictions
                    assert_array_equal(predictions, tp)
                    # if normalized -- check if estimates are such
                    if n and 'estimates' in es:
                        v = gnb_.ca.estimates
                        if ls:          # in log space -- take exp ;)
                            v = np.exp(v)
                        d1 = np.sum(v, axis=1) - 1.0
                        self.assertTrue(np.max(np.abs(d1)) < 1e-5)
                    # smoke test to see whether invocation of sensitivity analyser blows
                    # if gnb classifier isn't linear, and to see whether it doesn't blow
                    # when it is linear.
                    if cv:
                        assert 'has_sensitivity' in gnb_.__tags__
                        gnb_.get_sensitivity_analyzer()
                    if not cv:
                        with self.assertRaises(NotImplementedError):
                            gnb_.get_sensitivity_analyzer()


def test_gnb_sensitivities():
    gnb = GNB(common_variance=True)
    ds = normal_feature_dataset(perlabel=4,
                                nlabels=3,
                                nfeatures=5,
                                nchunks=4,
                                snr=20,
                                nonbogus_features=[0, 1, 2]
                                )

    s = gnb.get_sensitivity_analyzer()(ds)
    assert_in('targets', s.sa)
    assert_equal(s.shape, (((len(ds.uniquetargets) * (len(ds.uniquetargets) - 1))/2), ds.nfeatures))
    # test zero variance case
    # set variance of feature to zero
    ds.samples[:, 3] = 0.3
    s_zerovar = gnb.get_sensitivity_analyzer()
    sens = s_zerovar(ds)
    assert_equal(sens.T.dtype, 'O')  # we store pairs
    assert_equal(sens.T[0], ('L0', 'L1'))
    assert_true(all(sens.samples[:, 3] == 0))

    # test whether tagging and untagging works
    assert 'has_sensitivity' in gnb.__tags__
    gnb.untrain()
    assert 'has_sensitivity' not in gnb.__tags__

    # test whether content of sensitivities makes rough sense
    # First feature has information only about L0, so it would be of
    # no use for L1 -vs- L2 classification, so we will go through each pair
    # and make sure that signs etc all correct for each pair.
    # This in principle should be a generic test for multiclass sensitivities
    abssens = abs(sens.samples)
    for (t1, t2), t1t2sens in zip(sens.T, sens.samples):
        # go from literal L1 to 1, L0 to 0 - corresponds to feature
        i1 = int(t1[1])
        i2 = int(t2[1])
        assert t1t2sens[i1] < 0
        assert t1t2sens[i2] > 0
        assert t1t2sens[i2] > t1t2sens[4]


def suite():  # pragma: no cover
    return unittest.makeSuite(GNBTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

