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
                    if cv:
                        assert 'has_sensitivity' in gnb_.__tags__
                        gnb_.get_sensitivity_analyzer()
                    if not cv:
                        with self.assertRaises(TypeError):
                            gnb_.get_sensitivity_analyzer()



def test_gnb_sensitivities():
    gnb = GNB(common_variance=True)
    ds_train, ds_test_A, ds_test_B, ds_test_both  = two_feat_two_class()
    gnb.train(ds_train)

    #test classifier on two datasets with only one of the two targets present
    predictions_A = gnb.predict(ds_test_A)
    predictions_B = gnb.predict(ds_test_B)

    #test classifier on dataset with both targets present
    predictions_both = gnb.predict(ds_test_both)

    #does the classifier in single-target case classify at least 90% correct?
    assert_approx_equal(np.mean(predictions_A), 1, significant=2)
    assert_approx_equal(np.mean(predictions_B), 0, significant=2)
    assert_array_almost_equal(predictions_both, ds_train.targets)

    sens = gnb.get_sensitivity_analyzer(force_train=False)

    #TODO: if I dont call sens with the classifier instance, it will return only an instance of
    #TODO: GNBWeights instead of a dataset with the weights. This does not mirror what smlr does,
    #TODO: What am I missing?
    print(sens(gnb).samples[0])
    #print(sens.shape)
    #assert_equal(sens(gnb).shape, (len(ds_train.UT) - 1, ds_train.nfeatures))

def suite():  # pragma: no cover
    return unittest.makeSuite(GNBTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

