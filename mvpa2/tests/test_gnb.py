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

        # Store probabilities for further comparison
        probabilities = {}

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
                        probabilities[repr(gnb_)] = v
                    # smoke test to see whether invocation of sensitivity analyser blows
                    # if gnb classifier isn't linear, and to see whether it doesn't blow
                    # when it is linear.
                    if cv:
                        assert 'has_sensitivity' in gnb_.__tags__
                        gnb_.get_sensitivity_analyzer()
                    if not cv:
                        with self.assertRaises(NotImplementedError):
                            gnb_.get_sensitivity_analyzer()

        # Verify that probabilities are identical when we use logprob or not
        assert_array_almost_equal(
            probabilities["GNB(space='targets', normalize=True, logprob=False)"],
            probabilities["GNB(space='targets', normalize=True)"]
        )
        assert_array_almost_equal(
            probabilities["GNB(space='targets', normalize=True, logprob=False, prior='uniform')"],
            probabilities["GNB(space='targets', normalize=True, prior='uniform')"]
        )


@reseed_rng()
@sweepargs(logprob=[True, False])
def test_gnb_sensitivities(logprob):
    gnb = GNB(common_variance=True, logprob=logprob)
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

    gnb.untrain()

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


@reseed_rng()
def test_gnb_overflow():
    # https://github.com/PyMVPA/PyMVPA/issues/581
    gnb = GNB(enable_ca='estimates',
              #logprob=True,  # implemented only for True ATM
              normalize=True,
              # uncomment if interested to trigger:
              # guard_overflows=False,
              )

    # Having lots of features could trigger under/overflows
    ds = normal_feature_dataset(perlabel=4,
                                nlabels=2,
                                nfeatures=100000,
                                nchunks=2,
                                snr=5,
                                nonbogus_features=[0, 1]
                                )

    ds_train = ds[ds.chunks == ds.UC[0]]
    ds_test = ds[ds.chunks == ds.UC[1]]

    gnb.train(ds_train)
    res = gnb.predict(ds_test)
    res_est = gnb.ca.estimates

    probs = np.exp(res_est) if gnb.params.logprob else res_est

    assert np.all(np.isfinite(res_est))
    assert np.all(np.isfinite(probs))
    assert_equal(sorted(np.unique(probs)), [0, 1]) # quantized into 0, 1 given this many samples


def _test_gnb_overflow_haxby():   # pragma: no cover
    # example from https://github.com/PyMVPA/PyMVPA/issues/581
    # a heavier version of the above test
    import os
    import numpy as np

    from mvpa2.datasets.sources.native import load_tutorial_data
    from mvpa2.clfs.gnb import GNB
    from mvpa2.measures.base import CrossValidation
    from mvpa2.generators.partition import HalfPartitioner
    from mvpa2.mappers.zscore import zscore
    from mvpa2.mappers.detrend import poly_detrend
    from mvpa2.datasets.miscfx import remove_invariant_features
    from mvpa2.testing.datasets import *

    datapath = '/usr/share/data/pymvpa2-tutorial/'
    haxby = load_tutorial_data(datapath,
                               roi='vt',
                               add_fa={'vt_thr_glm': os.path.join(datapath,
                                                                  'haxby2001',
                                                                  'sub001',
                                                                  'masks',
                                                                  'orig',
                                                                  'vt.nii.gz')})
    # poly_detrend(haxby, polyord=1, chunks_attr='chunks')
    haxby = haxby[np.array([l in ['rest', 'scrambled'] # ''house', 'face']
                            for l in haxby.targets], dtype='bool')]
    #zscore(haxby, chunks_attr='chunks', param_est=('targets', ['rest']),
    #       dtype='float32')
    # haxby = haxby[haxby.sa.targets != 'rest']
    haxby = remove_invariant_features(haxby)

    clf = GNB(enable_ca='estimates',
              logprob=True,
              normalize=True)

    #clf.train(haxby)
    #clf.predict(haxby)
    # estimates a bit "overfit" to judge in the train/predict on the same data

    cv = CrossValidation(clf,
                         HalfPartitioner(attr='chunks'),
                         postproc=None,
                         enable_ca=['stats'])

    cv_results = cv(haxby)
    res1_est = clf.ca.estimates
    print "Estimates:\n", res1_est
    print "Exp(estimates):\n", np.round(np.exp(res1_est), 3)
    assert np.all(np.isfinite(res1_est))


def suite():  # pragma: no cover
    return unittest.makeSuite(GNBTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

