# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA ..."""

import unittest
import numpy as np
from itertools import combinations

from mvpa2.algorithms.statis import Statis
from mvpa2.misc.support import idhash
from mvpa2.misc.data_generators import random_affine_transformation
from mvpa2.base.dataset import hstack
from mvpa2.testing.datasets import datasets
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.datasets.base import Dataset
from mvpa2.testing import sweepargs
from mvpa2.testing.tools import assert_almost_equal


class StatisTests(unittest.TestCase):
    """
    Adapted test from Hyperalignment testing as they both aim to
    align datasets into common space
    """
    @sweepargs(hstacked=(True, False))
    def test_basic_alignment(self, hstacked):
        # get a dataset with some prominent trends in it
        ds4l = datasets['uni4large']
        ds4l = ds4l.get_mapped(mean_group_sample(['targets']))
        # lets select for now only meaningful features
        ds_orig = ds4l[:, ds4l.a.nonbogus_features]
        n = 4 # number of datasets to generate
        dss_rotated, dss_rotated_clean = [], []

        # now lets compose derived datasets by using some random
        # rotation(s)
        for i in xrange(n):
            ds_ = random_affine_transformation(ds_orig, scale_fac=10, shift_fac=10)
            # reusing random data from dataset itself
            random_noise = ds4l[:, ds4l.a.bogus_features[i*4:i*4+3]]
            dss_rotated_clean.append(ds_)
            ds_ = ds_.copy()
            #ds_.samples = ds_.samples + 0.1 * random_noise
            ds_ = hstack([ds_, random_noise])
            dss_rotated.append(ds_)
        # Add tables_attr
        for isub in xrange(n):
            dss_rotated[isub].fa['subject'] = np.repeat(isub,
                                                        dss_rotated[isub].nfeatures)
            dss_rotated_clean[isub].fa['subject'] = np.repeat(isub,
                                                        dss_rotated_clean[isub].nfeatures)
        # Error checks
        statis = Statis(tables_attr='subject')
        self.assertRaises(ValueError, statis, [dss_rotated[0]])
        self.assertRaises(ValueError, statis, dss_rotated[0])
        for noisy, dss in ((False, dss_rotated_clean),
                           (True, dss_rotated)):
            snoisy = ('clean', 'noisy')[int(noisy)]
            # Check to make sure STATIS didn't change input data
            idhashes = [idhash(ds.samples) for ds in dss]
            idhashes_targets = [idhash(ds.targets) for ds in dss]
            # run statis with list or hstack
            if hstacked:
                mappers = statis(hstack(dss))
            else:
                mappers = statis(dss)
            # Check to make sure STATIS didn't change input data
            idhashes_ = [idhash(ds.samples) for ds in dss]
            idhashes_targets_ = [idhash(ds.targets) for ds in dss]
            self.assertEqual(idhashes, idhashes_,
                msg="Statis must not change original data.")
            self.assertEqual(idhashes_targets, idhashes_targets_,
                msg="Statis must not change original data targets.")
            # Compare subject factor scores
            assert_almost_equal(statis.alpha, 1./n, decimal=1,
                                   err_msg="Datasets weights differ too much.")
            # forward with returned mappers
            dss_mapped = [m.forward(ds_)
                              for m, ds_ in zip(mappers, dss)]
            # Check all datasets have same shape
            for sd in dss_mapped:
                self.assertEqual(sd.shape, (ds_orig.nsamples, statis.outdim))
            # compute ISC in mapped space for all subject pairs
            ndcss = []
            for i,j in combinations(range(n), 2):
                ndcs = np.diag(np.corrcoef(dss_mapped[i].samples.T,
                                           dss_mapped[j].samples.T)[statis.outdim:, :statis.outdim], k=0)
                ndcss += [ndcs]
            # Compare correlations
            self.assertTrue(np.all(np.array(ndcss) >= (0.9, 0.7)[int(noisy)]),
                    msg="Should have reconstructed original dataset more or"
                    " less. Got correlations %s for %s case"
                    % (ndcss, snoisy))

    def test_incosistent_dataset(self):
        # get junk datasets
        # number of subjects
        n = 4
        dss = []
        for i in range(2):
            dss.append(Dataset(samples=np.random.randn(10, 5)))
        dss.append(dss[0][-1::-1, ])
        dss.append(dss[1][-1::-1, ])
        # Run statis
        statis = Statis()
        self.assertRaises(ValueError, statis, dss)

def suite():  # pragma: no cover
    return unittest.makeSuite(StatisTests)

if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()
