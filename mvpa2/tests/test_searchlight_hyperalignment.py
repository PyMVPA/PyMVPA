# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for Searchlight Hyperalignment ..."""

import unittest
import numpy as np

from mvpa2.algorithms.searchlight_hyperalignment import SearchlightHyperalignment, \
    HyperalignmentMeasure, compute_feature_scores
from mvpa2.mappers.zscore import zscore
from mvpa2.misc.support import idhash
from mvpa2.misc.data_generators import \
        random_affine_transformation, local_random_affine_transformations
from mvpa2.misc.neighborhood import Sphere, scatter_neighborhoods

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets
from mvpa2.base.dataset import hstack
from mvpa2.mappers.staticprojection import StaticProjectionMapper


class SearchlightHyperalignmentTests(unittest.TestCase):

    def get_testdata(self):
        # get a dataset with some prominent trends in it
        ds4l = datasets['uni4large']
        # lets select for now only meaningful features
        ds_orig = ds4l[:, ds4l.a.nonbogus_features]
        zscore(ds_orig, chunks_attr=None)
        n = 4  # # of datasets to generate
        Rs, dss_rotated, dss_rotated_clean = [], [], []
        # now lets compose derived datasets by using some random
        # rotation(s)
        for i in xrange(n):
            ds_ = random_affine_transformation(ds_orig, scale_fac=1.0, shift_fac=0.)
            Rs.append(ds_.a.random_rotation)
            zscore(ds_, chunks_attr=None)
            dss_rotated_clean.append(ds_)
            ds_2 = hstack([ds_, ds4l[:, ds4l.a.bogus_features[i*4:i*4+4]]])
            zscore(ds_2, chunks_attr=None)
            dss_rotated.append(ds_2)
        return ds_orig, dss_rotated, dss_rotated_clean, Rs

    def test_compute_feature_scores(self):
        ds_orig, dss_rotated, dss_rotated_clean, _ = self.get_testdata()
        fs_clean = compute_feature_scores(dss_rotated_clean)
        fs_noisy = compute_feature_scores(dss_rotated)
        # Making sure features with least score are all bogus features
        assert_true(np.all([np.all(np.argsort(fs)[:4] > 3) for fs in fs_noisy]))
        # Making sure the feature scores of true features are in the same order
        # assert_array_equal(np.argsort(np.asarray(fs_noisy)[:, :4]),
        #                   np.argsort(np.asarray(fs_clean)))
        # exclude dataset tests
        fs_no4 = compute_feature_scores(dss_rotated, exclude_from_model=[3])
        fs_3ds = compute_feature_scores(dss_rotated[:3])
        # feature scores of non-excluded datasets shouldn't change
        assert_array_equal(np.asarray(fs_no4)[:3, ], np.asarray(fs_3ds))
        # feature scores of excluded dataset should still make sense and be the same
        assert_array_equal(fs_noisy[3], fs_no4[3])
        # Not true for non-excluded datasets
        assert(np.alltrue(np.asarray(fs_noisy)[:3, ] != np.asarray(fs_no4)[:3, ]))
        fs_no34 = compute_feature_scores(dss_rotated, exclude_from_model=[2, 3])
        fs_2ds = compute_feature_scores(dss_rotated[:2])
        # feature scores of non-exluded datasets shouldn't change
        assert_array_equal(np.asarray(fs_no34)[:2, ], np.asarray(fs_2ds))

    def test_hyperalignment_measure(self):
        ref_ds = 0
        ha = HyperalignmentMeasure()
        ds_orig, dss_rotated, dss_rotated_clean, Rs = self.get_testdata()
        # Lets test two scenarios -- in one with no noise -- we should get
        # close to perfect reconstruction.  If noisy features were added -- not so good
        for noisy, dss in ((False, dss_rotated_clean),
                           (True, dss_rotated)):
            # to verify that original datasets didn't get changed by
            # Hyperalignment store their idhashes of samples
            idhashes = [idhash(ds.samples) for ds in dss]
            idhashes_targets = [idhash(ds.targets) for ds in dss]
            mappers = ha(dss)
            mappers = [StaticProjectionMapper(proj=m[0]['proj'], recon=m[0]['proj'].T)
                       for m in mappers.samples]
            idhashes_ = [idhash(ds.samples) for ds in dss]
            idhashes_targets_ = [idhash(ds.targets) for ds in dss]
            self.assertEqual(
                idhashes, idhashes_,
                msg="Hyperalignment must not change original data.")
            self.assertEqual(
                idhashes_targets, idhashes_targets_,
                msg="Hyperalignment must not change original data targets.")
            # Map data back
            dss_clean_back = [m.forward(ds_)
                              for m, ds_ in zip(mappers, dss)]
            _ = [zscore(sd, chunks_attr=None) for sd in dss_clean_back]
            ds_norm = np.linalg.norm(dss[ref_ds].samples)
            nddss = []
            ndcss = []
            nf = ds_orig.nfeatures
            ds_orig_Rref = np.dot(ds_orig.samples, Rs[ref_ds])
            zscore(ds_orig_Rref, chunks_attr=None)
            for ds_back in dss_clean_back:
                ndcs = np.diag(np.corrcoef(ds_back.samples.T[:nf, ],
                                           ds_orig_Rref.T)[nf:, :nf], k=0)
                ndcss += [ndcs]
                dds = ds_back.samples[:, :nf] - ds_orig_Rref
                ndds = np.linalg.norm(dds) / ds_norm
                nddss += [ndds]
            # First compare correlations
            snoisy = ('clean', 'noisy')[int(noisy)]
            self.assertTrue(
                np.all(np.array(ndcss) >= (0.9, 0.85)[int(noisy)]),
                msg="Should have reconstructed original dataset more or"
                " less. Got correlations %s in %s case."
                % (ndcss, snoisy))
            # normed differences
            self.assertTrue(
                np.all(np.array(nddss) <= (.2, 3)[int(noisy)]),
                msg="Should have reconstructed original dataset more or"
                " less for all. Got normed differences %s in %s case."
                % (nddss, snoisy))
            self.assertTrue(
                np.all(nddss[ref_ds] <= (.1, 0.2)[int(noisy)]),
                msg="Should have reconstructed original dataset quite "
                "well even with zscoring. Got normed differences %s "
                "in %s case." % (nddss, snoisy))
            self.assertTrue(
                np.all(np.array(nddss) >= 0.95 * nddss[ref_ds]),
                msg="Should have reconstructed orig_ds best of all. "
                "Got normed differences %s in %s case with ref_ds=%d."
                % (nddss, snoisy, ref_ds))
        # Testing feature selection within the measure
        ha_fs = HyperalignmentMeasure(featsel=0.5)
        ha = HyperalignmentMeasure()
        dss_rotated[ref_ds].fa['roi_seed'] = [1, 0, 0, 0, 0, 0, 0, 0]
        mappers_fs = ha_fs(dss_rotated)
        mappers = ha(dss_rotated_clean)
        mappers = [m[0]['proj'] for m in mappers.samples]
        mappers_fs = [m[0]['proj'] for m in mappers_fs.samples]
        # Testing the noisy features are eliminated
        assert_true(np.alltrue([np.all(m[4:, 4:] == 0) for m in mappers_fs]))
        # And it correctly maps the selected features
        for m, mfs in zip(mappers, mappers_fs):
            assert_array_equal(m, mfs[:4, :4])
        # testing roi_seed forces feature selection
        dss_rotated[ref_ds].fa['roi_seed'] = [0, 0, 0, 0, 0, 0, 0, 1]
        ha_fs = HyperalignmentMeasure(featsel=0.5)
        mappers_fs = ha_fs(dss_rotated)
        mappers_fs = [m[0]['proj'] for m in mappers_fs.samples]
        assert(np.alltrue([np.sum(m[7, :] == 0) == 4 for m in mappers_fs]))

    @sweepargs(nproc=(1, 2))
    def test_searchlight_hyperalignment(self, nproc):
        ds_orig = datasets['3dsmall']
        ds_orig.fa['voxel_indices'] = ds_orig.fa.myspace
        # toy data
        # data = np.random.randn(18,4,2)
        space = 'voxel_indices'
        # total number of datasets for the analysis
        nds = 5
        zscore(ds_orig, chunks_attr=None)
        dss = [ds_orig]
        # create  a few distorted datasets to match the desired number of datasets
        dss += [local_random_affine_transformations(ds_orig,
                    scatter_neighborhoods(Sphere(1),
                    ds_orig.fa[space].value, deterministic=True)[1], Sphere(2), space=space,
                    scale_fac=1.0, shift_fac=0.0)
                for i in range(nds-1)]
        _ = [zscore(sd, chunks_attr=None) for sd in dss[1:]]
        # we should have some distortion
        for ds in dss[1:]:
            assert_false(np.all(ds_orig.samples == ds.samples))
        # store projections for each mapper separately
        projs = [list() for i in range(nds)]
        # run the algorithm with all combinations of the two major parameters
        # for projection calculation.
        for kwargs in [{'combine_neighbormappers': True},
                       {'combine_neighbormappers': True, 'dtype': 'float64', 'compute_recon': True},
                       {'combine_neighbormappers': False},
                       {'combine_neighbormappers': False, 'mask_node_ids': np.arange(dss[0].nfeatures).tolist()},
                       {'combine_neighbormappers': True, 'exclude_from_model': [2, 4]},
                       {'combine_neighbormappers': True, 'sparse_radius': 1}]:
            slhyp = SearchlightHyperalignment(nproc=nproc, radius=2, **kwargs)
            print 'nds:', len(dss)
            mappers = slhyp(dss)
            # one mapper per input ds
            assert_equal(len(mappers), nds)
            for midx, m in enumerate(mappers):
                projs[midx].append(m)

        for midx, m in enumerate(mappers):
            # making sure mask_node_ids options works as expected
            assert_array_almost_equal(projs[midx][2].proj.todense(), projs[midx][3].proj.todense())
            # with different datatypes
            assert_array_almost_equal(projs[midx][0].proj.todense(), projs[midx][1].proj.todense(), decimal=5)
            assert_array_almost_equal(projs[midx][1].proj.todense(), projs[midx][1].recon.T.todense())
            assert_equal(projs[midx][1].proj.dtype, 'float64')
            assert_equal(projs[midx][0].proj.dtype, 'float32')
        # project data
        dss_hyper = [hm.forward(sd) for hm,sd in zip(mappers, dss)]
        _ = [zscore(sd, chunks_attr=None) for sd in dss_hyper]
        ndcss = []
        nf = ds_orig.nfeatures
        for ds_hyper in dss_hyper:
            ndcs = np.diag(np.corrcoef(ds_hyper.samples.T,
                                       ds_orig.samples.T)[nf:, :nf], k=0)
            ndcss += [ndcs]
        assert_true(np.median(ndcss[0]) > 0.9)
        assert_true(np.all([np.median(ndcs) > 0.2 for ndcs in ndcss[1:]]))


def suite():  # pragma: no cover
    return unittest.makeSuite(SearchlightHyperalignmentTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()
