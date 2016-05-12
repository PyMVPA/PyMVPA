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
    FeatureSelectionHyperalignment, compute_feature_scores
from mvpa2.mappers.zscore import zscore
from mvpa2.misc.support import idhash
from mvpa2.misc.data_generators import \
        random_affine_transformation, local_random_affine_transformations
from mvpa2.misc.neighborhood import Sphere, scatter_neighborhoods

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets
from mvpa2.base.dataset import hstack
from mvpa2.datasets.base import Dataset
from mvpa2.mappers.staticprojection import StaticProjectionMapper


class SearchlightHyperalignmentTests(unittest.TestCase):

    @reseed_rng()
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
        while len(dss_rotated_clean) < n:
            ds_ = random_affine_transformation(ds_orig, scale_fac=1.0, shift_fac=0.)
            if ds_.a.random_scale <= 0:
                continue
            Rs.append(ds_.a.random_rotation)
            zscore(ds_, chunks_attr=None)
            dss_rotated_clean.append(ds_)
            i = len(dss_rotated_clean) - 1
            ds_2 = hstack([ds_, ds4l[:, ds4l.a.bogus_features[i * 4: i * 4 + 4]]])
            zscore(ds_2, chunks_attr=None)
            dss_rotated.append(ds_2)
        return ds_orig, dss_rotated, dss_rotated_clean, Rs

    def test_compute_feature_scores(self):
        ds_orig, dss_rotated, dss_rotated_clean, _ = self.get_testdata()
        fs_clean = compute_feature_scores(dss_rotated_clean)
        fs_noisy = compute_feature_scores(dss_rotated)
        # Making sure features with least score are almost all bogus features
        assert_true(np.all([np.sum(np.argsort(fs)[:4] > 3) > 2 for fs in fs_noisy]))
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
        fsha = FeatureSelectionHyperalignment()
        ds_orig, dss_rotated, dss_rotated_clean, Rs = self.get_testdata()
        # Lets test two scenarios -- in one with no noise -- we should get
        # close to perfect reconstruction.  If noisy features were added -- not so good
        for noisy, dss in ((False, dss_rotated_clean),
                           (True, dss_rotated)):
            # to verify that original datasets didn't get changed by
            # Hyperalignment store their idhashes of samples
            idhashes = [idhash(ds.samples) for ds in dss]
            idhashes_targets = [idhash(ds.targets) for ds in dss]
            mappers = fsha(dss)
            mappers = [StaticProjectionMapper(proj=m, recon=m.T)
                       for m in mappers]
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
            nddss = []
            ndcss = []
            nf = ds_orig.nfeatures
            ds_norm = np.linalg.norm(dss[ref_ds].samples[:, :nf])
            ds_orig_Rref = np.dot(ds_orig.samples, Rs[ref_ds]) \
                           * np.sign(dss_rotated_clean[ref_ds].a.random_scale)
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
                nddss[ref_ds] <= (.1, 0.3)[int(noisy)],
                msg="Should have reconstructed original dataset quite "
                "well even with zscoring. Got normed differences %s "
                "in %s case." % (nddss, snoisy))
            self.assertTrue(
                np.all(np.array(nddss) / nddss[ref_ds] >= (0.95, 0.8)[int(noisy)]),
                msg="Should have reconstructed orig_ds best of all. "
                "Got normed differences %s in %s case with ref_ds=%d."
                % (nddss, snoisy, ref_ds))
        # Testing feature selection within the measure using fraction and count
        # same features
        fsha_fsf = FeatureSelectionHyperalignment(featsel=0.5)
        fsha_fsn = FeatureSelectionHyperalignment(featsel=4)
        fsha_fsf_same = FeatureSelectionHyperalignment(featsel=0.5, use_same_features=True)
        fsha = FeatureSelectionHyperalignment(full_matrix=False)
        # check for valueerror if full_matrix=False and no roi_seed fa
        self.assertRaises(ValueError, fsha, dss_rotated)
        fsha = FeatureSelectionHyperalignment()
        dss_rotated[ref_ds].fa['roi_seed'] = [1, 0, 0, 0, 0, 0, 0, 0]
        mappers_fsf = fsha_fsf(dss_rotated)
        mappers_fsf_same = fsha_fsf_same(dss_rotated)
        mappers_fsn = fsha_fsn(dss_rotated)
        mappers = fsha(dss_rotated_clean)
        mappers_diffsizedss = fsha_fsf([sd[:, nfs] for nfs, sd in
            zip([np.arange(5), np.random.permutation(np.arange(8)), np.arange(8)[::-1], np.arange(8)], dss_rotated)])
        # Testing that most of noisy features are eliminated from reference data
        assert_true(np.alltrue([np.sum(m[:4, :4].std(0) > 0) > 2 for m in mappers_fsf]))
        # using same features make it most likely to eliminate all noisy features
        assert_true(np.alltrue([np.sum(m[:4, :4].std(0) > 0) == 4 for m in mappers_fsf_same]))
        assert_true(np.alltrue([np.sum(m[:4, :4].std(0) > 0) > 2 for m in mappers_fsn]))
        # And it correctly maps the selected features if they are selected
        if np.alltrue([np.all(m[4:, :4] == 0) for m in mappers_fsf]):
            for m, mfs in zip(mappers, mappers_fsf):
                assert_array_equal(m, mfs[:4, :4])
        if np.alltrue([np.all(m[4:, :4] == 0) for m in mappers_fsf_same]):
            for m, mfs in zip(mappers, mappers_fsf_same):
                assert_array_equal(m, mfs[:4, :4])
        # testing roi_seed forces feature selection
        dss_rotated[ref_ds].fa['roi_seed'] = [0, 0, 0, 0, 0, 0, 0, 1]
        fsha_fsf = FeatureSelectionHyperalignment(featsel=0.5)
        mappers_fsf = fsha_fsf(dss_rotated)
        assert(np.alltrue([np.sum(m[7, :] == 0) == 4 for m in mappers_fsf]))

    @reseed_rng()
    def test_searchlight_hyperalignment(self):
        skip_if_no_external('scipy')
        skip_if_no_external('h5py')
        ds_orig = datasets['3dsmall'].copy()[:, :15]
        ds_orig.fa['voxel_indices'] = ds_orig.fa.myspace
        space = 'voxel_indices'
        # total number of datasets for the analysis
        nds = 5
        zscore(ds_orig, chunks_attr=None)
        dss = [ds_orig]
        # create a few distorted datasets to match the desired number of datasets
        # not sure if this truly mimics the real data, but at least we can test
        # implementation
        while len(dss) < nds - 1:
            sd = local_random_affine_transformations(
                ds_orig,
                scatter_neighborhoods(
                    Sphere(1),
                    ds_orig.fa[space].value, deterministic=True)[1],
                Sphere(2),
                space=space,
                scale_fac=1.0, shift_fac=0.0)
            # sometimes above function returns dataset with nans, infs, we don't want that.
            if np.sum(np.isnan(sd.samples)+np.isinf(sd.samples)) == 0 \
                    and np.all(sd.samples.std(0)):
                dss.append(sd)
        ds_orig_noisy = ds_orig.copy()
        ds_orig_noisy.samples += 0.1 * np.random.random(size=ds_orig_noisy.shape)
        dss.append(ds_orig_noisy)
        _ = [zscore(sd, chunks_attr=None) for sd in dss[1:]]
        # we should have some distortion
        for ds in dss[1:]:
            assert_false(np.all(ds_orig.samples == ds.samples))
        # testing checks
        slhyp = SearchlightHyperalignment(ref_ds=1, exclude_from_model=[1])
        self.assertRaises(ValueError, slhyp, dss[:3])
        slhyp = SearchlightHyperalignment(ref_ds=3)
        self.assertRaises(ValueError, slhyp, dss[:3])
        # store projections for each mapper separately
        projs = list()
        # run the algorithm with all combinations of the two major parameters
        # for projection calculation.
        for kwargs in [{'combine_neighbormappers': True, 'nproc': 1 + int(externals.exists('pprocess'))},
                       {'combine_neighbormappers': True, 'dtype': 'float64', 'compute_recon': True},
                       {'combine_neighbormappers': True, 'exclude_from_model': [2, 4]},
                       {'combine_neighbormappers': False},
                       {'combine_neighbormappers': False, 'mask_node_ids': np.arange(dss[0].nfeatures).tolist()},
                       {'combine_neighbormappers': True, 'sparse_radius': 1},
                       {'combine_neighbormappers': True, 'nblocks': 2}]:
            slhyp = SearchlightHyperalignment(radius=2, **kwargs)
            mappers = slhyp(dss)
            # one mapper per input ds
            assert_equal(len(mappers), nds)
            projs.append(mappers)
        # some checks
        for midx in range(nds):
            # making sure mask_node_ids options works as expected
            assert_array_almost_equal(projs[3][midx].proj.todense(),
                                      projs[4][midx].proj.todense())
            # recon check
            assert_array_almost_equal(projs[0][midx].proj.todense(),
                                      projs[1][midx].recon.T.todense(), decimal=5)
            assert_equal(projs[1][midx].proj.dtype, 'float64')
            assert_equal(projs[0][midx].proj.dtype, 'float32')
        # making sure the projections make sense
        for proj in projs:
            # no .max on sparse matrices on older scipy (e.g. on precise) so conver to array first
            max_weight = proj[0].proj.toarray().max(0).squeeze()
            diag_weight = proj[0].proj.diagonal()
            # Check to make sure diagonal is the max weight, in almost all rows for reference subject
            assert(np.sum(max_weight == diag_weight) / float(len(diag_weight)) > 0.90)
            # and not true for other subjects
            for i in range(1, nds - 1):
                assert(np.sum(proj[i].proj.toarray().max(0).squeeze() == proj[i].proj.diagonal())
                       / float(proj[i].proj.shape[0]) < 0.80)
            # Check to make sure projection weights match across duplicate datasets
            max_weight = proj[-1].proj.toarray().max(0).squeeze()
            diag_weight = proj[-1].proj.diagonal()
            # Check to make sure diagonal is the max weight, in almost all rows for reference subject
            assert(np.sum(max_weight == diag_weight) / float(len(diag_weight)) > 0.90)
        # project data
        dss_hyper = [hm.forward(sd) for hm, sd in zip(projs[0], dss)]
        _ = [zscore(sd, chunks_attr=None) for sd in dss_hyper]
        ndcss = []
        nf = ds_orig.nfeatures
        for ds_hyper in dss_hyper:
            ndcs = np.diag(np.corrcoef(ds_hyper.samples.T,
                                       ds_orig.samples.T)[nf:, :nf],
                           k=0)
            ndcss += [ndcs]
        assert_true(np.median(ndcss[0]) > 0.9)
        # noisy copy of original dataset should be similar to original after hyperalignment
        assert_true(np.median(ndcss[-1]) > 0.9)
        assert_true(np.all([np.median(ndcs) > 0.2 for ndcs in ndcss[1:-2]]))

    @reseed_rng()
    def test_searchlight_hyperalignment_warnings_and_exceptions(self):
        skip_if_no_external('scipy')
        skip_if_no_external('hdf5')  # needed for default results backend hdf5

        ds_orig = datasets['3dsmall'][:, :1]  # tiny dataset just to test exceptions
        ds_orig.fa['voxel_indices'] = ds_orig.fa.myspace
        slhyper = SearchlightHyperalignment()
        self.assertRaises(ValueError, slhyper, [ds_orig])  # need more than 1
        ds_orig.samples += 1.0  # not zscored for sure
        # TODO: we need assert_warnings to also capture our own warnings,
        # currently they are just suppressed :-/  So this is just a smoke test
        mappers = slhyper([ds_orig, ds_orig.copy()])

    @reseed_rng()
    def test_custom_qas(self):
        # Test if we could provide custom QEs per each of the datasets
        skip_if_no_external('scipy')
        skip_if_no_external('hdf5')  # needed for default results backend hdf5

        ns, nf = 10, 4  # # of samples/features -- a very BIG dataset ;)
        ds0 = Dataset(np.random.normal(size=(ns, nf)))
        zscore(ds0, chunks_attr=None)
        ds1 = ds0[:, [3, 0, 1, 2]]  # features circular shifted to the right

        qe0 = FancyQE([[0], [1], [2], [3]])  # does nothing
        qe1 = FancyQE([[1], [2], [3], [0]])  # knows to look into the right

        def apply_slhyper(queryengine, dss=[ds0, ds1], return_mappers=False, **kw):
            """Helper for a common code to create/call slhyper"""
            slhyper = SearchlightHyperalignment(queryengine=queryengine, **kw)
            mappers = slhyper(dss)
            proj = [m.proj.todense() for m in mappers]
            return (proj, mappers) if return_mappers else proj

        # since this single qe resulted in trying to match non-matching time series
        # projections should be non-identity, but no offdiagonal elements
        assert_no_offdiag(apply_slhyper(qe0))

        # both are provided
        projs, mappers = apply_slhyper([qe0, qe1], return_mappers=True)
        tprojs_shifted = [np.eye(nf), np.roll(np.eye(nf), 1, axis=0)]
        assert_array_equal(projs[0], tprojs_shifted[0])  # must be identity since we made them so
        assert_array_equal(projs[1], tprojs_shifted[1])  # pretty much incorporating that shift

        # TODO -- not identity assert_array_equal(projs[0], np.eye(len(p)))  # must be identity since we made them so
        # and must restore data properly
        assert_array_almost_equal(mappers[0].forward(ds0), mappers[1].forward(ds1))

        # give more then # of qes
        assert_raises(ValueError,
                      SearchlightHyperalignment(queryengine=[qe0, qe1]),
                      [ds0, ds1, ds0])

        # The one having no voxels for the "1st" id in "subj1"
        qe1_ = FancyQE([[1], [], [3], [0]])  # knows to look into the right

        projs = apply_slhyper(qe1_)
        assert_no_offdiag(projs)
        for proj in projs:
            # assess that both have '2nd' one 0
            # but not the others!
            assert_array_equal(np.diagonal(proj) != 0, [True, True, False, True])

        # smoke test whenever combine is False
        # In this case should work ok
        apply_slhyper(qe0, combine_neighbormappers=False)
        # this one ok as well since needs only matching ones in ref_ds
        apply_slhyper([qe0, qe1], combine_neighbormappers=False)
        # here since features do not match node_ids -- should raise ValueError
        assert_raises(ValueError, apply_slhyper, qe1, combine_neighbormappers=False)
        assert_raises(ValueError, apply_slhyper, [qe0, qe1], ref_ds=1, combine_neighbormappers=False)

        # and now only one qe lacking for that id
        projs = apply_slhyper([qe0, qe1_])
        tproj0 = np.eye(nf)
        tproj0[1, 1] = 0
        tprojs_shifted_1st0 = [tproj0, np.roll(tproj0, 1, axis=0)]
        for proj, tproj in zip(projs, tprojs_shifted_1st0):
            # assess that both have '2nd' one 0
            # but not the others!
            assert_array_equal(proj, tproj)

        # And now a test with varying number of selected fids, no shift
        qe0 = FancyQE([[0], [1, 2], [1, 2, 3], [0, 1, 2, 3]])
        projs = apply_slhyper(qe0)
        # Test that in general we get larger coefficients for "correct" transformation
        for p, tproj in zip(projs, tprojs_shifted):
            assert(np.all(np.asarray(p)[tproj>0] >= 1.0))
            assert_array_lequal(np.mean(np.asarray(p)[tproj == 0]), 0.3)

        qe1 = FancyQE([[0, 1, 2, 3], [1, 2, 3], [2, 3], [3]])
        # Just a smoke test, for now TODO
        projs = apply_slhyper([qe0, qe1])


def assert_no_offdiag(a):
    if isinstance(a, list):
        for a_ in a:
            assert_no_offdiag(a_)
    else:
        assert_array_equal(a - np.diag(np.diagonal(a)), 0)


class FancyQE(object):
    """Little helper QE which knows what to return for neighbors"""

    def __init__(self, results, ids=None):
        if ids is None:
            # then for all ids in results which is a list
            ids = list(range(len(results)))
        else:
            assert (len(results) >= max(ids))
            assert (min(ids) >= 0)
        self.ids = ids
        self.results = results

    def train(self, ds):
        # anything else to do ? ;)
        assert (ds.nfeatures > max(self.ids))

    def __getitem__(self, i):
        return self.results[i]


def suite():  # pragma: no cover
    return unittest.makeSuite(SearchlightHyperalignmentTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()
