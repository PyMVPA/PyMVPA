# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA searchlight algorithm"""

import tempfile, time
import numpy.random as rnd

from math import ceil

import mvpa2
from mvpa2.testing import *
from mvpa2.testing.clfs import *
from mvpa2.testing.datasets import *

from mvpa2.datasets import Dataset, hstack
from mvpa2.base.types import is_datasetlike
from mvpa2.base import externals
from mvpa2.mappers.base import ChainMapper
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.clfs.transerror import ConfusionMatrix
from mvpa2.measures.searchlight import sphere_searchlight, Searchlight
from mvpa2.measures.gnbsearchlight import sphere_gnbsearchlight, \
     GNBSearchlight
from mvpa2.clfs.gnb import GNB

from mvpa2.measures.nnsearchlight import sphere_m1nnsearchlight, \
     M1NNSearchlight
from mvpa2.clfs.knn import kNN

from mvpa2.misc.neighborhood import IndexQueryEngine, Sphere, HollowSphere
from mvpa2.misc.errorfx import corr_error
from mvpa2.generators.partition import NFoldPartitioner, OddEvenPartitioner
from mvpa2.generators.permutation import AttributePermutator
from mvpa2.measures.base import CrossValidation


class SearchlightTests(unittest.TestCase):

    def setUp(self):
        self.dataset = datasets['3dlarge']
        # give the feature coord a more common name, matching the default of
        # the searchlight
        self.dataset.fa['voxel_indices'] = self.dataset.fa.myspace
        self._tested_pprocess = False


    # https://github.com/PyMVPA/PyMVPA/issues/67
    # https://github.com/PyMVPA/PyMVPA/issues/69
    def test_gnbsearchlight_doc(self):
        # Test either we excluded nproc from the docstrings
        ok_(not 'nproc' in GNBSearchlight.__init__.__doc__)
        ok_(not 'nproc' in GNBSearchlight.__doc__)
        ok_(not 'nproc' in sphere_gnbsearchlight.__doc__)
        # but present elsewhere
        ok_('nproc' in sphere_searchlight.__doc__)
        ok_('nproc' in Searchlight.__init__.__doc__)

    # https://github.com/PyMVPA/PyMVPA/issues/106
    def test_searchlights_doc_qe(self):
        # queryengine should not be provided to sphere_* helpers
        for sl in (sphere_searchlight,
                   sphere_gnbsearchlight,
                   sphere_m1nnsearchlight):
            for kw in ('queryengine', 'qe'):
                ok_(not kw in sl.__doc__,
                    msg='There should be no %r in %s.__doc__' % (kw, sl))

        # queryengine should be provided in corresponding classes __doc__s
        for sl in (Searchlight,
                   GNBSearchlight,
                   M1NNSearchlight):
            for kw in ('queryengine',):
                ok_(kw in sl.__init__.__doc__,
                    msg='There should be %r in %s.__init__.__doc__' % (kw, sl))
            for kw in ('qe',):
                ok_(not kw in sl.__init__.__doc__,
                    msg='There should be no %r in %s.__init__.__doc__' % (kw, sl))



    #def _test_searchlights(self, ds, sls, roi_ids, result_all):  # pragma: no cover

    @sweepargs(lrn_sllrn_SL_partitioner=
               [(GNB(common_variance=v, descr='GNB'), None,
                 sphere_gnbsearchlight,
                 NFoldPartitioner(cvtype=1),
                 0.                       # correction for the error range
                 )
                 for v in (True, False)] +
               # Mean 1 NN searchlights
               [(ChainMapper(
                   [mean_group_sample(['targets', 'partitions']),
                    kNN(1)], space='targets', descr='M1NN'),
                 kNN(1),
                 sphere_m1nnsearchlight,
                 NFoldPartitioner(0.5, selection_strategy='random', count=20),
                 0.05),
                # the same but with NFold(1) partitioner since it still should work
                (ChainMapper(
                   [mean_group_sample(['targets', 'partitions']),
                    kNN(1)], space='targets', descr='NF-M1NN'),
                 kNN(1),
                 sphere_m1nnsearchlight,
                 NFoldPartitioner(1),
                 0.05),
                ]
               )
    @sweepargs(do_roi=(False, True))
    @sweepargs(results_backend=('native', 'hdf5'))
    @reseed_rng()
    def test_spatial_searchlight(self, lrn_sllrn_SL_partitioner, do_roi=False,
                                 results_backend='native'):
        """Tests both generic and ad-hoc searchlights (e.g. GNBSearchlight)
        Test of and adhoc searchlight anyways requires a ground-truth
        comparison to the generic version, so we are doing sweepargs here
        """
        lrn, sllrn, SL, partitioner, correction = lrn_sllrn_SL_partitioner
        ## if results_backend == 'hdf5' and not common_variance:
        ##     # no need for full combination of all possible arguments here
        ##     return

        if __debug__ and 'ENFORCE_CA_ENABLED' in debug.active \
           and  isinstance(lrn, ChainMapper):
            raise SkipTest("Known to fail while trying to enable "
                           "training_stats for the ChainMapper (M1NN here)")


        # e.g. for M1NN we need plain kNN(1) for m1nnsl, but to imitate m1nn
        #      "learner" we must use a chainmapper atm
        if sllrn is None:
            sllrn = lrn
        ds = datasets['3dsmall'].copy()
        # Let's test multiclass here, so boost # of labels
        ds[6:18].T += 2
        ds.fa['voxel_indices'] = ds.fa.myspace

        # To assure that users do not run into incorrect operation due to overflows
        ds.samples += 5000
        ds.samples *= 1000
        ds.samples = ds.samples.astype(np.int16)

        # compute N-1 cross-validation for each sphere
        # YOH: unfortunately sample_clf_lin is not guaranteed
        #      to provide exactly the same results due to inherent
        #      iterative process.  Therefore lets use something quick
        #      and pure Python
        cv = CrossValidation(lrn, partitioner)

        skwargs = dict(radius=1, enable_ca=['roi_sizes', 'raw_results',
                                            'roi_feature_ids'])

        if do_roi:
            # select some random set of features
            nroi = rnd.randint(1, ds.nfeatures)
            # and lets compute the full one as well once again so we have a reference
            # which will be excluded itself from comparisons but values will be compared
            # for selected roi_id
            sl_all = SL(sllrn, partitioner, **skwargs)
            result_all = sl_all(ds)
            # select random features
            roi_ids = rnd.permutation(range(ds.nfeatures))[:nroi]
            skwargs['center_ids'] = roi_ids
        else:
            nroi = ds.nfeatures
            roi_ids = np.arange(nroi)
            result_all = None

        if results_backend == 'hdf5':
            skip_if_no_external('h5py')

        sls = [sphere_searchlight(cv, results_backend=results_backend,
                                  **skwargs),
               #GNBSearchlight(gnb, NFoldPartitioner(cvtype=1))
               SL(sllrn, partitioner, indexsum='fancy', **skwargs)
               ]

        if externals.exists('scipy'):
            sls += [ SL(sllrn, partitioner, indexsum='sparse', **skwargs)]

        # Test nproc just once
        if externals.exists('pprocess') and not self._tested_pprocess:
            sls += [sphere_searchlight(cv, nproc=2, **skwargs)]
            self._tested_pprocess = True

        # Provide the dataset and all those searchlights for testing
        #self._test_searchlights(ds, sls, roi_ids, result_all)
        #nroi = len(roi_ids)
        #do_roi = nroi != ds.nfeatures
        all_results = []
        for sl in sls:
            # run searchlight
            mvpa2.seed()                # reseed rng again for m1nnsl
            results = sl(ds)
            all_results.append(results)
            #print `sl`
            # check for correct number of spheres
            self.assertTrue(results.nfeatures == nroi)
            # and measures (one per xfold)
            if partitioner.cvtype == 1:
                self.assertTrue(len(results) == len(ds.UC))
            elif partitioner.cvtype == 0.5:
                # here we had 4 unique chunks, so 6 combinations
                # even though 20 max was specified for NFold
                self.assertTrue(len(results) == 6)
            else:
                raise RuntimeError("Unknown yet type of partitioner to check")
            # check for chance-level performance across all spheres
            # makes sense only if number of features was big enough
            # to get some stable estimate of mean
            if not do_roi or nroi > 20:
                # correction here is for M1NN class which has wider distribution
                self.assertTrue(
                    0.67 - correction < results.samples.mean() < 0.85 + correction,
                    msg="Out of range mean result: "
                    "lrn: %s  sllrn: %s  NROI: %d  MEAN: %.3f"
                    % (lrn, sllrn, nroi, results.samples.mean(),))

            mean_errors = results.samples.mean(axis=0)
            # that we do get different errors ;)
            self.assertTrue(len(np.unique(mean_errors) > 3))

            # check resonable sphere sizes
            self.assertTrue(len(sl.ca.roi_sizes) == nroi)
            self.assertTrue(len(sl.ca.roi_feature_ids) == nroi)
            for i, fids in enumerate(sl.ca.roi_feature_ids):
                self.assertTrue(len(fids) == sl.ca.roi_sizes[i])
            if do_roi:
                # for roi we should relax conditions a bit
                self.assertTrue(max(sl.ca.roi_sizes) <= 7)
                self.assertTrue(min(sl.ca.roi_sizes) >= 4)
            else:
                self.assertTrue(max(sl.ca.roi_sizes) == 7)
                self.assertTrue(min(sl.ca.roi_sizes) == 4)

            # check base-class state
            self.assertEqual(sl.ca.raw_results.nfeatures, nroi)

            # Test if we got results correctly for 'selected' roi ids
            if do_roi:
                assert_array_equal(result_all[:, roi_ids], results)

        if len(all_results) > 1:
            # if we had multiple searchlights, we can check either they all
            # gave the same result (they should have)
            aresults = np.array([a.samples for a in all_results])
            dresults = np.abs(aresults - aresults.mean(axis=0))
            dmax = np.max(dresults)
            self.assertTrue(dmax <= 1e-13)

        # Test the searchlight's reuse of neighbors
        for indexsum in ['fancy'] + (
            externals.exists('scipy') and ['sparse'] or []):
            sl = SL(sllrn, partitioner, indexsum='fancy',
                    reuse_neighbors=True, **skwargs)
            mvpa2.seed()
            result1 = sl(ds)
            mvpa2.seed()
            result2 = sl(ds)                # must be faster
            assert_array_equal(result1, result2)

    @reseed_rng()
    def test_adhocsearchlight_perm_testing(self):
        # just a smoke test pretty much
        ds = datasets['3dmedium'].copy()
        #ds.samples += np.random.normal(size=ds.samples.shape)*10
        ds.fa['voxel_indices'] = ds.fa.myspace
        from mvpa2.mappers.fx import mean_sample
        from mvpa2.clfs.stats import MCNullDist
        permutator = AttributePermutator('targets', count=8,
                                         limit='chunks')
        distr_est = MCNullDist(permutator, tail='left',
                               enable_ca=['dist_samples'])
        slargs = (kNN(1),
                  NFoldPartitioner(0.5,
                                   selection_strategy='random',
                                   count=9))
        slkwargs = dict(radius=1, postproc=mean_sample())

        sl_nodistr = sphere_m1nnsearchlight(*slargs, **slkwargs)
        skip_if_no_external('scipy')    # needed for null_t
        sl = sphere_m1nnsearchlight(
            *slargs,
            null_dist=distr_est,
            enable_ca=['null_t'],
            reuse_neighbors=True,
            **slkwargs
            )
        mvpa2.seed()
        res_nodistr = sl_nodistr(ds)
        mvpa2.seed()
        res = sl(ds)
        # verify that we at least got the same main result
        # ah (yoh) -- null dist is estimated before the main
        # estimate so we can't guarantee correspondence :-/
        # assert_array_equal(res_nodistr, res)
        # only resemblance (TODO, may be we want to get/setstate
        # for rng before null_dist.fit?)

        # and dimensions correspond
        assert_array_equal(distr_est.ca.dist_samples.shape,
                           (1, ds.nfeatures, 8))
        assert_array_equal(sl.ca.null_t.samples.shape,
                           (1, ds.nfeatures))

    def test_partial_searchlight_with_full_report(self):
        ds = self.dataset.copy()
        center_ids = np.zeros(ds.nfeatures, dtype='bool')
        center_ids[[3, 50]] = True
        ds.fa['center_ids'] = center_ids
        # compute N-1 cross-validation for each sphere
        cv = CrossValidation(GNB(), NFoldPartitioner())
        # contruct diameter 1 (or just radius 0) searchlight
        # one time give center ids as a list, the other one takes it from the
        # dataset itself
        sls = (sphere_searchlight(cv, radius=0, center_ids=[3, 50]),
               sphere_searchlight(None, radius=0, center_ids=[3, 50]),
               sphere_searchlight(cv, radius=0, center_ids='center_ids'),
               )
        for sl in sls:
            # assure that we could set cv post constructor
            if sl.datameasure is None:
                sl.datameasure = cv
            # run searchlight
            results = sl(ds)
            # only two spheres but error for all CV-folds
            self.assertEqual(results.shape, (len(self.dataset.UC), 2))
            # Test if results hold if we "set" a "new" datameasure
            sl.datameasure = CrossValidation(GNB(), NFoldPartitioner())
            results2 = sl(ds)
            assert_array_almost_equal(results, results2)

        # test if we graciously puke if center_ids are out of bounds
        dataset0 = ds[:, :50] # so we have no 50th feature
        self.assertRaises(IndexError, sls[0], dataset0)
        # but it should be fine on the one that gets the ids from the dataset
        # itself
        results = sl(dataset0)
        assert_equal(results.nfeatures, 1)
        # check whether roi_seeds are correct
        sl = sphere_searchlight(lambda x: np.vstack((x.fa.roi_seed, x.samples)),
                                radius=1, add_center_fa=True, center_ids=[12])
        res = sl(ds)
        assert_array_equal(res.samples[1:, res.samples[0].astype('bool')].squeeze(),
                           ds.samples[:, 12])


    def test_add_center_fa(self):
        # just a smoke test pretty much
        ds = datasets['3dsmall'].copy()

        # check that we do not mark anything as center whenever there is none
        def check_no_center(ds):
            assert(not np.any(ds.fa.center))
            return 1.0
        # or just a single center in our case
        def check_center(ds):
            assert(np.sum(ds.fa.center) == 1)
            return 1.0
        for n, check in [(HollowSphere(1,0), check_no_center),
                         (Sphere(0), check_center),
                         (Sphere(1), check_center)]:
            Searchlight(check,
                    IndexQueryEngine(myspace=n),
                    add_center_fa='center')(ds)
            # and no changes to original ds data, etc
            assert_array_equal(datasets['3dsmall'].fa.keys(), ds.fa.keys())
            assert_array_equal(datasets['3dsmall'].samples, ds.samples)


    def test_partial_searchlight_with_confusion_matrix(self):
        ds = self.dataset
        from mvpa2.clfs.stats import MCNullDist
        from mvpa2.mappers.fx import mean_sample, sum_sample

        # compute N-1 cross-validation for each sphere
        cm = ConfusionMatrix(labels=ds.UT)
        cv = CrossValidation(
            sample_clf_lin, NFoldPartitioner(),
            # we have to assure that matrix does not get flatted by
            # first vstack in cv and then hstack in searchlight --
            # thus 2 leading dimensions
            # TODO: RF? make searchlight/crossval smarter?
            errorfx=lambda *a: cm(*a)[None, None, :])
        # contruct diameter 2 (or just radius 1) searchlight
        sl = sphere_searchlight(cv, radius=1, center_ids=[3, 5, 50])

        # our regular searchlight -- to compare results
        cv_gross = CrossValidation(sample_clf_lin, NFoldPartitioner())
        sl_gross = sphere_searchlight(cv_gross, radius=1, center_ids=[3, 5, 50])

        # run searchlights
        res = sl(ds)
        res_gross = sl_gross(ds)

        # only two spheres but error for all CV-folds and complete confusion matrix
        assert_equal(res.shape, (len(ds.UC), 3, len(ds.UT), len(ds.UT)))
        assert_equal(res_gross.shape, (len(ds.UC), 3))

        # briefly inspect the confusion matrices
        mat = res.samples
        # since input dataset is probably balanced (otherwise adjust
        # to be per label): sum within columns (thus axis=-2) should
        # be identical to per-class/chunk number of samples
        samples_per_classchunk = len(ds) / (len(ds.UT) * len(ds.UC))
        ok_(np.all(np.sum(mat, axis= -2) == samples_per_classchunk))
        # and if we compute accuracies manually -- they should
        # correspond to the one from sl_gross
        assert_array_almost_equal(res_gross.samples,
                           # from accuracies to errors
                           1 - (mat[..., 0, 0] + mat[..., 1, 1]).astype(float)
                           / (2 * samples_per_classchunk))

        # and now for those who remained sited -- lets perform H0 MC
        # testing of this searchlight... just a silly one with minimal
        # number of permutations
        no_permutations = 10
        permutator = AttributePermutator('targets', count=no_permutations)

        # once again -- need explicit leading dimension to avoid
        # vstacking during cross-validation
        cv.postproc = lambda x: sum_sample()(x)[None, :]

        sl = sphere_searchlight(cv, radius=1, center_ids=[3, 5, 50],
                                null_dist=MCNullDist(permutator, tail='right',
                                                     enable_ca=['dist_samples']))
        res_perm = sl(ds)
        # XXX all of the res_perm, sl.ca.null_prob and
        #     sl.null_dist.ca.dist_samples carry a degenerate leading
        #     dimension which was probably due to introduced new axis
        #     above within cv.postproc
        assert_equal(res_perm.shape, (1, 3, 2, 2))
        assert_equal(sl.null_dist.ca.dist_samples.shape,
                     res_perm.shape + (no_permutations,))
        assert_equal(sl.ca.null_prob.shape, res_perm.shape)
        # just to make sure ;)
        ok_(np.all(sl.ca.null_prob.samples >= 0))
        ok_(np.all(sl.ca.null_prob.samples <= 1))

        # we should have got sums of hits across the splits
        assert_array_equal(np.sum(mat, axis=0), res_perm.samples[0])


    def test_chi_square_searchlight(self):
        # only do partial to save time

        # Can't yet do this since test_searchlight isn't yet "under nose"
        #skip_if_no_external('scipy')
        if not externals.exists('scipy'):
            return

        from mvpa2.misc.stats import chisquare

        cv = CrossValidation(sample_clf_lin, NFoldPartitioner(),
                enable_ca=['stats'])


        def getconfusion(data):
            cv(data)
            return chisquare(cv.ca.stats.matrix)[0]

        sl = sphere_searchlight(getconfusion, radius=0,
                         center_ids=[3, 50])

        # run searchlight
        results = sl(self.dataset)
        self.assertTrue(results.nfeatures == 2)


    def test_1d_multispace_searchlight(self):
        ds = Dataset([np.arange(6)])
        ds.fa['coord1'] = np.repeat(np.arange(3), 2)
        # add a second space to the dataset
        ds.fa['coord2'] = np.tile(np.arange(2), 3)
        measure = lambda x: "+".join([str(x) for x in x.samples[0]])
        # simply select each feature once
        res = Searchlight(measure,
                          IndexQueryEngine(coord1=Sphere(0),
                                           coord2=Sphere(0)),
                          nproc=1)(ds)
        assert_array_equal(res.samples, [['0', '1', '2', '3', '4', '5']])
        res = Searchlight(measure,
                          IndexQueryEngine(coord1=Sphere(0),
                                           coord2=Sphere(1)),
                          nproc=1)(ds)
        assert_array_equal(res.samples,
                           [['0+1', '0+1', '2+3', '2+3', '4+5', '4+5']])
        res = Searchlight(measure,
                          IndexQueryEngine(coord1=Sphere(1),
                                           coord2=Sphere(0)),
                          nproc=1)(ds)
        assert_array_equal(res.samples,
                           [['0+2', '1+3', '0+2+4', '1+3+5', '2+4', '3+5']])

    #@sweepargs(regr=regrswh[:])
    @reseed_rng()
    def test_regression_with_additional_sa(self):
        regr = regrswh[:][0]
        ds = datasets['3dsmall'].copy()
        ds.fa['voxel_indices'] = ds.fa.myspace

        # Create a new sample attribute which will be used along with
        # every searchlight
        ds.sa['beh'] = np.random.normal(size=(ds.nsamples, 2))

        # and now for fun -- lets create custom linar regression
        # targets out of some random feature and beh linearly combined
        rfeature = np.random.randint(ds.nfeatures)
        ds.sa.targets = np.dot(
            np.hstack((ds.sa.beh,
                       ds.samples[:, rfeature:rfeature + 1])),
            np.array([0.3, 0.2, 0.3]))

        class CrossValidationWithBeh(CrossValidation):
            """An adapter for regular CV which would hstack
               sa.beh to the searchlighting ds"""
            def _call(self, ds):
                return CrossValidation._call(
                    self,
                    Dataset(np.hstack((ds, ds.sa.beh)),
                            sa=ds.sa))
        cvbeh = CrossValidationWithBeh(regr, OddEvenPartitioner(),
                                       errorfx=corr_error)
        # regular cv
        cv = CrossValidation(regr, OddEvenPartitioner(),
                             errorfx=corr_error)

        slbeh = sphere_searchlight(cvbeh, radius=1)
        slmapbeh = slbeh(ds)
        sl = sphere_searchlight(cv, radius=1)
        slmap = sl(ds)

        assert_equal(slmap.shape, (2, ds.nfeatures))
        # SL which had access to beh should have got for sure better
        # results especially in the vicinity of the chosen feature...
        features = sl.queryengine.query_byid(rfeature)
        assert_array_lequal(slmapbeh.samples[:, features],
                            slmap.samples[:, features])

        # elsewhere they should tend to be better but not guaranteed

    @labile(5, 1)
    def test_usecase_concordancesl(self):
        import numpy as np
        from mvpa2.base.dataset import vstack
        from mvpa2.mappers.fx import mean_sample

        # Take our sample 3d dataset
        ds1 = datasets['3dsmall'].copy(deep=True)
        ds1.fa['voxel_indices'] = ds1.fa.myspace
        ds1.sa['subject'] = [1]  # not really necessary -- but let's for clarity
        ds1 = mean_sample()(ds1) # so we get just a single representative sample

        def corr12(ds):
            corr = np.corrcoef(ds.samples)
            assert(corr.shape == (2, 2)) # for paranoid ones
            return corr[0, 1]

        for nsc, thr, thr_mean in (
            (0, 1.0, 1.0),
            (0.1, 0.3, 0.8)):   # just a bit of noise
            ds2 = ds1.copy(deep=True)    # make a copy for the 2nd subject
            ds2.sa['subject'] = [2]
            ds2.samples += nsc * np.random.normal(size=ds1.shape)

            # make sure that both have the same voxel indices
            assert(np.all(ds1.fa.voxel_indices == ds2.fa.voxel_indices))
            ds_both = vstack((ds1, ds2))# join 2 images into a single dataset
                                        # with .sa.subject distinguishing both

            sl = sphere_searchlight(corr12, radius=2)
            slmap = sl(ds_both)
            ok_(np.all(slmap.samples >= thr))
            ok_(np.mean(slmap.samples) >= thr)

    def test_swaroop_case(self):
        """Test hdf5 backend to pass results on Swaroop's usecase
        """
        skip_if_no_external('h5py')
        from mvpa2.measures.base import Measure
        class sw_measure(Measure):
            def __init__(self):
                Measure.__init__(self, auto_train=True)
            def _call(self, dataset):
                # For performance measures -- increase to 50-200
                # np.sum here is just to get some meaningful value in
                # them
                #return np.ones(shape=(2, 2))*np.sum(dataset)
                return Dataset(
                    np.array([{'d': np.ones(shape=(5, 5)) * np.sum(dataset)}],
                             dtype=object))
        results = []
        ds = datasets['3dsmall'].copy(deep=True)
        ds.fa['voxel_indices'] = ds.fa.myspace

        our_custom_prefix = tempfile.mktemp()
        for backend in ['native'] + \
                (externals.exists('h5py') and ['hdf5'] or []):
            sl = sphere_searchlight(sw_measure(),
                                    radius=1,
                                    tmp_prefix=our_custom_prefix,
                                    results_backend=backend)
            t0 = time.time()
            results.append(np.asanyarray(sl(ds)))
            # print "Done for backend %s in %d sec" % (backend, time.time() - t0)
        # because of swaroop's ad-hoc (who only could recommend such
        # a construct?) use case, and absent fancy working assert_objectarray_equal
        # let's compare manually
        #assert_objectarray_equal(*results)
        if not externals.exists('h5py'):
            self.assertRaises(RuntimeError,
                              sphere_searchlight,
                              sw_measure(),
                              results_backend='hdf5')
            raise SkipTest('h5py required for test of backend="hdf5"')
        assert_equal(results[0].shape, results[1].shape)
        results = [r.flatten() for r in results]
        for x, y in zip(*results):
            assert_equal(x.keys(), y.keys())
            assert_array_equal(x['d'], y['d'])
        # verify that no junk is left behind
        tempfiles = glob.glob(our_custom_prefix + '*')
        assert_equal(len(tempfiles), 0)


    def test_nblocks(self):
        skip_if_no_external('pprocess')
        # just a basic test to see that we are getting the same
        # results with different nblocks
        ds = datasets['3dsmall'].copy(deep=True)[:, :13]
        ds.fa['voxel_indices'] = ds.fa.myspace
        cv = CrossValidation(GNB(), OddEvenPartitioner())
        res1 = sphere_searchlight(cv, radius=1, nproc=2)(ds)
        res2 = sphere_searchlight(cv, radius=1, nproc=2, nblocks=5)(ds)
        assert_array_equal(res1, res2)


    def test_custom_results_fx_logic(self):
        # results_fx was introduced for the blow-up-the-memory-Swaroop
        # where keeping all intermediate results of the dark-magic SL
        # hyperalignment is not feasible.  So it is desired to split
        # searchlight computation in more blocks while composing the
        # target result "on-the-fly" from available so far results.
        #
        # Implementation relies on using generators feeding the
        # results_fx with fresh results whenever those become
        # available.
        #
        # This test/example's "measure" creates files which should be
        # handled by the results_fx function and removed in this case
        # to check if we indeed have desired high number of blocks while
        # only limited nproc.
        skip_if_no_external('pprocess')

        tfile = tempfile.mktemp('mvpa', 'test-sl')

        ds = datasets['3dsmall'].copy()[:, :25] # smaller copy
        ds.fa['voxel_indices'] = ds.fa.myspace
        ds.fa['feature_id'] = np.arange(ds.nfeatures)

        nproc = 3 # it is not about computing -- so we will can
                  # start more processes than possibly having CPUs just to test
        nblocks = nproc * 7
        # figure out max number of features to be given to any proc_block
        # yoh: not sure why I had to +1 here... but now it became more robust and
        # still seems to be doing what was demanded so be it
        max_block = int(ceil(ds.nfeatures / float(nblocks)) + 1)

        def print_(s, *args):
            """For local debugging"""
            #print s, args
            pass

        def results_fx(sl=None, dataset=None, roi_ids=None, results=None):
            """It will "process" the results by removing those files
               generated inside the measure
            """
            res = []
            print_("READY")
            for x in results:
                ok_(isinstance(x, list))
                res.append(x)
                print_("R: ", x)
                for r in x:
                    # Can happen if we requested those .ca's enabled
                    # -- then automagically _proc_block would wrap
                    # results in a dataset... Originally detected by
                    # running with MVPA_DEBUG=.* which triggered
                    # enabling all ca's
                    if is_datasetlike(r):
                        r = np.asscalar(r.samples)
                    os.unlink(r)         # remove generated file
                print_("WAITING")

            results_ds = hstack(sum(res, []))

            # store the center ids as a feature attribute since we use
            # them for testing
            results_ds.fa['center_ids'] = roi_ids
            return results_ds

        def results_postproc_fx(results):
            for ds in results:
                ds.fa['test_postproc'] = np.atleast_1d(ds.a.roi_center_ids**2)
            return results

        def measure(ds):
            """The "measure" will check if a run with the same "index" from
               previous block has been processed by now
            """
            f = '%s+%03d' % (tfile, ds.fa.feature_id[0] % (max_block * nproc))
            print_("FID:%d f:%s" % (ds.fa.feature_id[0], f))

            # allow for up to few seconds to wait for the file to
            # disappear -- i.e. its result from previous "block" was
            # processed
            t0 = time.time()
            while os.path.exists(f) and time.time() - t0 < 4.:
                time.sleep(0.5) # so it does take time to compute the measure
                pass
            if os.path.exists(f):
                print_("ERROR: ", f)
                raise AssertionError("File %s must have been processed by now"
                                     % f)
            open(f, 'w').write('XXX')   # signal that we have computing this measure
            print_("RES: %s" % f)
            return f

        sl = sphere_searchlight(measure,
                                radius=0,
                                nproc=nproc,
                                nblocks=nblocks,
                                results_postproc_fx=results_postproc_fx,
                                results_fx=results_fx,
                                center_ids=np.arange(ds.nfeatures)
                                )

        assert_equal(len(glob.glob(tfile + '*')), 0) # so no junk around
        try:
            res = sl(ds)
            assert_equal(res.nfeatures, ds.nfeatures)
            # verify that we did have results_postproc_fx called
            assert_array_equal(res.fa.test_postproc, np.power(res.fa.center_ids, 2))
        finally:
            # remove those generated left-over files
            for f in glob.glob(tfile + '*'):
                os.unlink(f)

def suite():  # pragma: no cover
    return unittest.makeSuite(SearchlightTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

