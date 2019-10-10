# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for rsa measures"""

from mvpa2.testing import *
skip_if_no_external('scipy')

from mvpa2.testing.datasets import datasets
from mvpa2.measures.anova import OneWayAnova

import numpy as np
from mvpa2.mappers.fx import *
from mvpa2.datasets.base import dataset_wizard, Dataset

from mvpa2.testing.tools import *
from mvpa2.testing import _ENFORCE_CA_ENABLED

from mvpa2.measures.rsa import *
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.measures.base import CrossValidation
from mvpa2.base import externals
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import rankdata, pearsonr

data = np.array([[ 0.22366105, 0.51562476, 0.62623543, 0.28081652, 0.56513533],
                [ 0.22077129, 0.63013374, 0.19641318, 0.38466208, 0.60788347],
                [ 0.64273055, 0.60455658, 0.71368501, 0.36652763, 0.51720253],
                [ 0.40148338, 0.34188668, 0.09174233, 0.33906488, 0.17804584],
                [ 0.60728718, 0.6110304 , 0.84817742, 0.33830628, 0.7123945 ],
                [ 0.32113428, 0.16916899, 0.53471886, 0.93321617, 0.22531679]])


def test_PDistConsistency():
    targets = np.tile(xrange(3),2)
    chunks = np.repeat(np.array((0,1)),3)
    # correct results
    cres1 = 0.41894348
    cres2 = np.array([[ 0.73062639, 0.16137995, 0.59441713]]).T
    dc1 = data[0:3,:] - np.mean(data[0:3,:],0)
    dc2 = data[3:6,:] - np.mean(data[3:6,:],0)
    center = squareform(np.corrcoef(pdist(dc1,'correlation'),pdist(dc2,'correlation')), 
                        checks=False).reshape((1,-1))
    dsm1 = stats.rankdata(pdist(data[0:3,:],'correlation').reshape((1,-1)))
    dsm2 = stats.rankdata(pdist(data[3:6,:],'correlation').reshape((1,-1)))

    spearman = squareform(np.corrcoef(np.vstack((dsm1,dsm2))), 
                        checks=False).reshape((1,-1))
    
    ds = dataset_wizard(samples=data, targets=targets, chunks=chunks)
    dscm = PDistConsistency()
    res1 = dscm(ds)
    dscm_c = PDistConsistency(center_data=True)
    res2 = dscm_c(ds)
    dscm_sp = PDistConsistency(consistency_metric='spearman')
    res3 = dscm_sp(ds)
    ds.append(ds)
    chunks = np.repeat(['one', 'two', 'three'], 4)
    ds.sa['chunks'] = chunks
    res4 = dscm(ds)
    dscm_sq = PDistConsistency(square=True)
    res4_sq = dscm_sq(ds)
    for i, p in enumerate(res4.sa.pairs):
        sqval =  np.asscalar(res4_sq[res4_sq.sa.chunks == p[0],
                                     res4_sq.sa.chunks == p[1]])
        assert_equal(sqval, res4.samples[i, 0])
    assert_almost_equal(np.mean(res1.samples),cres1)
    assert_array_almost_equal(res2.samples, center)
    assert_array_almost_equal(res3.samples, spearman)
    assert_array_almost_equal(res4.samples,cres2)


def test_CDist():
    targets = np.tile(range(3), 2)
    chunks = np.repeat(np.array((0,1)), 3)
    ds = dataset_wizard(samples=data, targets=targets, chunks=chunks)
    train_data = ds[ds.sa.chunks == 0, ]
    test_data = ds[ds.sa.chunks == 1, ]

    # Check for nsamples match
    pymvpa_cdist = CDist(sattr=['targets','chunks'])
    pymvpa_cdist.train(train_data)
    assert_raises(ValueError, pymvpa_cdist, test_data[test_data.T < 2, ])
    # Check it create sa as intended
    res = pymvpa_cdist(test_data)
    assert_dict_keys_equal(res.sa, test_data.sa)

    # Some distance metrics
    metrics = ['euclidean', 'correlation', 'cityblock', 'mahalanobis']
    VI_mahalanobis = np.eye(5)
    for sattr in [['targets'], None]:
        for metric in metrics:
            metric_kwargs = {'VI': VI_mahalanobis} if metric == 'mahalanobis' \
                else {}
            scipy_cdist = cdist(train_data.samples, test_data.samples,
                        metric, **metric_kwargs)
            scipy_pdist = pdist(train_data.samples,
                                metric, **metric_kwargs)
            pymvpa_cdist = CDist(pairwise_metric=metric,
                        pairwise_metric_kwargs=metric_kwargs,
                        sattr=sattr)

            assert_true(not pymvpa_cdist.is_trained)
            pymvpa_cdist.train(train_data)
            assert_true(pymvpa_cdist.is_trained)
            res_cv = pymvpa_cdist(test_data)
            res_nocv = pymvpa_cdist(train_data)
            # Check to make sure the cdist results are close to CDist results
            assert_array_almost_equal(res_cv.samples.ravel(),
                                      scipy_cdist.ravel())
            # if called with train_data again, results should match with pdist
            assert_array_almost_equal(res_nocv.samples.ravel(),
                                      squareform(scipy_pdist).ravel())


def test_CDist_cval():
    if _ENFORCE_CA_ENABLED:
        # skip testing for now, since we are having issue with 'training_stats'
        raise SkipTest("Skipping test to avoid issue with 'training_stats while CA enabled")
    
    targets = np.tile(range(3), 2)
    chunks = np.repeat(np.array((0,1)), 3)
    ds = dataset_wizard(samples=data, targets=targets, chunks=chunks)

    cv = CrossValidation(CDist(),
                         generator=NFoldPartitioner(),
                         errorfx=None)
    res = cv(ds)
    # Testing to make sure the both folds return same results, as they should
    assert_array_almost_equal(res[res.sa.cvfolds == 0, ].samples.reshape(3, 3),
                       res[res.sa.cvfolds == 1, ].samples.reshape(3, 3).T)
    # Testing to make sure the last dimension is always 1 to make it work with Searchlights
    assert_equal(res.nfeatures, 1)


def test_PDist():
    targets = np.tile(xrange(3),2)
    chunks = np.repeat(np.array((0,1)),3)
    ds = dataset_wizard(samples=data, targets=targets, chunks=chunks)
    data_c = data - np.mean(data,0)
    # DSM matrix elements should come out as samples of one feature
    # to be in line with what e.g. a classifier returns -- facilitates
    # collection in a searchlight ...
    euc = pdist(data, 'euclidean')[None].T
    pear = pdist(data, 'correlation')[None].T
    city = pdist(data, 'cityblock')[None].T
    center_sq = squareform(pdist(data_c,'correlation'))

    # Now center each chunk separately
    dsm1 = PDist()
    dsm2 = PDist(pairwise_metric='euclidean')
    dsm3 = PDist(pairwise_metric='cityblock')
    dsm4 = PDist(center_data=True,square=True)
    assert_array_almost_equal(dsm1(ds).samples,pear)
    assert_array_almost_equal(dsm2(ds).samples,euc)
    dsm_res = dsm3(ds)
    assert_array_almost_equal(dsm_res.samples,city)
    # length correspondings to a single triangular matrix
    assert_equal(len(dsm_res.sa.pairs), len(ds) * (len(ds) - 1) / 2)
    # generate label pairs actually reflect the vectorform generated by
    # squareform()
    dsm_res_square = squareform(dsm_res.samples.T[0])
    for i, p in enumerate(dsm_res.sa.pairs):
        assert_equal(dsm_res_square[p[0], p[1]], dsm_res.samples[i, 0])
    dsm_res = dsm4(ds)
    assert_array_almost_equal(dsm_res.samples,center_sq)
    # sample attributes are carried over
    assert_almost_equal(ds.sa.targets, dsm_res.sa.targets)


def test_PDistTargetSimilarity():
    ds = Dataset(data)
    tdsm = range(15)
    ans1 = np.array([0.30956920104253222, 0.26152022709856804])
    ans2 = np.array([0.53882710751962437, 0.038217527859375197])
    ans3 = np.array([0.33571428571428574, 0.22121153763932569])
    tdcm1 = PDistTargetSimilarity(tdsm)
    tdcm2 = PDistTargetSimilarity(tdsm,
                                            pairwise_metric='euclidean')
    tdcm3 = PDistTargetSimilarity(tdsm,
                                comparison_metric = 'spearman')
    tdcm4 = PDistTargetSimilarity(tdsm,
                                    corrcoef_only=True)
    a1 = tdcm1(ds)
    a2 = tdcm2(ds)
    a3 = tdcm3(ds)
    a4 = tdcm4(ds)
    assert_array_almost_equal(a1.samples.squeeze(), ans1)
    assert_array_equal(a1.fa.metrics, ['rho', 'p'])
    assert_array_almost_equal(a2.samples.squeeze(), ans2)
    assert_array_equal(a2.fa.metrics, ['rho', 'p'])
    assert_array_almost_equal(a3.samples.squeeze(), ans3)
    assert_array_equal(a3.fa.metrics, ['rho', 'p'])
    assert_array_almost_equal(a4.samples.squeeze(), ans1[0])
    assert_array_equal(a4.fa.metrics, ['rho'])


def test_PDistTargetSimilaritySearchlight():
    # Test ability to use PDistTargetSimilarity in a searchlight
    from mvpa2.testing.datasets import datasets
    from mvpa2.mappers.fx import mean_group_sample
    from mvpa2.mappers.shape import TransposeMapper
    from mvpa2.measures.searchlight import sphere_searchlight
    ds = datasets['3dsmall'][:, :3]
    ds.fa['voxel_indices'] = ds.fa.myspace
    # use chunks values (4 of them) for targets
    ds.sa['targets'] = ds.sa.chunks
    ds = mean_group_sample(['chunks'])(ds)
    tdsm = np.arange(6)
    # We can run on full dataset
    tdcm1 = PDistTargetSimilarity(tdsm)
    a1 = tdcm1(ds)
    assert_array_equal(a1.fa.metrics, ['rho', 'p'])

    tdcm1_rho = PDistTargetSimilarity(tdsm, corrcoef_only=True)
    sl_rho = sphere_searchlight(tdcm1_rho)(ds)
    assert_array_equal(sl_rho.shape, (1, ds.nfeatures))

    # now with both but we need to transpose datasets
    tdcm1_both = PDistTargetSimilarity(tdsm, postproc=TransposeMapper())
    sl_both = sphere_searchlight(tdcm1_both)(ds)
    assert_array_equal(sl_both.shape, (2, ds.nfeatures))
    assert_array_equal(sl_both.sa.metrics, ['rho', 'p'])
    # rho must be exactly the same
    assert_array_equal(sl_both.samples[0], sl_rho.samples[0])
    # just because we are here and we can
    # Actually here for some reason assert_array_lequal gave me a trouble
    assert_true(np.all(sl_both.samples[1] <= 1.0))
    assert_true(np.all(0 <= sl_both.samples[1]))


def test_Regression():
    skip_if_no_external('skl')
    # a very correlated dataset
    corrdata = np.array([[1, 2], [10, 20], [-1, -2], [-10, -20]])
    # a perfect predictor
    perfect_pred = np.array([0, 2, 2, 2, 2, 0])

    ds = Dataset(corrdata)

    reg_types = ['lasso', 'ridge']

    # assert it pukes because predictor is not of the right shape
    assert_raises(ValueError, Regression, perfect_pred)

    # now make it right
    perfect_pred = np.atleast_2d(perfect_pred).T
    # assert it pukes for unknown method
    assert_raises(ValueError, Regression, perfect_pred, method='bzot')

    for reg_type in reg_types:
        regr = Regression(perfect_pred, alpha=0, fit_intercept=False,
                          rank_data=False, normalize=False, method=reg_type)
    coefs = regr(ds)
    assert_almost_equal(coefs.samples, 1.)

    # assert it pukes if predictor and ds have different shapes
    regr = Regression(perfect_pred)
    assert_raises(ValueError, regr, ds[:-1])

    # what if we select some items?
    keep_pairss = [range(3), [1], np.arange(3)]
    for reg_type in reg_types:
        for keep_pairs in keep_pairss:
            regr = Regression(perfect_pred, keep_pairs=keep_pairs, alpha=0,
                              fit_intercept=False, rank_data=False, normalize=False,
                              method=reg_type)
            coefs = regr(ds)
            assert_almost_equal(coefs.samples, 1.)

    # make a less perfect predictor
    bad_pred = np.ones((6, 1))
    predictors = np.hstack((perfect_pred, bad_pred))

    # check it works with combination of parameters
    from itertools import product
    outputs =  [np.array([[0.], [0.], [0.]]),
               np.array([[0.76665188], [0.], [0.]]),
               np.array([[ 0.5], [0.], [1.75]]),
               np.array([[0.92307692], [0.], [0.26923077]]),
               np.array([[0.], [0.], [ 3.70074342e-17]]),
               np.array([[8.57142857e-01], [0.], [-2.64338815e-17]]),
               np.array([[0.], [0.], [1.33333333]]),
               np.array([[0.84210526], [0.], [0.21052632]]),
               np.array([[0.], [0.]]),
               np.array([[0.76665188], [0.]]),
               np.array([[0.92982456], [0.]]),
               np.array([[0.92850288], [0.07053743]]),
               np.array([[0.], [0.]]),
               np.array([[0.85714286], [0.]]),
               np.array([[0.625], [0.]]),
               np.array([[0.87272727], [0.14545455]])]

    for i, (fit_intercept, rank_data, normalize, reg_type) in \
            enumerate(
                    product([True, False], [True, False],
                            [True, False], reg_types)):
        regr = Regression(predictors, alpha=1,
                               fit_intercept=fit_intercept, rank_data=rank_data,
                               normalize=normalize, method=reg_type)
        coefs = regr(ds)
        # check we get all the coefficients we need
        wanted_samples = 3 if fit_intercept else 2
        assert_equal(coefs.nsamples, wanted_samples)
        # check we get the actual output
        assert_almost_equal(coefs.samples, outputs[i])




