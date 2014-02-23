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

from mvpa2.measures.rsa import *
from mvpa2.base import externals
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, pearsonr

data = np.array([[ 0.22366105, 0.51562476, 0.62623543, 0.28081652, 0.56513533],
                [ 0.22077129, 0.63013374, 0.19641318, 0.38466208, 0.60788347],
                [ 0.64273055, 0.60455658, 0.71368501, 0.36652763, 0.51720253],
                [ 0.40148338, 0.34188668, 0.09174233, 0.33906488, 0.17804584],
                [ 0.60728718, 0.6110304 , 0.84817742, 0.33830628, 0.7123945 ],
                [ 0.32113428, 0.16916899, 0.53471886, 0.93321617, 0.22531679]])


def test_DissimilarityConsistencyMeasure():
    targets = np.tile(xrange(3),2)
    chunks = np.repeat(np.array((0,1)),3)
    # correct results
    cres1 = 0.41894348
    cres2 = np.array([[ 0.16137995, 0.73062639, 0.59441713]]).T
    dc1 = data[0:3,:] - np.mean(data[0:3,:],0)
    dc2 = data[3:6,:] - np.mean(data[3:6,:],0)
    center = squareform(np.corrcoef(pdist(dc1,'correlation'),pdist(dc2,'correlation')), 
                        checks=False).reshape((1,-1))
    dsm1 = stats.rankdata(pdist(data[0:3,:],'correlation').reshape((1,-1)))
    dsm2 = stats.rankdata(pdist(data[3:6,:],'correlation').reshape((1,-1)))

    spearman = squareform(np.corrcoef(np.vstack((dsm1,dsm2))), 
                        checks=False).reshape((1,-1))
    
    ds = dataset_wizard(samples=data, targets=targets, chunks=chunks)
    dscm = DissimilarityConsistencyMeasure()
    res1 = dscm(ds)
    dscm_c = DissimilarityConsistencyMeasure(center_data=True)
    res2 = dscm_c(ds)
    dscm_sp = DissimilarityConsistencyMeasure(consistency_metric='spearman')
    res3 = dscm_sp(ds)
    ds.append(ds)
    chunks = np.repeat(np.array((0,1,2,)),4)
    ds.sa['chunks'] = chunks
    res4 = dscm(ds)
    assert_almost_equal(np.mean(res1.samples),cres1)
    assert_array_almost_equal(res2.samples, center)
    assert_array_almost_equal(res3.samples, spearman)
    assert_array_almost_equal(res4.samples,cres2)



def test_DissimilarityMatrixMeasure():
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
    dsm1 = DissimilarityMatrixMeasure()
    dsm2 = DissimilarityMatrixMeasure(pairwise_metric='euclidean')
    dsm3 = DissimilarityMatrixMeasure(pairwise_metric='cityblock')
    dsm4 = DissimilarityMatrixMeasure(center_data=True,square=True)
    assert_array_almost_equal(dsm1(ds).samples,pear)
    assert_array_almost_equal(dsm2(ds).samples,euc)
    assert_array_almost_equal(dsm3(ds).samples,city)
    assert_array_almost_equal(dsm4(ds).samples,center_sq)

def test_TargetDissimilarityCorrelationMeasure():
    ds = Dataset(data)
    tdsm = range(15)
    ans1 = np.array([0.30956920104253222, 0.26152022709856804])
    ans2 = np.array([0.53882710751962437, 0.038217527859375197])
    ans3 = np.array([0.33571428571428574, 0.22121153763932569])
    tdcm1 = TargetDissimilarityCorrelationMeasure(tdsm)
    tdcm2 = TargetDissimilarityCorrelationMeasure(tdsm,
                                            pairwise_metric='euclidean')
    tdcm3 = TargetDissimilarityCorrelationMeasure(tdsm,
                                comparison_metric = 'spearman')
    tdcm4 = TargetDissimilarityCorrelationMeasure(tdsm,
                                    corrcoef_only=True)
    a1 = tdcm1(ds)
    a2 = tdcm2(ds)
    a3 = tdcm3(ds)
    a4 = tdcm4(ds)
    assert_array_almost_equal(a1.samples,ans1.reshape(-1,1))
    assert_array_almost_equal(a2.samples,ans2.reshape(-1,1))
    assert_array_almost_equal(a3.samples,ans3.reshape(-1,1))
    assert_array_almost_equal(a4.samples,ans1[0].reshape(-1,1))




