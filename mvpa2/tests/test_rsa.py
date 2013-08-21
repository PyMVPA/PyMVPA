# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for rsa measures"""

from mvpa2.testing import sweepargs
from mvpa2.testing.datasets import datasets
from mvpa2.measures.anova import OneWayAnova

import numpy as np
from mvpa2.mappers.fx import *
from mvpa2.datasets.base import dataset_wizard, Dataset

from mvpa2.testing.tools import *

from mvpa2.measures.rsa import *
import scipy.stats as stats

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
    cres2 = np.array([[ 0.16137995, 0.73062639, 0.59441713]])
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
    euc = pdist(data, 'euclidean').reshape((1,-1))
    pear = pdist(data, 'correlation').reshape((1,-1))
    city = pdist(data, 'cityblock').reshape((1,-1))
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


