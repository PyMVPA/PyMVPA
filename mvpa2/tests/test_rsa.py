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
    cres2 = np.array([[ 0.16137995],[ 0.73062639],[ 0.59441713]])
    ds = dataset_wizard(samples=data, targets=targets, chunks=chunks)
    dscm = DissimilarityConsistencyMeasure()
    dset = dscm(ds)
    assert_almost_equal(np.mean(dset.samples),cres1)
    ds.append(ds)
    chunks = np.repeat(np.array((0,1,2,)),4)
    ds.sa['chunks'] = chunks
    dset = dscm(ds)
    assert_array_almost_equal(dset.samples,cres2)

def test_DissimilarityMatrixMeasure():
    targets = np.tile(xrange(3),2)
    chunks = np.repeat(np.array((0,1)),3)
    ds = dataset_wizard(samples=data, targets=targets, chunks=chunks)
    data_c = data - np.mean(data,0)
    euc = pdist(data_c, 'euclidean').reshape((-1,1))
    pear = pdist(data_c, 'correlation').reshape((-1,1))
    city = pdist(data_c, 'cityblock').reshape((-1,1))
    nocenter = pdist(data,'correlation').reshape((-1,1))

    # Now center each chunk separately
    chunk1 = data[0:3,:] - np.mean(data[0:3,:],0)
    chunk2 = data[3:6,:] - np.mean(data[3:6,:],0)
    chunk1 = squareform(pdist(chunk1,'correlation'))
    chunk2 = squareform(pdist(chunk2, 'correlation'))
    perchunk = np.vstack((chunk1,chunk2))
    dsm1 = DissimilarityMatrixMeasure()
    dsm2 = DissimilarityMatrixMeasure(pairwise_metric='euclidean')
    dsm3 = DissimilarityMatrixMeasure(pairwise_metric='cityblock')
    dsm4 = DissimilarityMatrixMeasure(center_data=False)
    dsm5 = DissimilarityMatrixMeasure(chunks_attr='chunks',square=True)
    assert_array_almost_equal(dsm1(ds).samples,pear)
    assert_array_almost_equal(dsm2(ds).samples,euc)
    assert_array_almost_equal(dsm3(ds).samples,city)
    assert_array_almost_equal(dsm4(ds).samples,nocenter)
    assert_array_almost_equal(dsm5(ds).samples,perchunk)


