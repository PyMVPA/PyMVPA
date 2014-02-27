# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA ZScore mapper"""


from mvpa2.base import externals

from mvpa2.support.copy import deepcopy
import numpy as np

from mvpa2.datasets.base import dataset_wizard
from mvpa2.mappers.zscore import ZScoreMapper, zscore
from mvpa2.testing.tools import assert_array_almost_equal, assert_array_equal, \
        assert_equal, assert_raises, ok_, nodebug
from mvpa2.misc.support import idhash

from mvpa2.testing.datasets import datasets

def test_mapper_vs_zscore():
    """Test by comparing to results of elderly z-score function
    """
    # data: 40 sample feature line in 20d space (40x20; samples x features)
    dss = [
        dataset_wizard(np.concatenate(
            [np.arange(40) for i in range(20)]).reshape(20,-1).T,
                targets=1, chunks=1),
        ] + datasets.values()

    for ds in dss:
        ds1 = deepcopy(ds)
        ds2 = deepcopy(ds)

        zsm = ZScoreMapper(chunks_attr=None)
        assert_raises(RuntimeError, zsm.forward, ds1.samples)
        idhashes = (idhash(ds1), idhash(ds1.samples))
        zsm.train(ds1)
        idhashes_train = (idhash(ds1), idhash(ds1.samples))
        assert_equal(idhashes, idhashes_train)

        # forward dataset
        ds1z_ds = zsm.forward(ds1)
        idhashes_forwardds = (idhash(ds1), idhash(ds1.samples))
        # must not modify samples in place!
        assert_equal(idhashes, idhashes_forwardds)

        # forward samples explicitly
        ds1z = zsm.forward(ds1.samples)
        idhashes_forward = (idhash(ds1), idhash(ds1.samples))
        assert_equal(idhashes, idhashes_forward)

        zscore(ds2, chunks_attr=None)
        assert_array_almost_equal(ds1z, ds2.samples)
        assert_array_equal(ds1.samples, ds.samples)

@nodebug(['ID_IN_REPR', 'MODULE_IN_REPR'])
def test_zcore_repr():
    # Just basic test if everything is sane... no proper comparison
    for m in (ZScoreMapper(chunks_attr=None),
              ZScoreMapper(params=(3, 1)),
              ZScoreMapper()):
        mr = eval(repr(m))
        ok_(isinstance(mr, ZScoreMapper))

def test_zscore():
    """Test z-scoring transformation
    """
    # dataset: mean=2, std=1
    samples = np.array((0, 1, 3, 4, 2, 2, 3, 1, 1, 3, 3, 1, 2, 2, 2, 2)).\
        reshape((16, 1))
    data = dataset_wizard(samples.copy(), targets=range(16), chunks=[0] * 16)
    assert_equal(data.samples.mean(), 2.0)
    assert_equal(data.samples.std(), 1.0)
    data_samples = data.samples.copy()
    zscore(data, chunks_attr='chunks')

    # copy should stay intact
    assert_equal(data_samples.mean(), 2.0)
    assert_equal(data_samples.std(), 1.0)
    # we should be able to operate on ndarrays
    # But we can't change type inplace for an array, can't we?
    assert_raises(TypeError, zscore, data_samples, chunks_attr=None)
    # so lets do manually
    data_samples = data_samples.astype(float)
    zscore(data_samples, chunks_attr=None)
    assert_array_equal(data.samples, data_samples)

    # check z-scoring
    check = np.array([-2, -1, 1, 2, 0, 0, 1, -1, -1, 1, 1, -1, 0, 0, 0, 0],
                    dtype='float64').reshape(16, 1)
    assert_array_equal(data.samples, check)

    data = dataset_wizard(samples.copy(), targets=range(16), chunks=[0] * 16)
    zscore(data, chunks_attr=None)
    assert_array_equal(data.samples, check)

    # check z-scoring taking set of labels as a baseline
    data = dataset_wizard(samples.copy(),
                   targets=[0, 2, 2, 2, 1] + [2] * 11,
                   chunks=[0] * 16)
    zscore(data, param_est=('targets', [0, 1]))
    assert_array_equal(samples, data.samples + 1.0)

    # check that zscore modifies in-place; only guaranteed if no upcasting is
    # necessary
    samples = samples.astype('float')
    data = dataset_wizard(samples,
                   targets=[0, 2, 2, 2, 1] + [2] * 11,
                   chunks=[0] * 16)
    zscore(data, param_est=('targets', [0, 1]))
    assert_array_equal(samples, data.samples)

    # these might be duplicating code above -- but twice is better than nothing

    # dataset: mean=2, std=1
    raw = np.array((0, 1, 3, 4, 2, 2, 3, 1, 1, 3, 3, 1, 2, 2, 2, 2))
    # dataset: mean=12, std=1
    raw2 = np.array((0, 1, 3, 4, 2, 2, 3, 1, 1, 3, 3, 1, 2, 2, 2, 2)) + 10
    # zscore target
    check = [-2, -1, 1, 2, 0, 0, 1, -1, -1, 1, 1, -1, 0, 0, 0, 0]

    ds = dataset_wizard(raw.copy(), targets=range(16), chunks=[0] * 16)
    pristine = dataset_wizard(raw.copy(), targets=range(16), chunks=[0] * 16)

    zm = ZScoreMapper()
    # should do global zscore by default
    zm.train(ds)                        # train
    assert_array_almost_equal(zm.forward(ds), np.transpose([check]))
    # should not modify the source
    assert_array_equal(pristine, ds)

    # if we tell it a different mean it should obey the order
    zm = ZScoreMapper(params=(3,1))
    zm.train(ds)
    assert_array_almost_equal(zm.forward(ds), np.transpose([check]) - 1 )
    assert_array_equal(pristine, ds)

    # let's look at chunk-wise z-scoring
    ds = dataset_wizard(np.hstack((raw.copy(), raw2.copy())),
                        targets=range(32),
                        chunks=[0] * 16 + [1] * 16)
    # by default chunk-wise
    zm = ZScoreMapper()
    zm.train(ds)                        # train
    assert_array_almost_equal(zm.forward(ds), np.transpose([check + check]))
    # we should be able to do that same manually
    zm = ZScoreMapper(params={0: (2,1), 1: (12,1)})
    zm.train(ds)                        # train
    assert_array_almost_equal(zm.forward(ds), np.transpose([check + check]))

    # And just a smoke test for warnings reporting whenever # of
    # samples per chunk is low.
    # on 1 sample per chunk
    zds1 = ZScoreMapper(chunks_attr='chunks', auto_train=True)(
        ds[[0, -1]])
    ok_(np.all(zds1.samples == 0))   # they all should be 0
    # on 2 samples per chunk
    zds2 = ZScoreMapper(chunks_attr='chunks', auto_train=True)(
        ds[[0, 1, -10, -1]])
    assert_array_equal(np.unique(zds2.samples), [-1., 1]) # they all should be -1 or 1
    # on 3 samples per chunk -- different warning
    ZScoreMapper(chunks_attr='chunks', auto_train=True)(
        ds[[0, 1, 2, -3, -2, -1]])

    # test if std provided as a list not as an array is handled
    # properly -- should zscore all features (not just first/none
    # as it was before)
    ds = dataset_wizard(np.arange(32).reshape((8,-1)),
                        targets=range(8), chunks=[0] * 8)
    means = [0, 1, -10, 10]
    std0 = np.std(ds[:, 0])             # std deviation of first one
    stds = [std0, 10, .1, 1]

    zm = ZScoreMapper(params=(means, stds),
                      auto_train=True)
    dsz = zm(ds)

    assert_array_almost_equal((np.mean(ds, axis=0) - np.asanyarray(means))/np.array(stds),
                              np.mean(dsz, axis=0))

    assert_array_almost_equal(np.std(ds, axis=0)/np.array(stds),
                              np.std(dsz, axis=0))

def test_zscore_withoutchunks():
    # just a smoke test to see if all issues of
    # https://github.com/PyMVPA/PyMVPA/issues/26
    # are fixed
    from mvpa2.datasets import Dataset
    ds = Dataset(np.arange(32).reshape((8,-1)), sa=dict(targets=range(8)))
    zscore(ds, chunks_attr=None)
    assert(np.any(ds.samples != np.arange(32).reshape((8,-1))))
    ds_summary = ds.summary()
    assert(ds_summary is not None)