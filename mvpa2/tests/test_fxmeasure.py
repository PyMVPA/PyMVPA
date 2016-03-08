# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Fx Measures"""

import numpy as np

from mvpa2.base import externals

from mvpa2.datasets import dataset_wizard
from mvpa2.measures.fx import BinaryFxFeaturewiseMeasure
from mvpa2.measures.fx import targets_dcorrcoef
if externals.exists('statsmodels'):
    from mvpa2.measures.fx import targets_mutualinfo_kde

from mvpa2.testing import sweepargs
from mvpa2.testing.datasets import datasets as tdatasets
from mvpa2.testing import assert_array_almost_equal, assert_array_less, assert_equal, ok_

if __debug__:
    from mvpa2.base import debug

@sweepargs(ds=tdatasets.itervalues())
def test_BinaryFxFeatureMeasure(ds):
    if not isinstance(ds.samples, np.ndarray):
        return
    # some simple function
    f = lambda x, y: np.sum((x.T*y).T, axis=0)
    fx = BinaryFxFeaturewiseMeasure(f, uni=False, numeric=True)
    fx_uni = BinaryFxFeaturewiseMeasure(f, uni=True, numeric=True)
    out = fx(ds)
    out_uni = fx_uni(ds)
    assert(len(out) == 1)
    assert_array_almost_equal(out.samples, out_uni)
    assert_equal(out.fa, out_uni.fa)
    ok_(str(fx).startswith("<BinaryFxFeaturewiseMeasure: lambda x, y:"))

_nonlin_tests = [(dataset_wizard([0, 1-0.01, 0, 1],
                                 targets=['a', 'b', 'a', 'b']),
                                ([0.99], [1])),
                 (dataset_wizard([0, 1-0.01, 2, 0, 1, 2],
                                 targets=['a', 'b', 'c', 'a', 'b', 'c']),
                                ([0.99], [1])),
                 # verify that order of 'labels' doesn't matter to get the same correspondence
                 (dataset_wizard([1-0.01, 0, 1, 0],
                                 targets=['a', 'b', 'a', 'b']),
                                ([0.99], [1])),
                 # unfortunately with both normal kde based MI and dcorr
                 # we are not getting "ideal" results in case of "non-linear"
                 # but strict dependencies
                 (dataset_wizard([0, 1-0.01, 2, 0, 1, 2],
                                 targets=['a', 'c', 'b', 'a', 'c', 'b']),
                                ([0.8], [1])),
                 # 2nd feature should have no information above the targets
                 (dataset_wizard([[0, 1], [1-0.01, 0], [0, 0], [1, 1]],
                                 targets=['a', 'b', 'a', 'b']),
                                ([0.99, -1e-10], [1.+1e-10, 0.01])),
                 ]

_nonlin_tests_fx = [targets_dcorrcoef()]
if externals.exists('statsmodels'):
    # -0.01 because current implementation would run into a singular matrix
    # if y is exactly the x. TODO: report/fix
    _nonlin_tests_fx.append(targets_mutualinfo_kde())


@sweepargs(ds_mi=_nonlin_tests)
@sweepargs(fx=_nonlin_tests_fx)
def test_BinaryFxFeatureMeasure_mi(fx, ds_mi):
    ds, (mi_min, mi_max) = ds_mi
    res = fx(ds)
    assert_equal(res.shape, (1, ds.nfeatures))
    mi = res.samples[0]
    assert_array_less(mi_min, mi)
    assert_array_less(mi, mi_max)
    fx_str = str(fx)
    if not (__debug__ and 'ID_IN_REPR' in debug.active):
        assert_equal(fx_str, "<BinaryFxFeaturewiseMeasure: %s>" % fx.fx.__name__)

