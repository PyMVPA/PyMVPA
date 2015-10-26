# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for algorithms result benchmarks ..."""

from mvpa2.algorithms.hyperalignment import Hyperalignment

from mvpa2.misc.data_generators import random_affine_transformation

from mvpa2.algorithms.benchmarks.hyperalignment import *
from mvpa2.algorithms.benchmarks.hyperalignment import _get_nonoverlapping_startpoints

from mvpa2.mappers.base import IdentityMapper

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets

def _wipe_out_offdiag(a, window_size, value=np.inf):
    for i in xrange(len(a)):
        for j in xrange(len(a)):
            if abs(i - j) < window_size and i != j:
                a[i, j] = value
    return a

def test_zero_out_offdiag():
    a = np.random.normal(size=(100, 100))
    for ws in (0, 1, 2, 88, 99, 101):
        assert(np.all(wipe_out_offdiag(a, ws) == _wipe_out_offdiag(a, ws)))

def test_timesegments_classification():
    # TODO: RF our construction of fake datasets for testing hyperalignment
    # so we could reuse it here and test classification performance
    ds_orig = datasets['uni4large']
    n = 3
    dss = [ds_orig.copy(deep=True) for i in xrange(n)]

    def nohyper(dss):
        return [IdentityMapper() for ds in dss]

    # clean case, assume "nohyper" which would be by default
    errors = timesegments_classification(dss)
    for ds in dss:
        # must not add any attribute, such as subjects
        assert('subjects' not in ds.sa)
    assert_array_equal(errors, 0)

    # very noisy case -- we must not be able to classify anything reasonably
    dss_noisy = [ds.copy() for ds in dss]
    for ds in dss_noisy:
        ds.samples = np.random.normal(size=ds.samples.shape)
    errors_nonoverlapping = timesegments_classification(dss_noisy, nohyper,
                                                        overlapping_windows=False)
    assert(np.all(errors_nonoverlapping <= 1.))
    assert(np.all(0.75 <= errors_nonoverlapping))

    errors_overlapping = timesegments_classification(dss_noisy, nohyper)
    # nononverlapping error should be less for random result
    assert_array_lequal(np.mean(errors_nonoverlapping), np.mean(errors_overlapping))

    # now the ultimate test with real hyperalignment on when we don't need much
    # of it anyways

    #import pdb; pdb.set_trace()
    dss_rotated = [random_affine_transformation(ds_orig, scale_fac=100, shift_fac=10)
                   for _ in dss]
    errors_hyper = timesegments_classification(dss_rotated, Hyperalignment())
    # Hyperalignment must not screw up and rotated and classify perfectly
    # since we didn't add any noise whatsoever
    assert_array_equal(errors, 0)

def test_get_nonoverlapping_startpoints():
    assert_equal(_get_nonoverlapping_startpoints(2, 1), [0, 1])
    assert_equal(_get_nonoverlapping_startpoints(2, 2), [0])
    assert_equal(_get_nonoverlapping_startpoints(4, 2), [0, 2])
    assert_equal(_get_nonoverlapping_startpoints(4, 3), [0])
    assert_equal(_get_nonoverlapping_startpoints(5, 3), [0])
    assert_equal(_get_nonoverlapping_startpoints(6, 3), [0, 3])

