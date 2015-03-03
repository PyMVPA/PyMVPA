# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for algorithms result benchmarks ..."""

from mvpa2.algorithms.benchmarks.hyperalignment import *
from mvpa2.algorithms.benchmarks.hyperalignment import _zero_out_offdiag

from mvpa2.testing import *

def test_zero_out_offdiag():
    a = np.random.normal(size=(100, 100))
    for ws in (0, 1, 2, 88, 99, 101):
        assert(np.all(zero_out_offdiag(a, ws) == _zero_out_offdiag(a, ws)))

def test_timesegments_classification():
    print "MUST BE A SUCCESS!"
    # TODO: RF our construction of fake datasets for testing hyperalignment
    # so we could reuse it here and test classification performance
    timesegments_classification # DO ME to TEST ME!

