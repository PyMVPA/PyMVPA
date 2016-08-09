# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA misc.plot.scatter"""

from mvpa2.testing import *
skip_if_no_external('pylab')

from mvpa2.misc.plot.scatter import plot_scatter, plot_scatter_matrix, \
    plot_scatter_files, _get_data, fill_nonfinites
import numpy as np

from glob import glob
from mock import patch
from os.path import join as pjoin

data2d = np.random.randn(2, 4, 4)
data3d = np.random.randn(3, 4, 4)

data2d_3d = np.random.randn(2, 4, 4, 4)
data2d_4d = np.random.randn(2, 4, 4, 4, 2)
data2d_5d = np.random.randn(2, 4, 4, 4, 2, 3)


def test_fill_nonfinites():
    a = np.array([np.nan, np.inf, 2])
    aa = a.copy()
    fill_nonfinites(a)
    assert_array_equal(a, [0, 0, 2])

    aaa = fill_nonfinites(aa, inplace=False)
    assert_array_equal(aaa, a)
    assert_false(np.array_equal(aa, aaa))


def test_plot_scatter():
    # smoke test with possible 2 sample datasets
    fig = plot_scatter(data2d)
    fig = plot_scatter(data2d_3d)
    fig = plot_scatter(data2d_4d)

    # smoke test with jitter
    fig = plot_scatter(data2d, x_jitter=0.1)
    fig = plot_scatter(data2d, y_jitter=0.1)
    fig = plot_scatter(data2d, x_jitter=0.1, y_jitter=0.1)

    # smoke test with mask
    mask = np.random.randint(0, 2, size=data2d.shape)
    fig = plot_scatter(data2d, mask=mask)
    fig = plot_scatter(data2d, mask=mask, masked_opacity=0.42)

    # smoke test with threshold
    fig = plot_scatter(data2d, thresholds=[0.2])
    fig = plot_scatter(data2d, thresholds=[0.2, 0.4])

    # smoke tests with stats
    fig = plot_scatter(data2d, include_stats=True)

    # test when it should fail
    assert_raises(ValueError, plot_scatter, data3d)
    assert_raises(ValueError, plot_scatter, data2d_5d)

def test_plot_scatter_matrix():
    # smoke test
    fig = plot_scatter_matrix(data3d)

    # check it calls plot_scatter the right amount of times
    with patch('mvpa2.misc.plot.scatter.plot_scatter') as pscatter_mock:
        fig = plot_scatter_matrix(data3d)
        assert_equal(len(pscatter_mock.call_args_list), 6)


def test_plot_scatter_files():
    fns = glob(pjoin(pymvpa_dataroot,
                     *('haxby2001/sub001/anatomy/lowres00*.nii.gz'.split('/'))))
    figs = plot_scatter_files(fns)


def test_get_data():
    fn = pjoin(pymvpa_dataroot,
               *('haxby2001/sub001/anatomy/lowres001.nii.gz'.split('/')))

    data = _get_data(fn)
    assert_true(data.ndim, 3)
