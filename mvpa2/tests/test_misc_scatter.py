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
from mvpa2.misc.plot.scatter import plot_scatter, plot_scatter_matrix, \
    plot_scatter_files
import numpy as np
from mock import patch


data2d = np.ones((2, 10, 10))
data3d = np.random.randn(3, 10, 10)

def test_plot_scatter():
    # smoke test
    fig = plot_scatter(data2d)

    assert_raises(ValueError, plot_scatter, data3d)

def test_plot_scatter_matrix():
    # smoke test
    fig = plot_scatter_matrix(data3d)

    # check it calls plot_scatter the right amount of times
    with patch('mvpa2.misc.plot.scatter.plot_scatter') as pscatter_mock:
        fig = plot_scatter_matrix(data3d)
        assert_equal(len(pscatter_mock.call_args_list), 6)

