# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA misc.plot"""

from mvpa2.testing import *
skip_if_no_external('pylab')

import pylab as pl
from matplotlib.figure import Figure
from mvpa2.misc.plot.base import plot_dataset_chunks
import numpy as np

from glob import glob
from mock import patch
from os.path import join as pjoin

data2d = np.random.randn(2, 4, 4)
data3d = np.random.randn(3, 4, 4)

data2d_3d = np.random.randn(2, 4, 4, 4)
data2d_4d = np.random.randn(2, 4, 4, 4, 2)
data2d_5d = np.random.randn(2, 4, 4, 4, 2, 3)

from mvpa2.testing.datasets import datasets

@sweepargs(dsp=datasets.items())
def test_plot_dataset_chunks(dsp):
    dsname, ds = dsp
    if ds.targets.dtype.kind == 'f':
        return
    # smoke test for now
    if 'chunks' not in ds.sa:
        return  # nothing to plot in this one
    print dsname
    plot_dataset_chunks(ds[:, :2])  # could only plot two
    pl.close(pl.gcf())
    if ds.nfeatures > 2:
        assert_raises(ValueError, plot_dataset_chunks, ds)