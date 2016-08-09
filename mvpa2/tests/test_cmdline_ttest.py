# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA cmdline ttest"""

from mvpa2.testing import *
from mvpa2.cmdline.cmd_ttest import run, guess_backend
from mvpa2.datasets import Dataset

import numpy as np
from mvpa2.misc.stats import ttest_1samp
import scipy.stats as stats

if __debug__:
    from mvpa2.base import debug

def test_guess_backend():
    assert_equal('nifti', guess_backend('meh.nii.gz'))
    assert_equal('nifti', guess_backend('meh.nii'))
    assert_equal('hdf5', guess_backend('meh.hdf5'))
    assert_equal('hdf5', guess_backend('meh.h5'))
    assert_equal('nifti', guess_backend('meh.tar'))


def test_cmdline_ttest():
    skip_if_no_external('mock')
    import mock

    # compute true data
    data = np.random.randn(2, 20)
    t, p = ttest_1samp(np.asarray([data + 10, data + 20]),
                       popmean=0., axis=0, alternative='two-sided')
    z = stats.norm.isf(p/2)
    z = np.abs(z) * np.sign(t)

    # make nifti and hdf5 type files
    # niftis will return at least a 3D array
    class MockNifti(object):
        def __init__(self):
            self.counter = 0
            self.nifti = data.reshape((2, 4, 5))
            self.header = 'header'
        def get_data(self):
            self.counter += 1
            return self.nifti + self.counter*10


    # hdf5 will be a dataset
    hdf5 = Dataset(data)

    with mock.patch("nibabel.load") as mock_nibload, \
        mock.patch("mvpa2.base.hdf5.h5load") as mock_h5load, \
        mock.patch("nibabel.Nifti1Image") as mock_nifti1:

        # so everytime load it's called it passes a mocknifti
        mock_nibload.return_value = MockNifti()
        # and everytime h5load is called it returns my hdf5
        mock_h5load.return_value = hdf5

        mock_args = mock.Mock()
        mock_args.chance_level = 0.
        mock_args.alternative = 'two-sided'
        mock_args.stat = 'z'

        # filetype_in, filetype_out = nifti
        mock_args.data = ['data1.nii.gz', 'data2.nii.gz']
        mock_args.output = 'bzot.nii.gz'
        run(mock_args)
        assert_array_equal(z.reshape(2, 4, 5), mock_nifti1.call_args[0][0])




