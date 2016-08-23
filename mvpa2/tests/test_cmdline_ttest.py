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
from mvpa2 import pymvpa_dataroot
skip_if_no_external('h5py')
from mvpa2.cmdline.cmd_ttest import run, guess_backend
from mvpa2.datasets import Dataset
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.base.hdf5 import h5save
import sys
from tempfile import mkdtemp
from shutil import rmtree

import numpy as np
from mvpa2.misc.stats import ttest_1samp
import scipy.stats as stats
from os.path import join as pjoin

if __debug__:
    from mvpa2.base import debug

def test_guess_backend():
    assert_equal('nifti', guess_backend('meh.nii.gz'))
    assert_equal('nifti', guess_backend('meh.nii'))
    assert_equal('hdf5', guess_backend('meh.hdf5'))
    assert_equal('hdf5', guess_backend('meh.h5'))
    assert_equal('nifti', guess_backend('meh.tar'))

# test that using a mask works
datafn = pjoin(pymvpa_dataroot,
               'haxby2001/sub001/BOLD/task001_run001/bold_25mm.nii.gz')
maskfn = pjoin(pymvpa_dataroot,
               'haxby2001/sub001/masks/25mm/brain.nii.gz')

tmpdir = mkdtemp()
# temporarly save the input nii to hdf5 so we can reuseit
data_ = fmri_dataset(datafn)
datafn_hdf5 = pjoin(tmpdir, 'datain.hdf5')
h5save(datafn_hdf5, data_)
outfn = [pjoin(tmpdir, 'output') + ext for ext in ['.nii.gz', '.nii',
                                                   '.hdf5', '.h5']]

@sweepargs(stat=['z', 'p', 't'])
@sweepargs(alternative=['two-sided', 'greater'])
@sweepargs(data=[datafn, datafn_hdf5])
@sweepargs(mask=['', maskfn])
@sweepargs(outfn=outfn)
def test_cmdline_args(stat, alternative, data, mask, outfn):
    """Just a big smoke test"""
    class FakeArg(object):
        def __init__(self):
            self.data = []
            self.chance_level = 0.
            self.alternative = 'two-sided'
            self.stat = 'z'
            self.mask = ''
            self.output = ''
            self.isample = 0

    args = FakeArg()
    args.chance_level = 0.
    args.alternative = alternative
    args.stat = stat
    args.mask = mask
    args.data = [data for _ in range(3)]
    args.output = outfn

    run(args)
