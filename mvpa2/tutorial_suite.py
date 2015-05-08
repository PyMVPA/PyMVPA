#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import os
import numpy as np

# later replace with
from mvpa2.suite import *

tutorial_data_path = mvpa2.cfg.get('location', 'tutorial data', default=os.path.curdir)

def get_raw_haxby2001_data(path=tutorial_data_path, roi='vt'):
    if roi is 0:
        # this means something special in the searchlight tutorial
        maskpath = os.path.join(path, 'haxby2001', 'sub001', 'masks', 'orig')
        nimg = nb.load(os.path.join(maskpath, 'hoc.nii.gz'))
        nimg_brain = nb.load(os.path.join(maskpath, 'brain.nii.gz'))
        tmpmask = nimg.get_data() == roi
        # trim it down to the lower anterior quadrant
        tmpmask[:, :, tmpmask.shape[-1]/2:] = False
        tmpmask[:, :tmpmask.shape[1]/2] = False
        tmpmask[nimg_brain.get_data() > 0] = False
        mask = nb.Nifti1Image(tmpmask.astype(int), None, nimg.get_header())
        return load_tutorial_data(path=path, roi=mask)
    else:
        return load_tutorial_data(path=path, roi=roi)


def get_haxby2001_data(path=None, roi='vt'):
    if path is None:
        ds = get_raw_haxby2001_data(roi=roi)
    else:
        ds = get_raw_haxby2001_data(path, roi=roi)

    # do chunkswise linear detrending on dataset
    poly_detrend(ds, polyord=1, chunks_attr='chunks', space='time_coords')

    # mark the odd and even runs
    rnames = {0: 'even', 1: 'odd'}
    ds.sa['runtype'] = [rnames[c % 2] for c in ds.sa.chunks]

    # compute the mean sample per condition and odd vs. even runs
    # aka "constructive interference"
    ds = ds.get_mapped(mean_group_sample(['targets', 'runtype']))

    # XXX suboptimal order: should be zscore->avg
    # but then: where is the difference between this and _alternative()?
    # zscore dataset relative to baseline ('rest') mean
    zscore(ds, param_est=('targets', ['rest']))

    # exclude the rest condition from the dataset
    ds = ds[ds.sa.targets != 'rest']

    return ds


def get_haxby2001_data_alternative(path=None, roi='vt', grp_avg=True):
    if path is None:
        ds = get_raw_haxby2001_data(roi=roi)
    else:
        ds = get_raw_haxby2001_data(path, roi=roi)

    # do chunkswise linear detrending on dataset
    poly_detrend(ds, polyord=1, chunks_attr='chunks', space='time_coords')

    # zscore dataset relative to baseline ('rest') mean
    zscore(ds, param_est=('targets', ['rest']))

    # exclude the rest condition from the dataset
    ds = ds[ds.sa.targets != 'rest']

    # mark the odd and even runs
    rnames = {0: 'even', 1: 'odd'}
    ds.sa['runtype'] = [rnames[c % 2] for c in ds.sa.chunks]

    if grp_avg:
        # compute the mean sample per condition and odd vs. even runs
        # aka "constructive interference"
        ds = ds.get_mapped(mean_group_sample(['targets', 'runtype']))

    return ds


def get_haxby2001_clf():
    clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
    return clf
