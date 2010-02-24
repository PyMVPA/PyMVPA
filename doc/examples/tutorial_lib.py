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
import numpy as N

# later replace with
from mvpa.suite import *


def get_raw_haxby2001_data(path=os.path.join(pymvpa_datadbroot,
                                            'demo_blockfmri',
                                            'demo_blockfmri'),
                           roi='vt'):
    if roi is None:
        mask = None
    elif isinstance(roi, str):
        mask = os.path.join(path, 'mask_' + roi + '.nii.gz')
    elif isinstance(roi, int):
        nimg = NiftiImage(os.path.join(path, 'mask_hoc.nii.gz'))
        nimg_brain = NiftiImage(os.path.join(path, 'mask_brain.nii.gz'))
        tmpmask = nimg.data == roi
        if roi == 0:
            # trim it down to the lower anterior quadrant
            tmpmask[tmpmask.shape[0]/2:] = False
            tmpmask[:, :tmpmask.shape[1]/2] = False
            tmpmask[nimg_brain.data > 0] = False
        mask = NiftiImage(tmpmask.astype(int), nimg.header)
    else:
        raise ValueError("Got something as mask that I cannot handle.")
    attr = SampleAttributes(os.path.join(path, 'attributes.txt'))
    ds = fmri_dataset(samples=os.path.join(path, 'bold.nii.gz'),
                      targets=attr.targets, chunks=attr.chunks,
                      mask=mask)

    return ds


def get_haxby2001_data(path=None, roi='vt'):
    if path is None:
        ds = get_raw_haxby2001_data(roi=roi)
    else:
        ds = get_raw_haxby2001_data(path, roi=roi)

    # do chunkswise linear detrending on dataset
    poly_detrend(ds, polyord=1, chunks='chunks', inspace='time_coords')

    # mark the odd and even runs
    rnames = {0: 'even', 1: 'odd'}
    ds.sa['runtype'] = [rnames[c % 2] for c in ds.sa.chunks]

    # compute the mean sample per condition and odd vs. even runs
    # aka "constructive interference"
    ds = ds.get_mapped(mean_group_sample(['targets', 'runtype']))

    # zscore dataset relative to baseline ('rest') mean
    zscore(ds, param_est=('targets', ['rest']))

    # exclude the rest condition from the dataset
    ds = ds[ds.sa.targets != 'rest']

    return ds


def get_haxby2001_data_alternative(path=None, roi='vt'):
    if path is None:
        ds = get_raw_haxby2001_data(roi=roi)
    else:
        ds = get_raw_haxby2001_data(path, roi=roi)

    # do chunkswise linear detrending on dataset
    poly_detrend(ds, polyord=1, chunks='chunks', inspace='time_coords')

    # zscore dataset relative to baseline ('rest') mean
    zscore(ds, param_est=('targets', ['rest']))

    # exclude the rest condition from the dataset
    ds = ds[ds.sa.targets != 'rest']

    # mark the odd and even runs
    rnames = {0: 'even', 1: 'odd'}
    ds.sa['runtype'] = [rnames[c % 2] for c in ds.sa.chunks]

    # compute the mean sample per condition and odd vs. even runs
    # aka "constructive interference"
    ds = ds.get_mapped(mean_group_sample(['targets', 'runtype']))

    return ds


def get_haxby2001_clf():
    clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
    return clf


def load_tutorial_results(name,
                          path=os.path.join(pymvpa_datadbroot,
                                            'demo_blockfmri',
                                            'demo_blockfmri',
                                            'results')):
    return h5load(os.path.join(path, name + '.hdf5'))
