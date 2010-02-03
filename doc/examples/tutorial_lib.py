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
                                            'demo_blockfmri')):
    attr = SampleAttributes(os.path.join(path, 'attributes.txt'))
    ds = fmri_dataset(samples=os.path.join(path, 'bold.nii.gz'),
                      labels=attr.labels, chunks=attr.chunks,
                      mask=os.path.join(path, 'mask_vt.nii.gz'))

    return ds


def get_haxby2001_data(path=None):
    if path is None:
        ds = get_raw_haxby2001_data()
    else:
        ds = get_raw_haxby2001_data(path)

    # do chunkswise linear detrending on dataset
    poly_detrend(ds, polyord=1, chunks='chunks', inspace='time_coords')

    # mark the odd and even runs
    rnames = {0: 'even', 1: 'odd'}
    ds.sa['runtype'] = [rnames[c % 2] for c in ds.sa.chunks]

    # compute the mean sample per condition and odd vs. even runs
    # aka "constructive interference"
    ds = ds.get_mapped(mean_group_sample(['labels', 'runtype']))

    # zscore dataset relative to baseline ('rest') mean
    zscore(ds, param_est=('labels', ['rest']))

    # exclude the rest condition from the dataset
    ds = ds[ds.sa.labels != 'rest']

    return ds


def get_haxby2001_data_alternative(path=None):
    if path is None:
        ds = get_raw_haxby2001_data()
    else:
        ds = get_raw_haxby2001_data(path)

    # do chunkswise linear detrending on dataset
    poly_detrend(ds, polyord=1, chunks='chunks', inspace='time_coords')

    # zscore dataset relative to baseline ('rest') mean
    zscore(ds, param_est=('labels', ['rest']))

    # exclude the rest condition from the dataset
    ds = ds[ds.sa.labels != 'rest']

    # mark the odd and even runs
    rnames = {0: 'even', 1: 'odd'}
    ds.sa['runtype'] = [rnames[c % 2] for c in ds.sa.chunks]

    # compute the mean sample per condition and odd vs. even runs
    # aka "constructive interference"
    ds = ds.get_mapped(mean_group_sample(['labels', 'runtype']))

    return ds


def get_haxby2001_clf():
    clf = kNN(k=1, dfx=oneMinusCorrelation, voting='majority')
    return clf

