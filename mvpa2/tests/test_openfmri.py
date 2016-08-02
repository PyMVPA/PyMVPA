# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA's OpenFMRI data source adaptor"""

import numpy as np
from os.path import join as pathjoin

from nose.tools import assert_greater

from mvpa2.testing import *
from mvpa2.testing.sweep import sweepargs

from mvpa2 import pymvpa_dataroot
import mvpa2.datasets.sources.openfmri as ofm
from mvpa2.datasets.sources.native import load_tutorial_data, \
    load_example_fmri_dataset
from mvpa2.misc.io.base import SampleAttributes
from mvpa2.datasets.eventrelated import events2sample_attr, assign_conditionlabels


@sweepargs(
    fname=('something',
           'something.nii',
           'something.nii.gz',
           'something.hdr',
           'something.hdr.gz',
           'something.img',
           'something.img.gz'))
def test_helpers(fname):
    assert_equal('something', ofm._stripext(fname))


def test_openfmri_dataset():
    skip_if_no_external('nibabel')

    of = ofm.OpenFMRIDataset(pathjoin(pymvpa_dataroot, 'haxby2001'))
    assert_equal(of.get_model_descriptions(), {1: 'visual object categories'})
    sub_ids = of.get_subj_ids()
    assert_equal(sub_ids, [1, 'phantom'])
    assert_equal(of.get_scan_properties(), {'TR': '2.5'})
    tasks = of.get_task_descriptions()
    assert_equal(tasks, {1: 'object viewing'})
    task = tasks.keys()[0]
    run_ids = of.get_bold_run_ids(sub_ids[0], task)
    assert_equal(run_ids, range(1, 13))
    task_runs = of.get_task_bold_run_ids(task)
    assert_equal(task_runs, {1: range(1, 13)})

    # test access anatomy image
    assert_equal(of.get_anatomy_image(1, fname='lowres001.nii.gz').shape,
                 (6, 10, 10))
    # try to get an image that isn't there
    assert_raises(IOError, of.get_bold_run_image, 1, 1, 1)
    # defined model contrasts
    contrast_spec = of.get_model_contrasts(1)
    # one dict per task
    assert_equal(len(contrast_spec), 1)
    assert_true(1 in contrast_spec)
    # six defined contrasts
    assert_equal(len(contrast_spec[1]), 1)
    # check one
    assert_array_equal(contrast_spec[1]['face_v_house'],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

    orig_attrs = SampleAttributes(pathjoin(pymvpa_dataroot,
                                               'attributes_literal.txt'))
    for subj, runs in task_runs.iteritems():
        for run in runs:
            # load single run
            ds = of.get_bold_run_dataset(subj, task, run, flavor='1slice',
                                         mask=pathjoin(pymvpa_dataroot,
                                                           'mask.nii.gz'),
                                         add_sa='bold_moest.txt')
            # basic shape
            assert_equal(len(ds), 121)
            assert_equal(ds.nfeatures, 530)
            # functional mapper
            assert_equal(ds.O.shape, (121, 40, 20, 1))
            # additional attributes present
            moest = of.get_bold_run_motion_estimates(subj, task, run)
            for i in range(6):
                moest_attr = 'bold_moest.txt_%i' % (i,)
                assert_true(moest_attr in ds.sa)
                assert_array_equal(moest[:, i], ds.sa[moest_attr].value)

            # check conversion of model into sample attribute
            events = of.get_bold_run_model(subj, task, run)
            for i, ev in enumerate(events):
                # we only have one trial per condition in the demo dataset
                assert_equal(ev['conset_idx'], 0)
                # proper enumeration and events sorted by time
                assert_equal(ev['onset_idx'], i)
            onsets = [e['onset'] for e in events]
            sorted_onsets = sorted(onsets)
            assert_array_equal(sorted_onsets, onsets)

            targets = events2sample_attr(events,
                                         ds.sa.time_coords,
                                         noinfolabel='rest')
            assert_array_equal(
                orig_attrs['targets'][(run - 1) * 121: run * len(ds)], targets)
            assert_equal(ds.sa['subj'][0], subj)

            # check that we can get the same result from the model dataset
            # (make it exercise the preproc interface too)

            def preproc_img(img):
                return img
            modelds = of.get_model_bold_dataset(
                1, subj, flavor='1slice',
                preproc_img=preproc_img,
                modelfx=assign_conditionlabels,
                mask=pathjoin(pymvpa_dataroot,
                                  'mask.nii.gz'),
                add_sa='bold_moest.txt')
            modelds = modelds[modelds.sa.run == run]
            targets = np.array(targets, dtype='object')
            targets[targets == 'rest'] = None
            assert_array_equal(targets, modelds.sa.targets)
    # more basic access
    motion = of.get_task_bold_attributes(1, 'bold_moest.txt', np.loadtxt)
    assert_equal(len(motion), 12)  # one per run
    # one per subject, per volume, 6 estimates
    assert_equal([(len(m),) + m[1].shape for m in motion], [(1, 121, 6)] * 12)


def test_tutorialdata_loader_masking():
    skip_if_no_external('nibabel')

    ds_brain = load_tutorial_data(flavor='25mm')
    ds_nomask = load_tutorial_data(roi=None, flavor='25mm')
    assert_greater(ds_nomask.nfeatures, ds_brain.nfeatures)


@sweepargs(roi=('brain', 'gray', 'hoc', 'vt', 'white'))
def test_tutorialdata_rois(roi):
    skip_if_no_external('nibabel')

    # just checking that we have the files
    ds = load_tutorial_data(roi=roi, flavor='25mm')
    assert_equal(len(ds), 1452)


@sweepargs(roi=(1, 4, 6, 12, 17, 22, 28, 32, 33, 36, 42, 43, 44))
def test_hoc_rois(roi):
    skip_if_no_external('nibabel')

    # just checking which harvard-oxford rois we can rely on in the downsampled
    # data
    ds = load_tutorial_data(roi=roi, flavor='25mm')
    assert_equal(len(ds), 1452)


def test_roi_combo():
    skip_if_no_external('nibabel')

    ds1 = load_tutorial_data(roi=1, flavor='25mm')
    ds4 = load_tutorial_data(roi=4, flavor='25mm')
    ds_combo = load_tutorial_data(roi=(1, 4), flavor='25mm')
    assert_equal(ds1.nfeatures + ds4.nfeatures, ds_combo.nfeatures)


def test_corner_cases():
    skip_if_no_external('nibabel')

    assert_raises(ValueError, load_tutorial_data,
                  roi=range, flavor='25mm')


def test_example_data():
    skip_if_no_external('nibabel')

    # both expected flavor are present
    ds1 = load_example_fmri_dataset()
    ds25 = load_example_fmri_dataset(name='25mm', literal=True)
    assert_equal(len(ds1), len(ds25))
    # no 25mm dataset with numerical labels
    assert_raises(ValueError, load_example_fmri_dataset, name='25mm')
    # the 25mm example is the same as the coarse tutorial data
    ds25tut = load_tutorial_data(flavor='25mm')
    assert_array_equal(ds25.samples, ds25tut.samples)


def _test_datalad_openfmri_dataset(d):
    of = ofm.OpenFMRIDataset(d)

    # smoke tests that we can load the dataset's attributes etc
    #assert(of.get_model_descriptions())
    sub_ids = of.get_subj_ids()
    assert(sub_ids)
    assert('TR' in of.get_scan_properties())
    tasks = of.get_task_descriptions()
    assert(tasks)
    task = tasks.keys()[0]
    run_ids = of.get_bold_run_ids(sub_ids[0], task)
    assert(run_ids)
    task_runs = of.get_task_bold_run_ids(task)
    assert(task_runs)
    model_ids = of.get_model_ids()
    assert(model_ids)
    from datalad.auto import AutomagicIO
    with AutomagicIO():  # so necessary files get fetched if necessary
        # try loading first run for some task of the first subject
        data = of.get_model_bold_dataset(model_ids[0], sub_ids[0], task_runs.values()[0][:1])
    assert(data.shape)



def test_datalad_openfmri_datasets():
    skip_if_no_external('nibabel')
    skip_if_no_external('datalad')

    # Test on datalad crawled datasets
    # TODO: deal with paths to be configurable etc
    #  or eventually rely on datalad's API to deploy those
    TOPDIR = os.path.expanduser(pathjoin('~', 'datalad', 'crawl', 'openfmri'))
    dss = glob.glob(pathjoin(TOPDIR, 'ds*105'))
    if dss:
        for d in dss:  # for now just one
            yield _test_datalad_openfmri_dataset, d
    else:
        raise SkipTest("No datalad openfmri datasets were found under %s" % TOPDIR)