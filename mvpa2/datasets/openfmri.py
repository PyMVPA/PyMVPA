# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helpers to build PyMVPA dataset instances from openfmri.org dataset
"""

__docformat__ = 'restructuredtext'

import os
from os.path import join as _opj
import nibabel as nb
import numpy as np
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.datasets import vstack

def _prefix(prefix, val):
    if isinstance(val, int):
        return '%s%.3i' % (prefix, val)
    else:
        return '%s%s' % (prefix, val)

def _cond2id(val):
    return _prefix('cond', val)

def _model2id(val):
    return _prefix('model', val)

def _sub2id(val):
    return _prefix('sub', val)

def _taskrun(task, run):
    return '%s_%s' % (_prefix('task', task), _prefix('run', run))


class OpenFMRIDataset(object):
    """Handler for datasets following the openfmri.org layout specifications

    At present, this handler provides functions to query and access a number of
    dataset properties, BOLD images of individual acquisition runs, build
    datasets from individual BOLD images, and load stimulation design
    specifications for individual runs.
    """
    def __init__(self, basedir):
        """
        Parameters
        ----------
        basedir : path
          Path to the dataset (i.e. the directory with the 'sub*'
          subdirectories).
        """
        self._basedir = basedir

    def get_subj_ids(self):
        """Return a (sorted) list of IDs for all subjects in the dataset

        Standard numerical subject IDs a returned as integer values. All other
        types of IDs are returned as strings with the 'sub' prefix stripped.
        """
        ids = []
        for item in os.listdir(self._basedir):
            if item.startswith('sub') \
               and os.path.isdir(_opj(self._basedir, item)):
                id_ = item[3:]
                try:
                    id_ = int(id_)
                except:
                    pass
                ids.append(id_)
        return sorted(ids)

    def get_scan_properties(self):
        """Returns a dictionary with the scan properties listed in scan_key.txt
        """
        props = {}
        fname = _opj(self._basedir, 'scan_key.txt')
        if os.path.exists(fname):
            for line in open(fname, 'r'):
                key = line.split()[0]
                value = line[len(key):].strip()
                props[key] = value
        return props

    def get_task_descriptions(self):
        """Returns a dictionary with the tasks defined in the dataset

        Dictionary keys are integer task IDs, values are task description
        strings.
        """
        tasks = {}
        fname = _opj(self._basedir, 'task_key.txt')
        if os.path.exists(fname):
            for line in open(fname, 'r'):
                key = line.split()[0]
                value = line[len(key):].strip()
                if key.startswith('task'):
                    key = key[4:]
                key = int(key)
                tasks[key] = value
        return tasks

    def get_bold_run_ids(self, subj, task):
        """Returns (sorted) list of run IDs for a given subject and task

        Typically, run IDs are integer values, but string IDs are supported
        as well.

        Parameters
        ----------
        subj : int or str
          Subject ID
        task : int or str
          Run ID
        """
        ids = []
        task_prefix = _prefix('task', task)
        bold_dir = _opj(self._basedir, _sub2id(subj), 'BOLD')
        if not os.path.exists(bold_dir):
            return ids
        for item in os.listdir(bold_dir):
            if item.startswith('%s_' % (task_prefix,)) \
               and os.path.isdir(_opj(bold_dir, item)):
                id_ = item[len(task_prefix) + 4:]
                try:
                    id_ = int(id_)
                except:
                    pass
                ids.append(id_)
        return sorted(ids)

    def get_task_bold_run_ids(self, task):
        """Return a dictionary with run IDs by subjects for a given task

        Dictionary keys are subject IDs, values are lists of run IDs.
        """
        out = {}
        for sub in self.get_subj_ids():
            runs = self.get_bold_run_ids(sub, task)
            if len(runs):
                out[sub] = runs
        return out

    def get_bold_run_image(self, subj, task, run, flavor=None):
        """Returns a NiBabel image instance for the BOLD data of a 
        particular subject/task/run combination.

        Parameters
        ----------
        subj : int
          Subject identifier.
        task : int
          Task ID (see task_key.txt)
        run : int
          Run ID.
        flavor : None or str
          BOLD data flavor to access (see dataset description)

        Returns
        -------
        NiBabel Nifti1Image
        """
        import nibabel as nb

        if flavor is None:
            flavor = ''
        else:
            flavor = '_' + flavor
        fname = 'bold%s.nii.gz' % flavor
        fname = _opj(self._basedir, _sub2id(subj),
                     'BOLD', _taskrun(task, run),
                     fname)
        return nb.load(fname)

    def get_bold_run_dataset(self, subj, task, run, flavor=None, **kwargs):
        """Returns a dataset instance for the BOLD data of a particular
        subject/task/run combination.

        This method support the same functionality as fmri_dataset(), while
        wrapping get_bold_run_image() to access the input fMRI data. Additional
        attributes, such as subject ID, task ID. and run ID are automatically
        stored as dataset sample attributes.

        Parameters
        ----------
        subj : int
          Subject identifier.
        task : int
          Task ID (see task_key.txt)
        run : int
          Run ID.
        flavor : None or str
          BOLD data flavor to access (see dataset description)
        **kwargs:
          All additional arguments are passed on to fmri_dataset()

        Returns
        -------
        Dataset
        """

        bold_img = self.get_bold_run_image(subj, task, run, flavor=flavor)

        # load and mask data
        ds = fmri_dataset(bold_img, **kwargs)
        # inject sample attributes
        for name, var in (('subj', subj), ('task', task), ('run', run)):
            ds.sa[name] = np.repeat(var, len(ds))
        return ds

    def get_bold_run_model(self, model, subj, run):
        """Returns the stimulation design for a particular subject/task/run.

        Parameters
        ----------
        subj : int
          Subject identifier.
        task : int
          Task ID (see task_key.txt)
        run : int
          Run ID.

        Returns
        -------
        dict
          Nested dictionary for all tasks and conditions contained in a
          particular model. First-level keys are task IDs. Second-level keys
          are condition IDs. Second-level values are rec-arrays with fields
          'onset', 'duration', 'intensity'.
        """

        def_fname = _opj(self._basedir, 'models', _model2id(model),
                         'condition_key.txt')
        def_data = np.recfromtxt(def_fname)
        defs = {}
        events = []
        # load model meta data
        for dd in def_data:
            task = defs.setdefault(int(dd[0][4:]), {})
            cond = task.setdefault(int(dd[1][4:]), {})
            cond['name'] = dd[2]
        ev_fields = ('onset', 'duration', 'intensity')
        # get onset info for specific subject/task/run combo
        for task_id, task_dict in defs.iteritems():
            task_descr = self.get_task_descriptions()[task_id]
            for cond_id, cond_dict in task_dict.iteritems():
                stim_fname = _opj(self._basedir, _sub2id(subj), 'model',
                                  _model2id(model), 'onsets',
                                  _taskrun(task_id, run),
                                  '%s.txt' % _cond2id(cond_id))
                evdata = np.atleast_1d(
                           np.recfromtxt(stim_fname, names=ev_fields))
                for ev in evdata:
                    evdict = dict(zip(ev_fields,
                                      [ev[field] for field in ev_fields]))
                    evdict['task'] = task_descr
                    evdict['condition'] = cond_dict['name']
                    evdict['run'] = run
                    events.append(evdict)
        return events

# XXX load model info for HRF model
