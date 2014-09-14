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

from os.path import join as _opj
import nibabel as nb
import numpy as np
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.datasets import vstack

def _cond2id(val):
    return 'cond%.3i' % val

def _model2id(val):
    return 'model%.3i' % val

def _sub2id(val):
    return 'sub%.3i' % val

def _taskrun(task, run):
    return 'task%.3i_run%.3i' % (task, run)


class OpenFMRIDataset(object):
    """Handler for datasets following the openfmri.org layout specifications

    At present, this handler provides functions to access BOLD images of
    individual acquisition runs, build datasets from individual BOLD images,
    and load stimulation design specifications for individual runs.
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
        # load model meta data
        for dd in def_data:
            task = defs.setdefault(int(dd[0][4:]), {})
            cond = task.setdefault(int(dd[1][4:]), {})
            cond['name'] = dd[2]
        # get onset info for specific subject/task/run combo
        for task_id, task_dict in defs.iteritems():
            for cond_id, cond_dict in task_dict.iteritems():
                stim_fname = _opj(self._basedir, _sub2id(subj), 'model',
                                  _model2id(model), 'onsets',
                                  _taskrun(task_id, run),
                                  '%s.txt' % _cond2id(cond_id))
                cond_dict['stimulation'] = \
                        np.atleast_1d(
                            np.recfromtxt(
                                stim_fname,
                                names=('onset', 'duration', 'intensity')))
        return defs

def openfmri_model2target_attr(time_coords, model, noinfolabel=None,
                               onset_shift=0.0):
    """Build a target attribute array form an openfmri stimulation model

    Parameters
    ----------
    time_coords : array
      sample timing information array
      (typically taking from dataset.sa.time_coords)
    model : dict
      stimulation design specifications from
      OpenFMRIDataset.get_bold_run_model()
    noinfolabel : str
      condition label to assign to all samples for which no stimulation
      condition information is contained in the model. Example: 'rest'
    onset_shift : float
      All stimulation onset timestamps are shifted by the given amount
      before being transformed into discrete sample indices.
      Default: 0.0

    Returns
    -------
    list
      Sequence with literal conditions labels -- one item per element
      in the ``time_coords`` array.
    """
    sa = [None] * len(time_coords)
    for task_id, task_dict in model.iteritems():
        for cond_id, cond_dict in task_dict.iteritems():
            for stim in cond_dict['stimulation']:
                onset = stim['onset'] + onset_shift
                # first sample ending after stimulus onset
                onset_samp_idx = np.argwhere(time_coords[1:] > onset)[0,0]
                # deselect all volume starting before the end of the stimulation
                duration_mask = time_coords < (onset + stim['duration'])
                duration_mask[:onset_samp_idx] = False
                # assign all matching samples the condition ID
                for samp_idx in np.argwhere(duration_mask).T[0]:
                    sa[samp_idx] = cond_dict['name']
    if not noinfolabel is None:
        for i, a in enumerate(sa):
            if a is None:
                sa[i] = noinfolabel
    return sa

# XXX load model info for HRF model
