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

__all__ = [ 'OpenFMRIDataset' ]

import os
from os.path import join as _opj
import numpy as np
from mvpa2.datasets import vstack
from mvpa2.base import warning

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

def _id2int(id_, strip=None):
    if strip is None and isinstance(id_, basestring):
        for s in ('sub', 'task', 'model', 'run', 'cond'):
            if id_.startswith(s):
                id_ = id_[len(s):]
                break
    else:
        id_ = id_[strip:]
    try:
        id_ = int(id_)
    except:
        pass
    return id_


def _subdirs2ids(path, prefix, **kwargs):
    ids = []
    if not os.path.exists(path):
        return ids
    for item in os.listdir(path):
        if item.startswith(prefix) \
           and os.path.isdir(_opj(path, item)):
            ids.append(_id2int(item, **kwargs))
    return sorted(ids)


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
        return _subdirs2ids(self._basedir, 'sub')

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
                key = _id2int(key)
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
        task_prefix = _prefix('task', task)
        return _subdirs2ids(_opj(self._basedir, _sub2id(subj), 'BOLD'),
                            '%s_' % (task_prefix,),
                            strip=len(task_prefix) + 4)

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

    def get_bold_run_dataset(self, subj, task, run, flavor=None, add_sa=None,
            **kwargs):
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
        add_sa: str or tuple(str)
          Single or sequence of names of files in the respective BOLD
          directory containing additional samples attributes. At this time
          all formats supported by NumPy's loadtxt() are supported.
          The number of lines in such a file needs to match the number of
          BOLD volumes. Each column is converted into a separate dataset
          sample attribute. The file name with a column index suffix is used
          to determine the attribute name.
        **kwargs:
          All additional arguments are passed on to fmri_dataset()

        Returns
        -------
        Dataset
        """
        from mvpa2.datasets.mri import fmri_dataset

        bold_img = self.get_bold_run_image(subj, task, run, flavor=flavor)

        # load and mask data
        ds = fmri_dataset(bold_img, **kwargs)
        # inject sample attributes
        for name, var in (('subj', subj), ('task', task), ('run', run)):
            ds.sa[name] = np.repeat(var, len(ds))

        if add_sa is None:
            return ds

        if isinstance(add_sa , basestring):
            add_sa = (add_sa,)
        for sa in add_sa:
            fname = _opj(self._basedir, _sub2id(subj),
                         'BOLD', _taskrun(task, run),
                         sa)
            # TODO: come up with a fancy way of detecting what kind of thing
            # we are accessing -- in any case: first axis needs to match
            # nsamples
            attrs = np.loadtxt(fname)
            if len(attrs.shape) == 1:
                ds.sa[sa] = attrs
            else:
                for col in xrange(attrs.shape[1]):
                    ds.sa['%s_%i' % (sa, col)] = attrs[:, col]
        return ds

    def get_model_ids(self):
        """Returns a sorted list of integer IDs for all available models"""
        return _subdirs2ids(_opj(self._basedir, 'models'), 'model')

    def get_model_conditions(self, model):
        """Returns a description of all conditions for a given model

        Parameters
        ----------
        model : int
          Model identifier.

        Returns
        -------
        list(dict)
          A list of a model conditions is returned, where each item is a
          dictionary with keys ``id`` (numerical condition ID), ``task``
          (numerical task ID for the task containing this condition), and
          ``name`` (the literal condition name). This information is
          returned in a list (instead of a dictionary), because the openfmri
          specification of model conditions contains no unique condition
          identifier. Conditions are only uniquely described by the combination
          of task and condition ID.
        """
        def_fname = _opj(self._basedir, 'models', _model2id(model),
                         'condition_key.txt')
        def_data = np.recfromtxt(def_fname)
        conds = []
        # load model meta data
        for dd in def_data:
            cond = {}
            cond['task'] = _id2int(dd[0])
            cond['id'] = _id2int(dd[1])
            cond['name'] = dd[2]
            conds.append(cond)
        return conds

    def get_bold_run_model(self, model, subj, run):
        """Returns the stimulation design for a particular subject/task/run.

        Parameters
        ----------
        model : int
          Model identifier.
        subj : int
          Subject identifier.
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

        conditions = self.get_model_conditions(model)
        events = []
        ev_fields = ('onset', 'duration', 'intensity')
        # get onset info for specific subject/task/run combo
        for cond in conditions:
            task_id = cond['task']
            task_descr = self.get_task_descriptions()[task_id]
            stim_fname = _opj(self._basedir, _sub2id(subj), 'model',
                              _model2id(model), 'onsets',
                              _taskrun(task_id, run),
                              '%s.txt' % _cond2id(cond['id']))
            try:
                evdata = np.atleast_1d(
                       np.recfromtxt(stim_fname, names=ev_fields))
            except IOError:
                warning("onset definition file '%s' not found; no information "
                        "about condition '%s' for run %i"
                        % (stim_fname, cond['name'], run))
                continue
            for ev in evdata:
                evdict = dict(zip(ev_fields,
                                  [ev[field] for field in ev_fields]))
                evdict['task'] = task_descr
                evdict['condition'] = cond['name']
                evdict['run'] = run
                events.append(evdict)
        return events

    def get_model_bold_dataset(self, model_id, subj_id,
                          preprocfx=None, modelfx=None, stack=True,
                          flavor=None, mask=None, add_fa=None,
                          add_sa=None, **kwargs):
        """Build a PyMVPA dataset for a model defined in the OpenFMRI dataset

        Parameters
        ----------
        model_id : int
          Model ID.
        subj_id : int or str or list
          Integer, or string ID of the subject whose data shall be considered.
          Alternatively, a list of IDs can be given and data from all matching
          subjects will be loaded at once.
        preprocfx : callable or None
          If not None, this callable will be called with each run bold dataset
          as an argument before ``modelfx`` is executed. The callable must
          return a dataset.
        modelfx : callable or None
          This callable will be called with each run dataset and the respective
          event list for each run as arguments, In addition all additional
          **kwargs of this method will be passed on to this callable. The
          callable must return a dataset. If None, ``conditionlabeled_dataset``
          will be used as a default callable.
        stack : boolean
          Flag whether to stack all run datasets into a single dataset, or whether
          to return a list of datasets.
        flavor
          See get_bold_run_dataset() documentation
        mask
          See fmri_dataset() documentation.
        add_fa
          See fmri_dataset() documentation.
        add_sa
          See get_bold_run_dataset() documentation.

        Returns
        -------
        Dataset or list
          Depending on the ``stack`` argument either a single dataset or a list
          of datasets for all subject/task/run combinations relevant to the model
          will be returned. In the stacked case the dataset attributes of the
          returned dataset are taken from the first run dataset, and are assumed
          to be identical for all of them.
        """
        if modelfx is None:
            # loading a model dataset without actually considering the model
            # probably makes little sense, so at least create an attribute
            from mvpa2.datasets.eventrelated import conditionlabeled_dataset
            modelfx=conditionlabeled_dataset
        conds = self.get_model_conditions(model_id)
        # what tasks do we need to consider for this model
        tasks = np.unique([c['task'] for c in conds])
        if isinstance(subj_id, int) or isinstance(subj_id, basestring):
            subj_id = [subj_id]
        dss = []
        for sub in subj_id:
            for task in tasks:
                for run in self.get_bold_run_ids(sub, task):
                    events = self.get_bold_run_model(model_id, task, run)
                    if not len(events):
                        # nothing in this run for the given model
                        # it could be argued whether we'd still want this data loaded
                        # XXX maybe a flag?
                        continue
                    d = self.get_bold_run_dataset(sub, task, run=run, flavor=flavor,
                            chunks=run, mask=mask, add_fa=add_fa, add_sa=add_sa)
                    if not preprocfx is None:
                        d = preprocfx(d)
                    d = modelfx(d, events, **kwargs)
                    # if the modelfx doesn't leave 'chunk' information, we put
                    # something minimal in
                    if not 'chunks' in d.sa:
                        d.sa['chunks'] = [run] * len(d)
                    dss.append(d)
        if stack:
            dss = vstack(dss, a=0)
        return dss
