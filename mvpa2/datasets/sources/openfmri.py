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

__all__ = [ 'OpenFMRIDataset']

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


def _get_description_dict(path, xfm_key=None):
    props = {}
    if os.path.exists(path):
        for line in open(path, 'r'):
            key = line.split()[0]
            value = line[len(key):].strip()
            if not xfm_key is None:
                key = xfm_key(key)
            props[key] = value
    return props


def _subdirs2ids(path, prefix, **kwargs):
    ids = []
    if not os.path.exists(path):
        return ids
    for item in os.listdir(path):
        if item.startswith(prefix) and os.path.isdir(_opj(path, item)):
                ids.append(_id2int(item, **kwargs))
    return sorted(ids)


def _stripext(path):
    for ext in ('.nii', '.nii.gz', '.hdr', '.hdr.gz', '.img', '.img.gz'):
        if path.endswith(ext):
            return path[:-len(ext)]
    return path


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
        self.basedir = os.path.expanduser(os.path.expandvars(basedir))

    def get_subj_ids(self):
        """Return a (sorted) list of IDs for all subjects in the dataset

        Standard numerical subject IDs a returned as integer values. All other
        types of IDs are returned as strings with the 'sub' prefix stripped.
        """
        return _subdirs2ids(self.basedir, 'sub')

    def get_scan_properties(self):
        """Return a dictionary with the scan properties listed in scan_key.txt
        """
        fname = _opj(self.basedir, 'scan_key.txt')
        return _get_description_dict(fname)

    def get_task_descriptions(self):
        """Return a dictionary with the tasks defined in the dataset

        Dictionary keys are integer task IDs, values are task description
        strings.
        """
        fname = _opj(self.basedir, 'task_key.txt')
        return _get_description_dict(fname, xfm_key=_id2int)

    def get_model_descriptions(self):
        """Return a dictionary with the models described in the dataset

        Dictionary keys are integer model IDs, values are description strings.

        Note that the return dictionary is not necessarily comprehensive. It
        only reflects the models described in ``model_key.txt``. If a dataset
        is inconsistently described, ``get_model_ids()`` actually may discover
        more or less models in comparison to the avauilable model descriptions.
        """
        fname = _opj(self.basedir, 'model_key.txt')
        return _get_description_dict(fname, xfm_key=_id2int)

    def get_bold_run_ids(self, subj, task):
        """Return (sorted) list of run IDs for a given subject and task

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
        return _subdirs2ids(_opj(self.basedir, _sub2id(subj), 'BOLD'),
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

    def _load_data(self, path, loadfx):
        # little helper to access stuff in datasets
        path = _opj(self.basedir, *path)
        return loadfx(path)

    def _load_subj_data(self, subj, path, loadfx):
        # little helper to access stuff in subjs of datasets
        path = [_sub2id(subj)] + path
        return self._load_data(path, loadfx)

    def _load_bold_task_run_data(self, subj, task, run, path, loadfx):
        # little helper for BOLD and associated data
        return self._load_subj_data(
            subj, ['BOLD', _taskrun(task, run)] + path, loadfx)

    def _load_model_task_run_onsets(self, subj, model, task, run, cond):
        # little helper for BOLD and associated data
        ev_fields = ('onset', 'duration', 'intensity')

        def _load_hlpr(fname):
            return np.recfromtxt(fname, names=ev_fields)

        return self._load_subj_data(
            subj,
            ['model', _model2id(model), 'onsets',
             _taskrun(task, run), '%s.txt' % _cond2id(cond)],
            _load_hlpr)

    def get_bold_run_image(self, subj, task, run, flavor=None):
        """Return a NiBabel image instance for the BOLD data of a
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

        Return
        -------
        NiBabel Nifti1Image
        """
        import nibabel as nb

        if flavor is None:
            flavor = ''
        else:
            flavor = '_' + flavor
        fname = 'bold%s.nii.gz' % flavor
        return self._load_bold_task_run_data(subj, task, run, [fname], nb.load)

    def get_bold_run_motion_estimates(self, subj, task, run,
                                      fname='bold_moest.txt'):
        """Return the volume-wise motion estimates for a particular BOLD run

        Parameters
        ----------
        subj : int
          Subject identifier.
        task : int
          Task ID (see task_key.txt)
        run : int
          Run ID.
        fname : str
          Filename.

        Returns
        -------
        array
          Array of floats -- one row per fMRI volume, 6 columns (typically,
          the first three are translation X, Y, Z in mm and the last three
          rotation in deg)
        """
        return self._load_bold_task_run_data(
            subj, task, run, [fname], np.loadtxt)

    def get_task_bold_attributes(self, task, fname, loadfx, exclude_subjs=None):
        """Return data attributes for all BOLD data from a specific task.

        This function can load arbitrary data from the directories where the
        relevant BOLD image files are stored. Data sources are described by
        specifying the file name containing the data in the BOLD directory,
        and by providing a function that returns the file content in array
        form. Optionally, data from specific subjects can be skipped.

        For example, this function can be used to access motion estimates.

        Parameters
        ----------
        task : int
          Task ID (see task_key.txt)
        fname : str
          Filename.
        loadfx : functor
          Function that can open the relevant files and return their content
          as an array. This function is called with the name of the data file
          as its only argument.
        exclude_subjs : list or None
          Optional list of subject IDs whose data shall be skipped.

        Returns
        -------
        list(array)
          A list of arrays, one for each BOLD run. Each array is
          (subjects x volumes x features).
        """
        if exclude_subjs is None:
            exclude_subjs = []
        # runs per task per subj
        tbri = self.get_task_bold_run_ids(task)
        nruns = max([max(tbri[s]) for s in tbri if not s in exclude_subjs])
        # structure to hold all data
        data = [None] * nruns

        # over all possible run ids
        for run in xrange(nruns):
            # for all actual subjects
            # TODO add subject filter
            for subj in sorted(tbri.keys()):
                try:
                    # run + 1 because openfmri is one-based
                    d = self._load_bold_task_run_data(subj, task, run + 1,
                                                      [fname], loadfx)
                    if data[run] is None:
                        data[run] = [d]
                    else:
                        data[run].append(d)
                except IOError:
                    # no data
                    pass
            # deal with missing values
            max_vol = max([len(d) for d in data[run]])
            for i, d in enumerate(data[run]):
                if len(d) == max_vol:
                    continue
                fixed_run = np.empty((max_vol, 6), dtype=np.float)
                fixed_run[:] = np.nan
                if len(d):
                    fixed_run[:len(d)] = d
                data[run][i] = fixed_run

        return [np.array(d) for d in data]

    def get_bold_run_dataset(self, subj, task, run, flavor=None,
                             preproc_img=None, add_sa=None, **kwargs):
        """Return a dataset instance for the BOLD data of a particular
        subject/task/run combination.

        This method support the same functionality as fmri_dataset(), while
        wrapping get_bold_run_image() to access the input fMRI data. Additional
        attributes, such as subject ID, task ID, and run ID are automatically
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
          BOLD data flavor to access (see dataset description). If ``flavor``
          corresponds to an existing file in the respective task/run directory,
          it is assumed to be a stored dataset in HDF5 format and loaded via
          ``h5load()`` -- otherwise datasets are constructed from NIfTI images.
        preproc_img : callable or None
          If not None, this callable will be called with the loaded source BOLD
          image instance as an argument before fmri_dataset() is executed.
          The callable must return an image instance.
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

        # check whether flavor corresponds to a particular file
        if not flavor is None:
            path = _opj(self.basedir, _sub2id(subj),
                        'BOLD', _taskrun(task, run), flavor)
        if not flavor is None and os.path.exists(path):
            from mvpa2.base.hdf5 import h5load
            ds = h5load(path)
        else:
            bold_img = self.get_bold_run_image(subj, task, run, flavor=flavor)
            if not preproc_img is None:
                bold_img = preproc_img(bold_img)
            # load (and mask) data
            ds = fmri_dataset(bold_img, **kwargs)

        # inject sample attributes
        for name, var in (('subj', subj), ('task', task), ('run', run)):
            ds.sa[name] = np.repeat(var, len(ds))

        if add_sa is None:
            return ds

        if isinstance(add_sa, basestring):
            add_sa = (add_sa,)
        for sa in add_sa:
            # TODO: come up with a fancy way of detecting what kind of thing
            # we are accessing -- in any case: first axis needs to match
            # nsamples
            attrs = self._load_bold_task_run_data(
                subj, task, run, [sa], np.loadtxt)
            if len(attrs.shape) == 1:
                ds.sa[sa] = attrs
            else:
                for col in xrange(attrs.shape[1]):
                    ds.sa['%s_%i' % (sa, col)] = attrs[:, col]
        return ds

    def get_model_ids(self):
        """Return a sorted list of integer IDs for all available models"""
        return _subdirs2ids(_opj(self.basedir, 'models'), 'model')

    def get_model_conditions(self, model):
        """Return a description of all conditions for a given model

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
        def_data = self._load_data(
            ['models', _model2id(model), 'condition_key.txt'],
            open)
        conds = []
        # load model meta data
        for dd in def_data:
            if not dd.strip():
                # ignore empty lines
                continue
            dd = dd.split()
            cond = {}
            cond['task'] = _id2int(dd[0])
            cond['id'] = _id2int(dd[1])
            cond['name'] = ' '.join(dd[2:])
            conds.append(cond)
        return conds

    def get_model_contrasts(self, model):
        """Return a defined contrasts for a model

        Parameters
        ----------
        model : int
          Model identifier.

        Returns
        -------
        dict(dict)
          A dictionary is returned, where each key is a (numerical) task ID
          and each value is a dictionary with contrast labels (str) as keys and
          contrast vectors as values.
        """
        from collections import OrderedDict
        props = {}
        try:
            def_data = self._load_data(
                ['models', _model2id(model), 'task_contrasts.txt'],
                open)
        except IOError:
            return props

        for line in def_data:
            line = line.split()
            task_id = _id2int(line[0])
            task = props.get(task_id, OrderedDict())
            task[line[1]] = np.array(line[2:], dtype=float)
            props[task_id] = task
        return props

    def get_bold_run_model(self, model, subj, run):
        """Return the stimulation design for a particular subject/task/run.

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
        list
          One item per event in the run. All items are dictionaries with the
          following keys: 'condition', 'onset', 'duration', 'intensity',
          'run', 'task', 'trial_idx', 'ctrial_idx', where the first is a
          literal label, the last four are integer IDs, and the rest are
          typically floating point values. 'onset_idx' is the index of the
          event specification sorted by time across the entire run (typically
          corresponding to a trial index), 'conset_idx' is analog but contains
          the onset index per condition, i.e. the nth trial of the respective
          condition in a run.
        """

        conditions = self.get_model_conditions(model)
        events = []
        ev_fields = ('onset', 'duration', 'intensity')

        # get onset info for specific subject/task/run combo
        for cond in conditions:
            task_id = cond['task']
            try:
                evdata = np.atleast_1d(
                    self._load_model_task_run_onsets(
                        subj, model, task_id, run, cond['id']))
            except IOError:
                warning("onset definition file not found; no information "
                        "about condition '%s' for run %i"
                        % (cond['name'], run))
                continue
            for i, ev in enumerate(evdata):
                evdict = dict(zip(ev_fields,
                                  [ev[field] for field in ev_fields]))
                evdict['task'] = task_id
                evdict['condition'] = cond['name']
                evdict['run'] = run
                evdict['conset_idx'] = i
                events.append(evdict)
        events = sorted(events, key=lambda x: x['onset'])
        for i, ev in enumerate(events):
            ev['onset_idx'] = i
        return events

    def get_model_bold_dataset(self, model_id, subj_id, preproc_img=None,
                               preproc_ds=None, modelfx=None, stack=True,
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
        preproc_img : callable or None
          See get_bold_run_dataset() documentation
        preproc_ds : callable or None
          If not None, this callable will be called with each run bold dataset
          as an argument before ``modelfx`` is executed. The callable must
          return a dataset.
        modelfx : callable or None
          This callable will be called with each run dataset and the respective
          event list for each run as arguments, In addition all additional
          **kwargs of this method will be passed on to this callable. The
          callable must return a dataset. If None, ``assign_conditionlabels``
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
            from mvpa2.datasets.eventrelated import assign_conditionlabels
            modelfx = assign_conditionlabels
        conds = self.get_model_conditions(model_id)
        # what tasks do we need to consider for this model
        tasks = np.unique([c['task'] for c in conds])
        if isinstance(subj_id, int) or isinstance(subj_id, basestring):
            subj_id = [subj_id]
        dss = []
        for sub in subj_id:
            # we need to loop over tasks first in order to be able to determine
            # what runs exists: that means we have to load the model info
            # repeatedly
            for task in tasks:
                for i, run in enumerate(self.get_bold_run_ids(sub, task)):
                    events = self.get_bold_run_model(model_id, sub, run)
                    # at this point our events should only contain those
                    # matching the current task. If not, this model violates
                    # the implicit assumption that one condition (label) can
                    # only be present in a single task. The current OpenFMRI
                    # spec does not allow for a more complex setup. I think
                    # this is worth a runtime check
                    check_events = [ev for ev in events if ev['task'] == task]
                    if not len(check_events) == len(events):
                        warning(
                            "not all event specifications match the expected "
                            "task ID -- something is wrong -- check that each "
                            "model condition label is only associated with a "
                            "single task")

                    if not len(events):
                        # nothing in this run for the given model
                        # it could be argued whether we'd still want this data loaded
                        # XXX maybe a flag?
                        continue
                    d = self.get_bold_run_dataset(
                        sub, task, run=run, flavor=flavor,
                        preproc_img=preproc_img, chunks=i, mask=mask,
                        add_fa=add_fa, add_sa=add_sa)
                    if not preproc_ds is None:
                        d = preproc_ds(d)
                    d = modelfx(
                        d, events, **dict([(k, v) for k, v in kwargs.iteritems()
                                          if not k in ('preproc_img', 'preproc_ds',
                                                       'modelfx', 'stack', 'flavor',
                                                       'mask', 'add_fa', 'add_sa')]))
                    # if the modelfx doesn't leave 'chunk' information, we put
                    # something minimal in
                    for attr, info in (('chunks', i), ('run', run), ('subj', sub)):
                        if not attr in d.sa:
                            d.sa[attr] = [info] * len(d)
                    dss.append(d)
        if stack:
            dss = vstack(dss, a=0)
        return dss

    def get_anatomy_image(self, subj, path=None, fname='highres001.nii.gz'):
        """Return a NiBabel image instance for a structural image of a subject.

        Parameters
        ----------
        subj : int
          Subject identifier.
        path : list or None
          Path to the structural file within the anatomy/ tree.
        fname : str
          Access a particular anatomy data flavor via its filename (see dataset
          description). Defaults to the first T1-weighted image.

        Returns
        -------
        NiBabel Nifti1Image
        """
        import nibabel as nb

        if path is None:
            path = []
        return self._load_subj_data(
            subj, ['anatomy'] + path + [fname], nb.load)
