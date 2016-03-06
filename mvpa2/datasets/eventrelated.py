# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Functions for event segmentation or modeling of dataset."""

__docformat__ = 'restructuredtext'

import copy
import numpy as np
from mvpa2.misc.support import Event, value2idx
from mvpa2.datasets import Dataset
from mvpa2.base.dataset import _expand_attribute
from mvpa2.mappers.fx import _uniquemerge2literal
from mvpa2.mappers.flatten import FlattenMapper
from mvpa2.mappers.boxcar import BoxcarMapper
from mvpa2.base import warning, externals

def assign_conditionlabels(ds, events, label_attr='targets',
                           time_attr='time_coords', **kwargs):
    """Convert events into a condition label attribute of a dataset

    This function is a convenience front-end to ``events2sample_attr()`` and supports
    the same arguments.

    Parameters
    ----------
    ds : dataset
      To be labeled dataset.
    events : list
      List of dictionaries with event definitions.
    label_attr : str
      Name of the sample attribute with the conditions labels in the output dataset.
      Note that any existing attribute with this name will be overwritten.
    time_attr : str
      Name of the sample attribute with the time stamps (time coordinate) for all data
      samples. This information will be used to match samples with conditions
    **kwargs
      All other arguments will be passed on to ``events2sample_attr()``

    Returns
    -------
    Dataset
      with condition label attribute
    """
    attr = events2sample_attr(events, ds.sa[time_attr].value, **kwargs)
    ds.sa[label_attr] = attr
    return ds

def events2sample_attr(events, time_coords, noinfolabel=None,
                       onset_shift=0.0, condition_attr='condition'):
    """Build a sample attribute array form an event list

    Parameters
    ----------
    events : list
      event specifications as consumed by eventrelated_dataset()
    time_coords : array
      sample timing information array
      (typically taking from dataset.sa.time_coords)
    noinfolabel : str
      condition label to assign to all samples for which no stimulation
      condition information is contained in the events. Example: 'rest'
    onset_shift : float
      All stimulation onset timestamps are shifted by the given amount
      before being transformed into discrete sample indices.
      Default: 0.0
    condition_attr : str
      Name of the key in the event dictionary whose value shall be used as
      as attribute value for the associated sample(s).

    Returns
    -------
    list
      Sequence with literal conditions labels -- one item per element
      in the ``time_coords`` array.
    """
    sa = [None] * len(time_coords)
    for ev in events:
        onset = ev['onset'] + onset_shift
        # first sample ending after stimulus onset
        onset_samp_idx = np.argwhere(time_coords[1:] > onset)[0,0]
        # deselect all volume starting before the end of the stimulation
        duration_mask = time_coords < (onset + ev['duration'])
        duration_mask[:onset_samp_idx] = False
        # assign all matching samples the condition ID
        for samp_idx in np.argwhere(duration_mask).T[0]:
            sa[samp_idx] = ev[condition_attr]
    if noinfolabel is not None:
        for i, a in enumerate(sa):
            if a is None:
                sa[i] = noinfolabel
    return sa


def find_events(**kwargs):
    """Detect changes in multiple synchronous sequences.

    Multiple sequence arguments are scanned for changes in the unique value
    combination at corresponding locations. Each change in the combination is
    taken as a new event onset.  The length of an event is determined by the
    number of identical consecutive combinations.

    Parameters
    ----------
    **kwargs : sequences
      Arbitrary number of sequences that shall be scanned.

    Returns
    -------
    list
      Detected events, where each event is a dictionary with the unique
      combination of values stored under their original name. In addition, the
      dictionary also contains the ``onset`` of the event (as index in the
      sequence), as well as the ``duration`` (as number of identical
      consecutive items).

    See Also
    --------
    eventrelated_dataset : event-related segmentation of a dataset

    Examples
    --------
    >>> seq1 = ['one', 'one', 'two', 'two']
    >>> seq2 = [1, 1, 1, 2]
    >>> events = find_events(targets=seq1, chunks=seq2)
    >>> for e in events:
    ...     print e
    {'chunks': 1, 'duration': 2, 'onset': 0, 'targets': 'one'}
    {'chunks': 1, 'duration': 1, 'onset': 2, 'targets': 'two'}
    {'chunks': 2, 'duration': 1, 'onset': 3, 'targets': 'two'}
    """
    def _build_event(onset, duration, combo):
        ev = Event(onset=onset, duration=duration, **combo)
        return ev

    events = []
    prev_onset = 0
    old_combo = None
    duration = 1
    # over all samples
    for r in xrange(len(kwargs.values()[0])):
        # current attribute combination
        combo = dict([(k, v[r]) for k, v in kwargs.iteritems()])

        # check if things changed
        if not combo == old_combo:
            # did we ever had an event
            if old_combo is not None:
                events.append(_build_event(prev_onset, duration, old_combo))
                # reset duration for next event
                duration = 1
                # store the current samples as onset for the next event
                prev_onset = r

            # update the reference combination
            old_combo = combo
        else:
            # current event is lasting
            duration += 1

    # push the last event in the pipeline
    if old_combo is not None:
        events.append(_build_event(prev_onset, duration, old_combo))

    return events

def _events2dict(events):
    evvars = {}
    for k in events[0]:
        try:
            evvars[k] = [e[k] for e in events]
        except KeyError:
            raise ValueError("Each event property must be present for all "
                             "events (could not find '%s')" % k)
    return evvars

def _evvars2ds(ds, evvars, eprefix):
    for a in evvars:
        if eprefix is not None and a in ds.sa:
            # if there is already a samples attribute like this, it got mapped
            # previously (e.g. by BoxcarMapper and is multi-dimensional).
            # We move it aside under new `eprefix` name
            ds.sa[eprefix + '_' + a] = ds.sa[a]
        ds.sa[a] = evvars[a]
    return ds


def extract_boxcar_event_samples(
        ds, events=None, time_attr=None, match='prev',
        event_offset=None, event_duration=None,
        eprefix='event', event_mapper=None):
    """Segment a dataset by extracting boxcar events

    (Multiple) consecutive samples are extracted for each event, and are either
    returned in a flattened shape, or subject to further processing.

    Events are specified as a list of dictionaries
    (see:class:`~mvpa2.misc.support.Event`) for a helper class. Each dictionary
    contains all relevant attributes to describe an event. This is at least the
    ``onset`` time of an event, but can also comprise of ``duration``,
    ``amplitude``, and arbitrary other attributes.

    Boxcar event model details
    --------------------------

    For each event all samples covering that particular event are used to form
    a corresponding sample. One sample for each event is returned. Event
    specification dictionaries must contain an ``onset`` attribute (as sample
    index in the input dataset), ``duration`` (as number of consecutive samples
    after the onset). Any number of additional attributes can be present in an
    event specification. Those attributes are included as sample attributes in
    the returned dataset.

    Alternatively, ``onset`` and ``duration`` may also be given in a
    non-discrete time specification. In this case a dataset attribute needs to
    be specified that contains time-stamps for each input data sample, and is
    used to convert times into discrete sample indices (see ``match``
    argument).

    A mapper instance can be provided (see ``event_mapper``) to implement
    futher processing of each event sample, for example in order to yield
    average samples.

    Parameters
    ----------
    ds : Dataset
      The samples of this input dataset have to be in whatever ascending order.
    events : list
      Each event definition has to specify ``onset`` and ``duration``. All
      other attributes will be passed on to the sample attributes collection of
      the returned dataset.
    time_attr : str or None
      Attribute with dataset sample time-stamps.
      If not None, the ``onset`` and ``duration`` specs
      from the event list will be converted using information from this sample
      attribute. Its values will be treated as in-the-same-unit and are used to
      determine corresponding samples from real-value onset and duration
      definitions.
      For HRF modeling this argument is mandatory.
    match : {'prev', 'next', 'closest'}
      Strategy used to match real-value onsets to sample
      indices. 'prev' chooses the closes preceding samples, 'next' the closest
      following sample and 'closest' to absolute closest sample.
    event_offset : None or float
      If not None, all event ``onset`` specifications will be offset by this
      value before boxcar modeling is performed.
    event_duration : None or float
      If not None, all event ``duration`` specifications will be set to this
      value before boxcar modeling is done.
    eprefix : str or None
      If not None, this prefix is used to name additional
      attributes generated by the underlying
      `~mvpa2.mappers.boxcar.BoxcarMapper`. If it is set to None, no additional
      attributes will be created.
    event_mapper : Mapper
      This mapper is used to forward-map the dataset containing the boxcar event
      samples. If None (default) a FlattenMapper is employed to convert
      multi-dimensional sample matrices into simple one-dimensional sample
      vectors. This option can be used to implement temporal compression, by
      e.g. averaging samples within an event boxcar using an FxMapper. Any
      mapper needs to keep the sample axis unchanged, i.e. number and order of
      samples remain the same.

    Returns
    -------
    Dataset
      One sample per each event definition that has been passed to the
      function. Additional event attributes are included as sample attributes.

    Examples
    --------
    The documentation also contains an :ref:`example script
    <example_eventrelated>` showing a spatio-temporal analysis of fMRI data
    that involves this function.

    >>> from mvpa2.datasets import Dataset
    >>> ds = Dataset(np.random.randn(10, 25))
    >>> events = [{'onset': 2, 'duration': 4},
    ...           {'onset': 4, 'duration': 4}]
    >>> eds = eventrelated_dataset(ds, events)
    >>> len(eds)
    2
    >>> eds.nfeatures == ds.nfeatures * 4
    True
    >>> 'mapper' in ds.a
    False
    >>> print eds.a.mapper
    <Chain: <Boxcar: bl=4>-<Flatten>>

    And now the same conversion, but with events specified as real time. This is
    on possible if the input dataset contains a sample attribute with the
    necessary information about the input samples.

    >>> ds.sa['record_time'] = np.linspace(0, 5, len(ds))
    >>> rt_events = [{'onset': 1.05, 'duration': 2.2},
    ...              {'onset': 2.3, 'duration': 2.12}]
    >>> rt_eds = eventrelated_dataset(ds, rt_events, time_attr='record_time',
    ...                               match='closest')
    >>> np.all(eds.samples == rt_eds.samples)
    True
    >>> # returned dataset e.g. has info from original samples
    >>> rt_eds.sa.record_time
    array([[ 1.11111111,  1.66666667,  2.22222222,  2.77777778],
           [ 2.22222222,  2.77777778,  3.33333333,  3.88888889]])
    """
    # relabel argument
    conv_strategy = {'prev': 'floor',
                     'next': 'ceil',
                     'closest': 'round'}[match]

    if not (event_offset is None and event_duration is None):
        descr_events = []
        for ev in events:
            # do not mess with the input data
            ev = copy.deepcopy(ev)
            if event_offset is not None:
                ev['onset'] += event_offset
            if event_duration is not None:
                ev['duration'] = event_duration
            descr_events.append(ev)
        events = descr_events

    if time_attr is not None:
        tvec = ds.sa[time_attr].value
        # we are asked to convert onset time into sample ids
        descr_events = []
        for ev in events:
            # do not mess with the input data
            ev = copy.deepcopy(ev)
            # best matching sample
            idx = value2idx(ev['onset'], tvec, conv_strategy)
            # store offset of sample time and real onset
            ev['orig_offset'] = ev['onset'] - tvec[idx]
            # rescue the real onset into a new attribute
            ev['orig_onset'] = ev['onset']
            ev['orig_duration'] = ev['duration']
            # figure out how many samples we need
            ev['duration'] = \
                    len(tvec[idx:][tvec[idx:] < ev['onset'] + ev['duration']])
            # new onset is sample index
            ev['onset'] = idx
            descr_events.append(ev)
    else:
        descr_events = events
    # convert the event specs into the format expected by BoxcarMapper
    # take the first event as an example of contained keys
    evvars = _events2dict(descr_events)
    # checks
    for p in ['onset', 'duration']:
        if not p in evvars:
            raise ValueError("'%s' is a required property for all events."
                             % p)
    boxlength = max(evvars['duration'])
    if __debug__:
        if not max(evvars['duration']) == min(evvars['duration']):
            warning('Boxcar mapper will use maximum boxlength (%i) of all '
                    'provided Events.'% boxlength)

    # finally create, train und use the boxcar mapper
    bcm = BoxcarMapper(evvars['onset'], boxlength, space=eprefix)
    bcm.train(ds)
    ds = ds.get_mapped(bcm)
    if event_mapper is None:
        # at last reflatten the dataset
        # could we add some meaningful attribute during this mapping, i.e. would
        # assigning 'inspace' do something good?
        ds = ds.get_mapped(FlattenMapper(shape=ds.samples.shape[1:]))
    else:
        ds = ds.get_mapped(event_mapper)
    # add samples attributes for the events, simply dump everything as a samples
    # attribute
    # special case onset and duration in case of conversion into descrete time
    if time_attr is not None:
        for attr in ('onset', 'duration'):
            evvars[attr] = [e[attr] for e in events]
    ds = _evvars2ds(ds, evvars, eprefix)

    return ds


def fit_event_hrf_model(
        ds, events, time_attr, condition_attr='targets', design_kwargs=None,
        glmfit_kwargs=None, regr_attrs=None, return_model=False):
    """Fit a GLM with HRF regressor and yield a dataset with model parameters

    A univariate GLM is fitted for each feature and model parameters are
    returned as samples. Model parameters are returned for each regressor in
    the design matrix. Using functionality from NiPy, design matrices can be
    generated from event definitions with a variety of customizations (HRF
    model, confound regressors, ...).

    Events need to be specified as a list of dictionaries
    (see:class:`~mvpa2.misc.support.Event`) for a helper class. Each dictionary
    contains all relevant attributes to describe an event.

    HRF event model details
    -----------------------

    The event specifications are used to generate a design matrix for all
    present conditions. In addition to the mandatory ``onset`` information
    each event definition needs to include a label in order to associate
    individual events to conditions (the design matrix will contain at least
    one regressor for each condition). The name of this label attribute must
    be specified too (see ``condition_attr`` argument).

    NiPy is used to generate the actual design matrix.  It is required to
    specify a dataset sample attribute that contains time stamps for all input
    data samples (see ``time_attr``).  NiPy operation could be customized (see
    ``design_kwargs`` argument). Additional regressors from sample attributes
    of the input dataset can be included in the design matrix (see
    ``regr_attrs``).

    The actual GLM fit is also performed by NiPy and can be fully customized
    (see ``glmfit_kwargs``).

    Parameters
    ----------
    ds : Dataset
      The samples of this input dataset have to be in whatever ascending order.
    events : list
      Each event definition has to specify ``onset`` and ``duration``. All
      other attributes will be passed on to the sample attributes collection of
      the returned dataset.
    time_attr : str
      Attribute with dataset sample time stamps.
      Its values will be treated as in-the-same-unit and are used to
      determine corresponding samples from real-value onset and duration
      definitions. For HRF modeling this argument is mandatory.
    condition_attr : str
      Name of the event attribute with the condition labels.
      Can be a list of those (e.g. ['targets', 'chunks'] combination of which
      would constitute a condition.
    design_kwargs : dict
      Arbitrary keyword arguments for NiPy's make_dmtx() used for design matrix
      generation. Choose HRF model, confound regressors, etc.
    glmfit_kwargs : dict
      Arbitrary keyword arguments for NiPy's GeneralLinearModel.fit() used for
      estimating model parameter. Choose fitting algorithm: OLS or AR1.
    regr_attrs : list
      List of dataset sample attribute names that shall be extracted from the
      input dataset and used as additional regressors in the design matrix.
    return_model : bool
      Flag whether to included the fitted GLM model in the returned dataset.
      For large input data this can be problematic, as the model may contain
      the residuals (same size is input data), hence multiplies the memory
      demand. Off by default.

    Returns
    -------
    Dataset
      One sample for each regressor/condition in the design matrix is returned.
      The condition names are included as a sample attribute with the name
      specified by the ``condition_attr`` argument.  The actual design
      regressors are included as ``regressors`` sample attribute. If enabled,
      an instance with the fitted NiPy GLM results is included as a dataset
      attribute ``model``, and can be used for computing contrasts subsequently.

    Examples
    --------
    The documentation also contains an :ref:`example script
    <example_eventrelated>` showing a spatio-temporal analysis of fMRI data
    that involves this function.

    >>> from mvpa2.datasets import Dataset
    >>> ds = Dataset(np.random.randn(10, 25))
    >>> ds.sa['time_coords'] = np.linspace(0, 50, len(ds))
    >>> events = [{'onset': 2, 'duration': 4, 'condition': 'one'},
    ...           {'onset': 4, 'duration': 4, 'condition': 'two'}]
    >>> hrf_estimates = fit_event_hrf_model(
    ...                   ds, events,
    ...                   time_attr='time_coords',
    ...                   condition_attr='condition',
    ...                   design_kwargs=dict(drift_model='blank'),
    ...                   glmfit_kwargs=dict(model='ols'),
    ...                   return_model=True)
    >>> print hrf_estimates.sa.condition
    ['one' 'two']
    >>> print hrf_estimates.shape
    (2, 25)
    >>> len(hrf_estimates.a.model.get_mse())
    25

    Additional regressors used in GLM modeling are also available in a
    dataset attribute:

    >>> print hrf_estimates.a.add_regs.sa.regressor_names
    ['constant']
    """
    if externals.exists('nipy', raise_=True):
        from nipy.modalities.fmri.design_matrix import make_dmtx
        from mvpa2.mappers.glm import NiPyGLMMapper

    # Decide/device condition attribute on which GLM will actually be done
    if isinstance(condition_attr, basestring):
        # must be a list/tuple/array for the logic below
        condition_attr = [condition_attr]

    glm_condition_attr = 'regressor_names' # actual regressors
    glm_condition_attr_map = dict([(con, dict()) for con in condition_attr])    #
    # to map back to original conditions
    events = copy.deepcopy(events)  # since we are modifying in place
    for event in events:
        if glm_condition_attr in event:
            raise ValueError("Event %s already has %s defined.  Should not "
                             "happen.  Choose another name if defined it"
                             % (event, glm_condition_attr))
        compound_label = event[glm_condition_attr] = \
            'glm_label_' + '+'.join(
                str(event[con]) for con in condition_attr)
        # and mapping back to original values, without str()
        # for each condition:
        for con in condition_attr:
            glm_condition_attr_map[con][compound_label] = event[con]

    evvars = _events2dict(events)
    add_paradigm_kwargs = {}
    if 'amplitude' in evvars:
        add_paradigm_kwargs['amplitude'] = evvars['amplitude']
    # create paradigm
    if 'duration' in evvars:
        from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
        # NiPy considers everything with a duration as a block paradigm
        paradigm = BlockParadigm(
                        con_id=evvars[glm_condition_attr],
                        onset=evvars['onset'],
                        duration=evvars['duration'],
                        **add_paradigm_kwargs)
    else:
        from nipy.modalities.fmri.experimental_paradigm \
                import EventRelatedParadigm
        paradigm = EventRelatedParadigm(
                        con_id=evvars[glm_condition_attr],
                        onset=evvars['onset'],
                        **add_paradigm_kwargs)
    # create design matrix -- all kinds of fancy additional regr can be
    # auto-generated
    if design_kwargs is None:
        design_kwargs = {}

    if regr_attrs is not None:
        names = []
        regrs = []
        for attr in regr_attrs:
            regr = ds.sa[attr].value
            # add rudimentary dimension for easy hstacking later on
            if regr.ndim < 2:
                regr = regr[:, np.newaxis]
            if regr.shape[1] == 1:
                names.append(attr)
            else:
                #  add one per each column of the regressor
                for i in xrange(regr.shape[1]):
                    names.append("%s.%d" % (attr, i))
            regrs.append(regr)
        regrs = np.hstack(regrs)

        if 'add_regs' in design_kwargs:
            design_kwargs['add_regs'] = np.hstack((design_kwargs['add_regs'],
                                                   regrs))
        else:
            design_kwargs['add_regs'] = regrs
        if 'add_reg_names' in design_kwargs:
            design_kwargs['add_reg_names'].extend(names)
        else:
            design_kwargs['add_reg_names'] = names

    design_matrix = make_dmtx(ds.sa[time_attr].value,
                              paradigm,
                              **design_kwargs)

    # push design into source dataset
    glm_regs = [
        (reg, design_matrix.matrix[:, i])
        for i, reg in enumerate(design_matrix.names)]

    # GLM
    glm = NiPyGLMMapper([], glmfit_kwargs=glmfit_kwargs,
            add_regs=glm_regs,
            return_design=True, return_model=return_model,
            space=glm_condition_attr)

    model_params = glm(ds)

    # some regressors might be corresponding not to original condition_attr
    # so let's separate them out
    regressor_names = model_params.sa[glm_condition_attr].value
    condition_regressors = np.array([v in glm_condition_attr_map.values()[0]
                                     for v in regressor_names])
    assert(condition_regressors.dtype == np.bool)
    if not np.all(condition_regressors):
        # some regressors do not correspond to conditions and would need
        # to be taken into a separate dataset
        model_params.a['add_regs'] = model_params[~condition_regressors]
        # then we process the rest
        model_params = model_params[condition_regressors]
        regressor_names = model_params.sa[glm_condition_attr].value

    # now define proper condition sa's
    for con, con_map in glm_condition_attr_map.iteritems():
        model_params.sa[con] = [con_map[v] for v in regressor_names]
    model_params.sa.pop(glm_condition_attr) # remove generated one
    return model_params


def eventrelated_dataset(ds, events, time_attr=None, match='prev',
                         eprefix='event', event_mapper=None,
                         condition_attr='targets', design_kwargs=None,
                         glmfit_kwargs=None, regr_attrs=None, model='boxcar'):
    """This function is deprecated.

    It is kept in order to maintain API compatibility with previous versions
    of PyMVPA. For new code, please use the following alternative functions:

    * :func:`~mvpa2.datasets.eventrelated.fit_event_hrf_model`
    * :func:`~mvpa2.datasets.eventrelated.extract_boxcar_event_samples`
    * :func:`~mvpa2.datasets.eventrelated.assign_conditionlabels`.
    """
    if not len(events):
        raise ValueError("no events specified")

    if model == 'boxcar':
        return extract_boxcar_event_samples(
                    ds, events=events, time_attr=time_attr, match=match,
                    eprefix=eprefix, event_mapper=event_mapper)
    elif model == 'hrf':
        if condition_attr is None:
            raise ValueError(
                    "missing name of event attribute with condition names")
        if time_attr is None:
            raise ValueError(
                    "missing name of attribute with sample timing information")
        return fit_event_hrf_model(
                    ds, events=events, time_attr=time_attr,
                    condition_attr=condition_attr,
                    design_kwargs=design_kwargs, glmfit_kwargs=glmfit_kwargs,
                    regr_attrs=regr_attrs)
    else:
        raise ValueError("unknown event model '%s'" % model)
