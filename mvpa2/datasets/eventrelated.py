# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset for event-related samples."""

__docformat__ = 'restructuredtext'

import copy
import numpy as np
from mvpa2.misc.support import Event, value2idx
from mvpa2.base.dataset import _expand_attribute
from mvpa2.mappers.fx import _uniquemerge2literal
from mvpa2.mappers.flatten import FlattenMapper
from mvpa2.mappers.boxcar import BoxcarMapper
from mvpa2.base import warning


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
            if not old_combo is None:
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
    if not old_combo is None:
        events.append(_build_event(prev_onset, duration, old_combo))

    return events


def eventrelated_dataset(ds, events=None, time_attr=None, match='prev',
                         eprefix='event', event_mapper=None):
    """Segment a dataset into a set of events.

    This function can be used to extract event-related samples from any
    time-series based dataset (actually, it don't have to be time series, but
    could also be any other type of ordered samples). Boxcar-shaped event
    samples, potentially spanning multiple input samples can be automatically
    extracted using :class:`~mvpa2.misc.support.Event` definition lists.  For
    each event all samples covering that particular event are used to form the
    corresponding sample.

    An event definition is a dictionary that contains ``onset`` (as sample index
    in the input dataset), ``duration`` (as number of consecutive samples after
    the onset), as well as an arbitrary number of additional attributes.

    Alternatively, ``onset`` and ``duration`` may also be given as real time
    stamps (or durations). In this case a to be specified samples attribute in
    the input dataset will be used to convert these into sample indices.

    Parameters
    ----------
    ds : Dataset
      The samples of this input dataset have to be in whatever ascending order.
    events : list
      Each event definition has to specify ``onset`` and ``duration``. All other
      attributes will be passed on to the sample attributes collection of the
      returned dataset.
    time_attr : str or None
      If not None, the ``onset`` and ``duration`` specs from the event list will
      be converted using information from this sample attribute. Its values will
      be treated as in-the-same-unit and are used to determine corresponding
      samples from real-value onset and duration definitions.
    match : {'prev', 'next', 'closest'}
      Strategy used to match real-value onsets to sample indices. 'prev' chooses
      the closes preceding samples, 'next' the closest following sample and
      'closest' to absolute closest sample.
    eprefix : str or None
      If not None, this prefix is used to name additional attributes generated
      by the underlying `~mvpa2.mappers.boxcar.BoxcarMapper`. If it is set to
      None, no additional attributes will be created.
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
      The returned dataset has one sample per each event definition that has
      been passed to the function.

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

    if not time_attr is None:
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
    evvars = {}
    for k in descr_events[0]:
        try:
            evvars[k] = [e[k] for e in descr_events]
        except KeyError:
            raise ValueError("Each event property must be present for all "
                             "events (could not find '%s')" % k)
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
    for a in evvars:
        if not eprefix is None and a in ds.sa:
            # if there is already a samples attribute like this, it got mapped
            # by BoxcarMapper (i.e. is multi-dimensional). We move it aside
            # under new `eprefix` name
            ds.sa[eprefix + '_' + a] = ds.sa[a]
        if a in ['onset', 'duration']:
            # special case: we want the non-discrete, original onset and
            # duration
            if not time_attr is None:
                # but only if there was a conversion happining, since otherwise
                # we get the same info from BoxcarMapper
                ds.sa[a] = [e[a] for e in events]
        else:
            ds.sa[a] = evvars[a]
    return ds
