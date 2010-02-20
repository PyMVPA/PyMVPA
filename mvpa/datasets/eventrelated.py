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
import numpy as N
from mvpa.misc.support import Event, value2idx
from mvpa.base.dataset import _expand_attribute
from mvpa.mappers.fx import _uniquemerge2literal
from mvpa.mappers.flatten import FlattenMapper
from mvpa.mappers.boxcar import BoxcarMapper
from mvpa.base import warning


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
                         eprefix='event'):
    """XXX All docs need to be rewritten.

    Dataset with event-defined samples from a NIfTI timeseries image.

    This is a convenience dataset to facilitate the analysis of event-related
    fMRI datasets. Boxcar-shaped samples are automatically extracted from the
    full timeseries using :class:`~mvpa.misc.support.Event` definition lists.
    For each event all volumes covering that particular event in time
    (including partial coverage) are used to form the corresponding sample.

    The class supports the conversion of events defined in 'realtime' into the
    descrete temporal space defined by the NIfTI image. Moreover, potentially
    varying offsets between true event onset and timepoint of the first selected
    volume can be stored as an additional feature in the dataset.

    Additionally, the dataset supports masking. This is done similar to the
    masking capabilities of :class:`~mvpa.datasets.nifti.NiftiDataset`. However,
    the mask can either be of the same shape as a single NIfTI volume, or
    can be of the same shape as the generated boxcar samples, i.e.
    a samples consisting of three volumes with 24 slices and 64x64 inplane
    resolution needs a mask with shape (3, 24, 64, 64). In the former case the
    mask volume is automatically expanded to be identical in a volumes of the
    boxcar.

    Parameters
    ----------
    ds : Dataset
    events : list
    tr : float or None
      Temporal distance of two adjacent NIfTI volumes. This can be used
      to override the corresponding value in the NIfTI header.
    eprefix : str or None

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
            # figure out how many sample we need
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
    bcm = BoxcarMapper(evvars['onset'], boxlength, inspace=eprefix)
    bcm.train(ds)
    ds = ds.get_mapped(bcm)
    # at last reflatten the dataset
    # could we add some meaningful attribute during this mapping, i.e. would 
    # assigning 'inspace' do something good?
    ds = ds.get_mapped(FlattenMapper(shape=ds.samples.shape[1:]))
    # add samples attributes for the events, simply dump everything as a samples
    # attribute
    for a in evvars:
        if not eprefix is None and a in ds.sa:
            # if there is already a samples attribute like this, it got mapped
            # by BoxcarMapper (i.e. is multi-dimensional). We move it aside
            # under new `eprefix` name
            ds.sa[eprefix + '_' + a] = ds.sa[a]
        if a in ['onset', 'duration']:
            # special case: we want the non-descrete, original onset and
            # duration
            if not time_attr is None:
                # but only if there was a conversion happining, since otherwise
                # we get the same info from BoxcarMapper
                ds.sa[a] = [e[a] for e in events]
        else:
            ds.sa[a] = evvars[a]
    return ds
