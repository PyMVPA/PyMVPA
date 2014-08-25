# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Extract (multi-sample) events from a dataset

An arbitrary number of input datasets is loaded from HDF5 storage. All loaded
datasets are concatenated along the samples axis. Based on information about
onset and duration of a sequence of events corresponding samples are extracted
from the input datasets and converted into event samples. It is possible for an
event sample to consist of multiple input samples (i.e. temporal windows).

Events are defined by onset sample ID and number of consecutive samples that
comprise an event. However, events can also be defined as temporal onsets and
durations, which will be translated into sample IDs using time stamp information
in the input datasets.

Analogous to the 'mkds' command the event-related dataset can be extended with
arbitrary feature and sample attributes (one value per event for the latter).

The finished event-related dataset is written to an HDF5 file.

Examples:

Extract two events comprising of four consecutive samples from a dataset.

  $ pymvpa2 mkevds --onsets 3 9 --duration 4 -o evds.hdf5 -i 'mydata*.hdf5'

"""

# magic line for manpage summary
# man: -*- % extract (multi-sample) events from a dataset

__docformat__ = 'restructuredtext'

import numpy as np
import sys
import argparse
from mvpa2.base import verbose, warning, error
from mvpa2.datasets import Dataset, vstack
from mvpa2.mappers.fx import FxMapper, merge2first
from mvpa2.datasets.eventrelated import eventrelated_dataset, find_events
if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers \
    import parser_add_common_opt, \
           ds2hdf5, arg2ds, process_common_dsattr_opts, _load_csv_table

parser_args = {
    'formatter_class': argparse.RawDescriptionHelpFormatter,
}

define_events_grp = ('options for defining events (choose one)', [
    (('--event-attrs',), dict(type=str, nargs='+', metavar='ATTR',
        help="""define events as a unique combinations of values from a set of
        sample attributes. Going through all samples in the order in which they
        appear in the input dataset, onset of events are determined by changes
        in the combination of attribute values. The length of an event is
        determined by the number of identical consecutive value combinations."""
        )),
    (('--onsets',), dict(type=float, nargs='*', metavar='TIME',
        help="""reads a list of event onsets (float) from the command line
        (space-separated). If this option is given, but no arguments are
        provided, onsets will be read from STDIN (one per line). If --time-attr
        is also given, onsets will be interpreted as time stamps, otherwise
        they are treated a integer ID of samples."""
        )),
    (('--csv-events',), dict(type=str, metavar='FILENAME',
        help="""read event information from a CSV table. A variety of dialects
        are supported. A CSV file must contain a header line with field names
        as a first row. The table must include an 'onset' column, and can
        optionally include an arbitrary number of additional columns
        (e.g. duration, target). All values are passed on to the event-related
        samples. If '-' is given as a value the CSV table is read from STDIN.
        """)),
    (('--fsl-ev3',), dict(type=str, nargs='+', metavar='FILENAME',
        help="""read event information from a text file in FSL's EV3 format
        (one event per line, three columns: onset, duration, intensity). One
        of more filenames can be given.""")),
])

mod_events_grp = ('options for modifying or converting events', [
    (('--time-attr',), dict(type=str, metavar='ATTR',
        help="""dataset attribute with time stamps for input samples. Onset and
        duration for all events will be converted using this information. All
        values are assumed to be of the same units.""")),
    (('--onset-column',), dict(type=str, metavar='ATTR',
        help="""name of the column in the CSV event table that indicates event
        onsets""")),
    (('--offset',), dict(type=float, metavar='VALUE',
        help="""fixed uniform event offset for all events. If no --time-attr
        option is given, this value indicates the number of input samples all
        event onsets shall be shifted. If --time-attr is given, this is treated
        as a temporal offset that needs to be given in the same unit as the time
        stamp attribute (see --time-attr).""")),
    (('--duration',), dict(type=float, metavar='VALUE',
        help="""fixed uniform duration for all events. If no --time-attr option
        is given, this value indicates the number of consecutive input samples
        following an onset that belong to an event. If --time-attr is given,
        this is treated as a temporal duration that needs to be given in the
        same unit as the time stamp attribute (see --time-attr).""")),
    (('--match-strategy',), dict(type=str, choices=('prev', 'next', 'closest'),
        default='prev',
        help="""strategy used to match time-based onsets to sample indices.
        'prev' chooses the closes preceding samples, 'next' the closest
        following sample and 'closest' to absolute closest sample. Default:
        'prev'""")),
    (('--event-compression',), dict(choices=('mean', 'median', 'min', 'max'),
        help="""specify whether and how events spanning multiple input samples
        shall be compressed. A number of methods can be chosen. Selecting, for
        example, 'mean' will yield the mean of all relevant input samples for
        an event. By default (when this option is not given) an event will
        comprise of all concatenated input samples.""")),
])

def setup_parser(parser):
    from .helpers import parser_add_optgroup_from_def, \
        parser_add_common_attr_opts, single_required_hdf5output
    parser_add_common_opt(parser, 'multidata', required=True)
    parser_add_optgroup_from_def(parser, define_events_grp, exclusive=True)
    parser_add_optgroup_from_def(parser, mod_events_grp)
    parser_add_common_attr_opts(parser)
    parser_add_optgroup_from_def(parser, single_required_hdf5output)


def run(args):
    ds = arg2ds(args.data)
    verbose(3, 'Concatenation yielded %i samples with %i features' % ds.shape)
    # build list of events
    events = []
    timebased_events = False
    if not args.event_attrs is None:
        def_attrs = dict([(k, ds.sa[k].value) for k in args.event_attrs])
        events = find_events(**def_attrs)
    elif not args.csv_events is None:
        if args.csv_events == '-':
            csv = sys.stdin.read()
            import cStringIO
            csv = cStringIO.StringIO(csv)
        else:
            csv = open(args.csv_events, 'rU')
        csvt = _load_csv_table(csv)
        if not len(csvt):
            raise ValueError("no CSV columns found")
        if args.onset_column:
            csvt['onset'] = csvt[args.onset_column]
        nevents = len(csvt[csvt.keys()[0]])
        events = []
        for ev in xrange(nevents):
            events.append(dict([(k, v[ev]) for k, v in csvt.iteritems()]))
    elif not args.onsets is None:
        if not len(args.onsets):
            args.onsets = [i for i in sys.stdin]
        # time or sample-based?
        if args.time_attr is None:
            oconv = int
        else:
            oconv = float
        events = [{'onset': oconv(o)} for o in args.onsets]
    elif not args.fsl_ev3 is None:
        timebased_events = True
        from mvpa2.misc.fsl import FslEV3
        events = []
        for evsrc in args.fsl_ev3:
            events.extend(FslEV3(evsrc).to_events())
    if not len(events):
        raise ValueError("no events defined")
    verbose(2, 'Extracting %i events' % len(events))
    if args.offset:
        # shift events
        for ev in events:
            ev['onset'] += args.offset
    if args.duration:
        # overwrite duration
        for ev in events:
            ev['duration'] = args.duration
    if args.event_compression is None:
        evmap = None
    elif args.event_compression == 'mean':
        evmap = FxMapper('features', np.mean, attrfx=merge2first)
    elif args.event_compression == 'median':
        evmap = FxMapper('features', np.median, attrfx=merge2first)
    elif args.event_compression == 'min':
        evmap = FxMapper('features', np.min, attrfx=merge2first)
    elif args.event_compression == 'max':
        evmap = FxMapper('features', np.max, attrfx=merge2first)
    # convert to event-related ds
    evds = eventrelated_dataset(ds, events, time_attr=args.time_attr,
                                match=args.match_strategy,
                                event_mapper=evmap)
    # act on all attribute options
    evds = process_common_dsattr_opts(evds, args)
    # and store
    ds2hdf5(evds, args.output, compression=args.hdf5_compression)
    return evds
