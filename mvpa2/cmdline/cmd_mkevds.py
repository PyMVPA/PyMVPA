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
event sample to consist of multiple input sample (i.e. temporal windows).

Events are defined by onset sample ID and number of consecutive samples that
comprise an event. However, events can also be defined as temporal onsets and
durations, which will be translated into sample IDs using time stamp information
in the input datasets

Analogous to the 'mkds' command the event-related dataset can be extended with
arbitrary feature and sample attributes (one value per event for the latter).

The finished event-related dataset is written to an HDF5 file.

Examples:

Extract two events comprising of four consecutive samples from a dataset.

  $ pymvpa2 mkevds --onset-samples 3 9 --event-nsamples 4 -o evds.hdf5 mydata*.hdf5

"""

# magic line for manpage summary
# man: -*- % extract (multi-sample) events from a dataset

__docformat__ = 'restructuredtext'

import numpy as np
import sys
import argparse
from mvpa2.base import verbose, warning, error
from mvpa2.datasets import Dataset, vstack
from mvpa2.datasets.eventrelated import eventrelated_dataset, find_events
if __debug__:
    from mvpa2.base import debug
from mvpa2.cmdline.helpers \
    import parser_add_common_args, parser_add_common_opt, \
           ds2hdf5, hdf2ds, process_common_attr_opts

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
    (('--onset-samples',), dict(type=int, nargs='*', metavar='ID',
        help="""reads a list of onset sample IDs (integer) from the command line
        (space-separated). If this option is given, but no arguments are
        provided, onset sample IDs will be read from STDIN (one per line).
        By default the onset input sample will compose the event. This can be
        changed via --events-nsamples.""")),
    (('--onset-times',), dict(type=float, nargs='*', metavar='TIME',
        help="""reads a list of onset time stamps (float) from the command line
        (space-separated). If this option is given, but no arguments are
        provided, onset time stamps will be read from STDIN (one per line)."""
        )),
    (('--fsl-ev3',), dict(type=str, nargs='+', metavar='FILENAME',
        help="""read event information from a text file in FSL's EV3 format
        (one event per line, three columns: onset, duration, intensity). One
        of more filenames can be given.""")),
])

timebased_events_grp = ('options for time-based event definitions', [
    (('--time-attr',), dict(type=str, metavar='ATTR',
        help="""dataset attribute with time stamps for input samples. Onset and
        duration for all events will be converted using this information. All
        values are assumed to be in the same unit.""")),
    (('--event-duration',), dict(type=float, metavar='TIME',
        help="""fixed uniform duration for all time-based event definitions.
        Needs to be given in the same unit as the time stamp attribute (see
        --time-attr).""")),
    (('--match-strategy',), dict(type=str, choices=('prev', 'next', 'closest'),
        default='prev',
        help="""strategy used to match time-based onsets to sample indices.
        'prev' chooses the closes preceding samples, 'next' the closest
        following sample and 'closest' to absolute closest sample. Default:
        'prev'""")),
])

samplebased_events_grp = ('options for sample-based event definitions', [
    (('--event-nsamples',), dict(type=int, metavar='#SAMPLES',
        help="""fixed uniform duration for all sample-based event definitions.
        Needs to be given as number of consecutive input samples.""")),
])



def setup_parser(parser):
    from .helpers import parser_add_optgroup_from_def, \
        parser_add_common_attr_opts, single_required_hdf5output
    parser_add_common_args(parser, pos=['multidata'])
    parser_add_optgroup_from_def(parser, define_events_grp, exclusive=True)
    parser_add_optgroup_from_def(parser, timebased_events_grp)
    parser_add_optgroup_from_def(parser, samplebased_events_grp)
    parser_add_common_attr_opts(parser)
    parser_add_optgroup_from_def(parser, single_required_hdf5output)


def run(args):
    dss = hdf2ds(args.data)
    verbose(3, 'Loaded %i dataset(s)' % len(dss))
    ds = vstack(dss)
    verbose(3, 'Concatenation yielded %i samples with %i features' % ds.shape)
    # build list of events
    events = []
    timebased_events = False
    if not args.event_attrs is None:
        def_attrs = dict([(k, ds.sa[k].value) for k in args.event_attrs])
        events = find_events(**def_attrs)
    elif not args.onset_samples is None:
        if not len(args.onset_samples):
            args.onset_samples = [int(i) for i in sys.stdin]
        events = [{'onset': o, 'duration': 1} for o in args.onset_samples]
    elif not args.onset_times is None:
        timebased_events = True
        if not len(args.onset_times):
            args.onset_times = [float(i) for i in sys.stdin]
        events = [{'onset': o} for o in args.onset_times]
    elif not args.fsl_ev3 is None:
        timebased_events = True
        from mvpa2.misc.fsl import FslEV3
        events = []
        for evsrc in args.fsl_ev3:
            events.extend(FslEV3(evsrc).to_events())
    if not len(events):
        raise ValueError("no events defined")
    verbose(2, 'Extracting %i events' % len(events))
    if timebased_events:
        if args.time_attr is None:
            raise ValueError("a dataset attribute with sample time stamps"
                             " needs to be specified")
        if args.event_duration:
            # overwrite duration
            for ev in events:
                ev['duration'] = args.event_duration
    else:
        if args.event_nsamples:
            # overwrite duration
            for ev in events:
                ev['duration'] = args.event_nsamples
    # convert to event-related ds
    evds = eventrelated_dataset(ds, events, time_attr=args.time_attr,
                                match=args.match_strategy)
    # act on all attribute options
    evds = process_common_attr_opts(evds, args)
    # and store
    ds2hdf5(ds, args.output, compression=args.compression)
    return evds
