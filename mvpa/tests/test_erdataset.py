# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for the event-related dataset'''

from mvpa.testing import *
from mvpa.datasets import dataset_wizard
from mvpa.mappers.flatten import FlattenMapper
from mvpa.mappers.boxcar import BoxcarMapper
from mvpa.datasets.eventrelated import find_events, eventrelated_dataset


def test_erdataset():
    # 3 chunks, 5 targets, blocks of 5 samples each
    nchunks = 3
    ntargets = 5
    blocklength = 5
    nfeatures = 10
    targets = N.tile(N.repeat(range(ntargets), blocklength), nchunks)
    chunks = N.repeat(N.arange(nchunks), ntargets * blocklength)
    samples = N.repeat(
                N.arange(nchunks * ntargets * blocklength),
                nfeatures).reshape(-1, nfeatures)
    ds = dataset_wizard(samples, targets=targets, chunks=chunks)
    # check if events are determined properly
    evs = find_events({'targets':ds.sa.targets, 'chunks':ds.sa.chunks})
    for ev in evs:
        assert_equal(ev['duration'], blocklength)
    assert_equal(ntargets * nchunks, len(evs))
    for t in range(ntargets):
        assert_equal(len([ev for ev in evs if ev['targets'] == t]),
                     nchunks)
    # now turn `ds` into an eventreleated dataset
    erds = eventrelated_dataset(ds, evs)
    # the only unprefixed sample attributes are 
    assert_equal(sorted([a for a in ds.sa if not a.startswith('event')]),
                 ['chunks', 'targets'])
    # samples as expected?
    assert_array_equal(erds.samples[0],
                       N.repeat(N.arange(blocklength), nfeatures))
    # that should also be the temporal feature offset
    assert_array_equal(erds.samples[0], erds.fa.event_offsetidx)
    assert_array_equal(erds.sa.event_onsetidx, N.arange(0,71,5))
    # finally we should see two mappers
    assert_equal(len(erds.a.mapper), 2)
    assert_true(isinstance(erds.a.mapper[0], BoxcarMapper))
    assert_true(isinstance(erds.a.mapper[1], FlattenMapper))
    #
    # now check the same dataset with event descretization
    tr = 2.5
    ds.sa['time'] = N.arange(nchunks * ntargets * blocklength) * tr
    evs = [{'onset': 4.9, 'duration': 6.2}]
    # doesn't work without conversion
    assert_raises(ValueError, eventrelated_dataset, ds, evs)
    erds = eventrelated_dataset(ds, evs, time_attr='time')
    assert_equal(len(erds), 1)
    assert_array_equal(erds.samples[0], N.repeat(N.arange(1,5), nfeatures))
    assert_array_equal(erds.sa.orig_onset, [evs[0]['onset']])
    assert_array_equal(erds.sa.orig_duration, [evs[0]['duration']])
    assert_array_almost_equal(erds.sa.orig_offset, [2.4])
    assert_array_equal(erds.sa.time, [N.arange(2.5, 11, 2.5)])
    # now with closest match
    erds = eventrelated_dataset(ds, evs, time_attr='time', match='closest')
    assert_equal(len(erds), 1)
    assert_array_equal(erds.samples[0], N.repeat(N.arange(2,5), nfeatures))
    assert_array_equal(erds.sa.orig_onset, [evs[0]['onset']])
    assert_array_equal(erds.sa.orig_duration, [evs[0]['duration']])
    assert_array_almost_equal(erds.sa.orig_offset, [-0.1])
    assert_array_equal(erds.sa.time, [N.arange(5.0, 11, 2.5)])
