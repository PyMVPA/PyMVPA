# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Support for EEGLAB's electrode-time series text file format.

This module offers functions to import data from EEGLAB_ text files.

.. _EEGLAB: http://sccn.ucsd.edu/eeglab/
"""

__docformat__ = 'restructuredtext'

import numpy as np
import os

from mvpa2.datasets.base import Dataset
from mvpa2.mappers.flatten import FlattenMapper

# restrict public interface to not misguide sphinx
__all__ = [ 'eeglab_dataset' ]

def _looks_like_filename(s):
    if os.path.exists(s):
        return True
    return len(s) <= 256 and not '\n' in s

def eeglab_dataset(samples):
    '''Make a Dataset instance from EEGLAB input data

    Parameters
    ----------
    samples: str
        Filename of EEGLAB text file

    Returns
    -------
    ds: mvpa2.base.dataset.Dataset
        Dataset with the contents of the input file
    '''
    if not isinstance(samples, basestring):
        raise ValueError("Samples should be a string")

    if _looks_like_filename(samples):
        if not os.path.exists(samples):
            raise ValueError("Input looks like a filename, but file"
                                " %s does not exist" % samples)
        with open(samples) as f:
            samples = f.read()

    lines = samples.split('\n')
    samples = []
    cur_sample = None

    for i, line in enumerate(lines):
        if not line:
            continue
        if i == 0:
            # first line contains the channel names
            channel_labels = line.split()
            n_channels = len(channel_labels)
        else:
            # first value is the time point, the remainders the value 
            # for each channel
            values = map(float, line.split())
            t = values[0]  # time 
            eeg = values[1:] # values for each electrode

            if len(eeg) != n_channels:
                raise ValueError("Line %d: expected %d values but found %d" %
                                    (n_channels, len(eeg)))

            if cur_sample is None or t < prev_t:
                # new sample
                cur_sample = []
                samples.append(cur_sample)

            cur_sample.append((t, eeg))
            prev_t = t

    # get and verify number of elements in each dimension
    n_samples = len(samples)
    n_timepoints_all = map(len, samples)

    n_timepoints_unique = set(n_timepoints_all)
    if len(n_timepoints_unique) != 1:
        raise ValueError("Different number of time points in different"
                            "samples: found %d different lengths" %
                            len(n_timepoints_unique))

    n_timepoints = n_timepoints_all[0]

    shape = (n_samples, n_timepoints, n_channels)

    # allocate space for data
    data = np.zeros(shape)

    # make a list of all channels and timepoints
    channel_array = np.asarray(channel_labels)
    timepoint_array = np.asarray([samples[0][i][0]
                                  for i in xrange(n_timepoints)])

    dts = timepoint_array[1:] - timepoint_array[:-1]
    if not np.all(dts == dts[0]):
        raise ValueError("Delta time points are different")

    # put the values in the data array
    for i, sample in enumerate(samples):
        for j, (t, values) in enumerate(sample):
            # check that the time is the same
            if i > 0 and timepoint_array[j] != t:
                raise ValueError("Sample %d, time point %s is different "
                                 "than the first sample (%s)" %
                                 (i, t, timepoint_array[j]))

            for k, value in enumerate(values):
                data[i, j, k] = value

    samples = None # and let gc do it's job

    # make a Dataset instance with the data
    ds = Dataset(data)

    # append a flatten_mapper to go from 3D (sample X time X channel)
    # to 2D (sample X (time X channel))
    flatten_mapper = FlattenMapper(shape=shape[1:], space='time_channel_indices')
    ds = ds.get_mapped(flatten_mapper)

    # make this a 3D array of the proper size
    channel_array_3D = np.tile(channel_array, (1, n_timepoints, 1))
    timepoint_array_3D = np.tile(np.reshape(timepoint_array, (-1, 1)),
                                            (1, 1, n_channels))

    # for consistency use the flattan_mapper defined above to 
    # flatten channel and timepoint names as well
    ds.fa['channelids'] = flatten_mapper.forward(channel_array_3D).ravel()
    ds.fa['timepoints'] = flatten_mapper.forward(timepoint_array_3D).ravel()

    # make some dynamic properties
    # XXX at the moment we don't have propert 'protection' in case
    # the feature space is sliced in a way so that some channels and/or
    # timepoints occur more often than others 
    _eeglab_set_attributes(ds)

    return ds

def _eeglab_set_attributes(ds):
    setattr(ds.__class__, 'nchannels', property(
            fget=lambda self: len(set(self.fa['time_channel_indices'][:, 1]))))
    setattr(ds.__class__, 'ntimepoints', property(
            fget=lambda self: len(set(self.fa['time_channel_indices'][:, 0]))))

    setattr(ds.__class__, 'channelids', property(
            fget=lambda self: np.unique(self.fa['channelids'].value)))
    setattr(ds.__class__, 'timepoints', property(
            fget=lambda self: np.unique(self.fa['timepoints'].value)))


    setattr(ds.__class__, 't0', property(
                    fget=lambda self: np.min(self.fa['timepoints'].value)))

    def _get_dt(ds):
        ts = np.unique(ds.fa['timepoints'].value)
        if len(ts) >= 1:
            delta = ts[1:] - ts[:-1]
            if len(np.unique(delta)) == 1:
                return delta[0]
        return float(numpy.nan)

    setattr(ds.__class__, 'dt', property(fget=lambda self: _get_dt(self)))

    def selector(f, xs):
        if type(f) in (list, tuple):
            flist = f
            f = lambda x:x in flist

        return np.nonzero(map(f, xs))[0]

    # attributes for getting certain time points or channels ids
    # the argument f should be either be a function, or a list or tuple
    setattr(ds.__class__, 'get_features_by_timepoints', property(
                            fget=lambda self: lambda f: selector(f,
                                            self.fa['timepoints']),
                            doc='Given a filter function f returns the '
                                'indices of features for which f(x) holds '
                                ' for each x in timepoints'))
    setattr(ds.__class__, 'get_features_by_channelids', property(
                            fget=lambda self: lambda f: selector(f,
                                            self.fa['channelids']),
                            doc='Given a filter function f returns the '
                                'indices of features for which f(x) holds '
                                ' for each x in channelids'))
