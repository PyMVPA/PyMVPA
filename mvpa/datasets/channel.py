#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset handling data structured in channels."""

__docformat__ = 'restructuredtext'


import numpy as N

from mvpa.datasets.mapped import MappedDataset
from mvpa.mappers.mask import MaskMapper
from mvpa.base.dochelpers import enhancedDocString


class ChannelDataset(MappedDataset):
    """Dataset handling data structured into channels.

    Channels are assumes to contain several timepoints, thus this Dataset
    stores some additional properties (reference time `t0`, temporal
    distance of two timepoints `dt` and `channelids` (names)).
    """
    def __init__(self, samples=None, dsattr=None,
                 t0=None, dt=None, channelids=None, **kwargs):
        """Initialize ChannelDataset.

        :Parameters:
          samples: ndarray
            Three-dimensional array: (samples x channels x timepoints).
          t0: float
            Reference time of the first timepoint. Can be used to preserve
            information about the onset of some stimulation. Preferably in
            seconds.
          dt: float
            Temporal distance between two timepoints. Has to be given in
            seconds. Otherwise `samplingrate` property will not return
            `Hz`.
          channelids: list
            List of channel names.
        """
        # if dsattr is none, set it to an empty dict
        if dsattr is None:
            dsattr = {}

        # check samples
        if not samples is None and len(samples.shape) != 3:
                raise ValueError, \
                  "ChannelDataset takes 3D array as samples."

        # charge dataset properties
        # but only if some value
        if (not dt is None) or not dsattr.has_key('ch_dt'):
            dsattr['ch_dt'] = dt
        if (not channelids is None) or not dsattr.has_key('ch_ids'):
            dsattr['ch_ids'] = channelids
        if (not t0 is None) or not dsattr.has_key('ch_t0'):
            dsattr['ch_t0'] = t0

        # come up with mapper if fresh samples were provided
        if not samples is None:
            mapper = MaskMapper(N.ones(samples.shape[1:], dtype='bool'))
        else:
            mapper = None

        # init dataset
        MappedDataset.__init__(self,
                               samples=samples,
                               mapper=mapper,
                               dsattr=dsattr,
                               **(kwargs))


    __doc__ = enhancedDocString('ChannelDataset', locals(), MappedDataset)


    channelids = property(fget=lambda self: self._dsattr['ch_ids'],
                          doc='List of channel IDs')
    t0 = property(fget=lambda self: self._dsattr['ch_t0'],
                          doc='Temporal position of first sample in the ' \
                              'timeseries (in seconds) -- possibly relative ' \
                              'to stimulus onset.')
    dt = property(fget=lambda self: self._dsattr['ch_dt'],
                          doc='Time difference between two samples ' \
                              '(in seconds).')
    samplingrate = property(fget=lambda self: 1.0 / self._dsattr['ch_dt'],
                          doc='Yeah, sampling rate.')
