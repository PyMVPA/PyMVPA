# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Dataset handling data structured in channels."""

__docformat__ = 'restructuredtext'


import numpy as N

from mvpa.datasets.base import Dataset
from mvpa.mappers.mask import MaskMapper
from mvpa.base.dochelpers import enhancedDocString

import mvpa.support.copy as copy

from mvpa.base import externals

if externals.exists('scipy'):
    from scipy import signal


class ChannelDataset(Dataset):
    """Dataset handling data structured into channels.

    Channels are assumes to contain several timepoints, thus this Dataset
    stores some additional properties (reference time `t0`, temporal
    distance of two timepoints `dt` and `channelids` (names)).
    """
    @classmethod
    def from_temporaldata(cls, samples, labels=None, chunks=None,
                          t0=None, dt=None, channelids=None):
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
        # check samples
        if len(samples.shape) != 3:
            raise DatasetError("ChannelDataset takes 3D array as samples.")

        ds = cls.from_masked(samples, labels=labels, chunks=chunks)

        # charge dataset properties
        # but only if some value
        if not dt is None:
            ds.a.add('dt', dt)
        if not channelids is None:
            ds.a.add('channelids', channelids)
        if not t0 is None:
            ds.a.add('t0', t0)

        return ds

    __doc__ = enhancedDocString('ChannelDataset', locals(), Dataset)


    def substractBaseline(self, t=None):
        """Substract mean baseline signal from the each timepoint.

        The baseline is determined by computing the mean over all timepoints
        specified by `t`.

        The samples of the dataset are modified in-place and nothing is
        returned.

        :Parameter:
          t: int | float | None
            If an integer, `t` denotes the number of timepoints in the from the
            start of each sample to be used to compute the baseline signal.
            If a floating point value, `t` is the duration of the baseline
            window from the start of each sample in whatever unit
            corresponding to the datasets `samplingrate`. Finally, if `None`
            the `t0` property of the dataset is used to determine `t` as it
            would have been specified as duration.
        """
        # if no baseline length is given, use t0
        if t is None:
            t = N.abs(self.t0)

        # determine length of baseline in samples
        if isinstance(t, float):
            t = N.round(t * self.samplingrate)

        # get original data
        data = self.O

        # compute baseline
        # XXX: shouldn't this be done per chunk?
        baseline = N.mean(data[:, :, :t], axis=2)
        # remove baseline
        data -= baseline[..., N.newaxis]

        # put data back into dataset
        self.samples[:] = self.mapForward(data)


    if externals.exists('scipy'):
        #XXX MH: The whole things needs to get out of this class and be turned
        # into a generec function that deals with all datasets having some
        # frequency information
        def resample(self, nt=None, sr=None, dt=None, window='ham',
                     inplace=True, **kwargs):
            """Convenience method to resample data sample channel-wise.

            Resampling target can be specified by number of timepoint
            or temporal distance or sampling rate.

            Please note that this method only operates on
            `ChannelDataset` and always returns such.

            :Parameters:
              nt: int
                Number of timepoints to resample to.
              dt: float
                Temporal distance of samples after resampling.
              sr: float
                Target sampling rate.
              inplace : bool
                If inplace=False, it would create and return a new dataset
                with new samples
              **kwargs:
                All additional arguments are passed to resample() from
                scipy.signal

            :Return:
              ChannelDataset
            """
            if nt is None and sr is None and dt is None:
                raise ValueError, \
                      "Required argument missing. Either needs ntimepoints, sr or dt."

            # get data in original shape
            orig_data = self.O

            if len(orig_data.shape) != 3:
                raise ValueError, "resample() only works with data from ChannelDataset."

            orig_nt = orig_data.shape[2]
            orig_length = self.dt * orig_nt

            if nt is None:
                # translate dt or sr into nt
                if not dt is None:
                    nt = orig_nt * float(self.dt) / dt
                elif not sr is None:
                    nt = orig_nt * float(sr) / self.samplingrate
                else:
                    raise RuntimeError, 'This should not happen!'
            else:
                raise RuntimeError, 'This should not happen!'


            nt = N.round(nt)

            # downsample data
            data = signal.resample(orig_data, nt, axis=2, window=window, **kwargs)
            new_dt = float(orig_length) / nt

            # would be needed for not inplace generation
            if inplace:
                self.a['dt'].value = new_dt
                # XXX We could have resampled range(nsamples) and
                #     rounded  it. and adjust then mapper's mask
                #     accordingly instead of creating a new one.
                #     It would give us opportunity to assess what
                #     resampling did...
                mapper = MaskMapper(N.ones(data.shape[1:], dtype='bool'))
                # reassign a new mapper.
                # XXX this is very evil -- who knows what mapper it is replacing
                self.a['mapper'].value = mapper
                self.samples = mapper.forward(data)
                return self
            else:
                # we have to pass dsattr inside to don't loose
                # some additional attributes such as
                # labels_map
                dsattr = copy.deepcopy(dsattr)
                return ChannelDataset.from_temporaldata(
                        data=self._data,
                                      dsattr=dsattr,
                                      samples=data,
                                      t0=self.t0,
                                      dt=new_dt,
                                      channelids=self.channelids,
                                      copy_data=True,
                                      copy_dsattr=False)

    channelids = property(fget=lambda self: self.a.channelids,
                          doc='List of channel IDs')
    t0 = property(fget=lambda self: self.a.t0,
                          doc='Temporal position of first sample in the ' \
                              'timeseries (in seconds) -- possibly relative ' \
                              'to stimulus onset.')
    dt = property(fget=lambda self: self.a.dt,
                          doc='Time difference between two samples ' \
                              '(in seconds).')
    samplingrate = property(fget=lambda self: 1.0 / self.a.dt,
                          doc='Yeah, sampling rate.')
