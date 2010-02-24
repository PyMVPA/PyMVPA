# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Filtering mappers."""

#
# This code is broken and sits here till it becomes something amazing.
#

raise NotImplementedError

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa.base import externals
if externals.exists('scipy', raiseException=True):
    from scipy.signal import resample

from mvpa.base.dochelpers import _str, borrowkwargs
from mvpa.mappers.base import Mapper


class FFTResamplemapper(Mapper):
    def __init__(self, chunks_attr=None, inspace=None):
        Mapper.__init__(self, inspace=inspace)

        self.__chunks_attr = chunks_attr


    def __repr__(self):
        s = super(FFTResamplemapper, self).__repr__()
        return s.replace("(",
                         "(chunks_attr=%s, "
                          % (repr(self.__chunks_attr),),
                         1)


    def __str__(self):
        return _str(self, chunks_attr=self.__chunks_attr)


    def _resample(self, nt=None, sr=None, dt=None, window='ham',
                  inplace=True, **kwargs):
        """Convenience method to resample data sample channel-wise.

        Resampling target can be specified by number of timepoint
        or temporal distance or sampling rate.

        Please note that this method only operates on
        `ChannelDataset` and always returns such.

        Parameters
        ----------
        nt : int
          Number of timepoints to resample to.
        dt : float
          Temporal distance of samples after resampling.
        sr : float
          Target sampling rate.
        inplace : bool
          If inplace=False, it would create and return a new dataset
          with new samples
        **kwargs
          All additional arguments are passed to resample() from
          scipy.signal

        Returns
        -------
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


        nt = np.round(nt)

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
            mapper = MaskMapper(np.ones(data.shape[1:], dtype='bool'))
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



@borrowkwargs(FFTResamplemapper, '__init__')
def fft_resample(ds, **kwargs):
    """
    Parameters
    ----------
    ds : Dataset
    **kwargs
      For all other arguments, please see the documentation of
      FFTResamplemapper.
    """
    dm = FFTResamplemapper(**kwargs)
    return dm(ds)
