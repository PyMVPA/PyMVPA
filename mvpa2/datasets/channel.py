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

#
#
# THIS CODE IS OBSOLETE!
#
# PLEASE PORT substract_baseline() AND resample() TO WORK WITH ANY DATASET.
#

from mvpa2.base import warning

warning("Deprecated: ChannelDataset has vanished already")

if False:           # just to please Python so it could parse the file
    ##REF: Name was automagically refactored
    def substract_baseline(self, t=None):
        """Substract mean baseline signal from the each timepoint.

        The baseline is determined by computing the mean over all timepoints
        specified by `t`.

        The samples of the dataset are modified in-place and nothing is
        returned.

        Parameters
        ----------
        t : int or float or None
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
            t = np.abs(self.t0)

        # determine length of baseline in samples
        if isinstance(t, float):
            t = np.round(t * self.samplingrate)

        # get original data
        data = self.O

        # compute baseline
        # XXX: shouldn't this be done per chunk?
        baseline = np.mean(data[:, :, :t], axis=2)
        # remove baseline
        data -= baseline[..., np.newaxis]

        # put data back into dataset
        self.samples[:] = self.mapForward(data)
