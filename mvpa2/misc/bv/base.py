# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tiny snippets to interface with FSL easily."""

__docformat__ = 'restructuredtext'

from mvpa2.misc.io import ColumnData

if __debug__:
    from mvpa2.base import debug


class BrainVoyagerRTC(ColumnData):
    """IO helper to read BrainVoyager RTC files.

    This is a textfile format that is used to specify stimulation
    protocols for data analysis in BrainVoyager. It looks like

    FileVersion:     2
    Type:            DesignMatrix
    NrOfPredictors:  4
    NrOfDataPoints:  147

    "fm_l_60dB" "fm_r_60dB" "fm_l_80dB" "fm_r_80dB"
    0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000

    Data is always read as `float` and header is actually ignored
    """
    def __init__(self, source):
        """Read and write BrainVoyager RTC files.

        Parameters
        ----------
        source : str
          Filename of an RTC file
        """
        # init data from known format
        ColumnData.__init__(self, source, header=True,
                            sep=None, headersep='"', dtype=float, skiplines=5)


    def toarray(self):
        """Returns the data as an array
        """
        import numpy as np

        # return as array with time axis first
        return np.array([self[i] for i in self._header_order],
                       dtype='float').T

