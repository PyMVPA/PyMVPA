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

from mvpa.misc.io import ColumnData
from mvpa.misc.support import Event

if __debug__:
    from mvpa.base import debug


class FslEV3(ColumnData):
    """IO helper to read FSL's EV3 files.

    This is a three-column textfile format that is used to specify stimulation
    protocols for fMRI data analysis in FSL's FEAT module.

    Data is always read as `float`.
    """
    def __init__(self, source):
        """Read and write FSL EV3 files.

        :Parameter:
          source: filename of an EV3 file
        """
        # init data from known format
        ColumnData.__init__(self, source,
                            header=['onsets', 'durations', 'intensities'],
                            sep=None, dtype=float)


    def getNEVs(self):
        """Returns the number of EVs in the file.
        """
        return self.getNRows()


    def getEV(self, evid):
        """Returns a tuple of (onset time, simulus duration, intensity) for a
        certain EV.
        """
        return (self['onsets'][evid],
                self['durations'][evid],
                self['intensities'][evid])


    def tofile(self, filename):
        """Write data to a FSL EV3 file.
        """
        ColumnData.tofile(self, filename,
                          header=False,
                          header_order=['onsets', 'durations', 'intensities'],
                          sep=' ')


    def toEvents(self, **kwargs):
        """Convert into a list of `Event` instances.

        :Parameters:
          kwargs
            Any keyword arugment provided would be replicated, through all
            the entries. Useful to specify label or even a chunk
        """
        return \
            [Event(onset=self['onsets'][i],
                   duration=self['durations'][i],
                   features=[self['intensities'][i]],
                   **kwargs)
             for i in xrange(self.nevs)]


    onsets = property(fget=lambda self: self['onsets'])
    durations = property(fget=lambda self: self['durations'])
    intensities = property(fget=lambda self: self['intensities'])
    nevs = property(fget=getNEVs)



class McFlirtParams(ColumnData):
    """Read and write McFlirt's motion estimation parameters from and to text
    files.
    """
    header_def = ['rot1', 'rot2', 'rot3', 'x', 'y', 'z']

    def __init__(self, source):
        """
        :Parameter:

            source: str
                Filename of a parameter file.
        """
        ColumnData.__init__(self, source,
                            header=McFlirtParams.header_def,
                            sep=None, dtype=float)


    def tofile(self, filename):
        """Write motion parameters to file.
        """
        ColumnData.tofile(self, filename,
                          header=False,
                          header_order=McFlirtParams.header_def,
                          sep=' ')


    def plot(self):
        """Produce a simple plot of the estimated translation and rotation
        parameters using.

        You still need to can pylab.show() or pylab.savefig() if you want to
        see/get anything.
        """
        # import internally as it takes some time and might not be needed most
        # of the time
        import pylab as P

        # translations subplot
        P.subplot(211)
        P.plot(self.x)
        P.plot(self.y)
        P.plot(self.z)
        P.ylabel('Translations in mm')
        P.legend(('x', 'y', 'z'), loc=0)

        # rotations subplot
        P.subplot(212)
        P.plot(self.rot1)
        P.plot(self.rot2)
        P.plot(self.rot3)
        P.ylabel('Rotations in rad')
        P.legend(('rot1', 'rot2', 'rot3'), loc=0)


    def toarray(self):
        """Returns the data as an array with six columns (same order as in file).
        """
        import numpy as N

        # return as array with time axis first
        return N.array([self[i] for i in McFlirtParams.header_def],
                       dtype='float').T

