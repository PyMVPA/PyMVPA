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

import numpy as N

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
        """Initialize McFlirtParams

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


class FslGLMDesign(object):
    """Load FSL GLM design matrices from file.

    Be aware that such a desig matrix has its regressors in columns and the
    samples in its rows.
    """
    def __init__(self, source):
        """
        :Parameter:
          source: filename
            Compressed files will be read as well, if their filename ends with
            '.gz'.
        """
        # XXX maybe load from array as well
        self._loadFile(source)


    def _loadFile(self, fname):
        """Helper function to load GLM definition from a file.
        """
        # header info
        nwaves = 0
        ntimepoints = 0
        matrix_offset = 0

        # open the file compressed or not
        if fname.endswith('.gz'):
            fh = gzip.open(fname, 'r')
        else:
            fh = open(fname, 'r')

        # read header
        for i, line in enumerate(fh):
            if line.startswith('/NumWaves'):
                nwaves = int(line.split()[1])
            if line.startswith('/NumPoints'):
                ntimepoints = int(line.split()[1])
            if line.startswith('/PPheights'):
                self.ppheights = [float(i) for i in line.split()[1:]]
            if line.startswith('/Matrix'):
                matrix_offset = i + 1

        # done with the header, now revert to NumPy's loadtxt for convenience
        fh.close()
        self.mat = N.loadtxt(fname, skiprows=matrix_offset)

        # checks
        if not self.mat.shape == (ntimepoints, nwaves):
            raise IOError, "Design matrix file '%s' did not contain expected " \
                           "matrix size (expected %s, got %s)" \
                           % (fname, str((ntimepoints, nwaves)), self.mat.shape)


    def plot(self, style='lines', **kwargs):
        """Visualize the design matrix.

        :Parameters:
          style: 'lines', 'matrix'
          **kwargs:
            Additional arguments will be passed to the corresponding matplotlib
            plotting functions 'plot()' and 'pcolor()' for 'lines' and 'matrix'
            plots respectively.
        """
        # import internally as it takes some time and might not be needed most
        # of the time
        import pylab as P

        if style == 'lines':
            # common y-axis
            yax = N.arange(0, self.mat.shape[0])
            axcenters = []
            col_offset = max(self.ppheights)

            # for all columns
            for i in xrange(self.mat.shape[1]):
                axcenter = i * col_offset
                P.plot(self.mat[:, i] + axcenter, yax, **kwargs)
                axcenters.append(axcenter)

            P.xticks(N.array(axcenters), range(self.mat.shape[1]))
        elif style == 'matrix':
            P.pcolor(self.mat, **kwargs)
            ticks = N.arange(1, self.mat.shape[1]+1)
            P.xticks(ticks - 0.5, ticks)
        else:
            raise ValueError, "Unknown plotting style '%s'" % style

        # labels and turn y-axis upside down
        P.ylabel('Samples (top to bottom)')
        P.xlabel('Regressors')
        P.ylim(self.mat.shape[0],0)
