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

import numpy as np

from mvpa2.misc.io import ColumnData
from mvpa2.misc.support import Event

if __debug__:
    from mvpa2.base import debug


class FslEV3(ColumnData):
    """IO helper to read FSL's EV3 files.

    This is a three-column textfile format that is used to specify stimulation
    protocols for fMRI data analysis in FSL's FEAT module.

    Data is always read as `float`.
    """
    def __init__(self, source):
        """Read and write FSL EV3 files.

        Parameters
        ----------
        source : str
          Filename of an EV3 file
        """
        # init data from known format
        ColumnData.__init__(self, source,
                            header=['onsets', 'durations', 'intensities'],
                            sep=None, dtype=float)

    @property
    def nevs(self):
        """Returns the number of EVs in the file.
        """
        return self.nrows


    ##REF: Name was automagically refactored
    def get_ev(self, evid):
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


    ##REF: Name was automagically refactored
    def to_events(self, **kwargs):
        """Convert into a list of `Event` instances.

        Parameters
        ----------
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



class McFlirtParams(ColumnData):
    """Read and write McFlirt's motion estimation parameters from and to text
    files.
    """
    header_def = ['rot1', 'rot2', 'rot3', 'x', 'y', 'z']

    def __init__(self, source):
        """Initialize McFlirtParams

        Parameters
        ----------
        source : str
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
        import pylab as pl

        # translations subplot
        pl.subplot(211)
        pl.plot(self.x)
        pl.plot(self.y)
        pl.plot(self.z)
        pl.ylabel('Translations in mm')
        pl.legend(('x', 'y', 'z'), loc=0)

        # rotations subplot
        pl.subplot(212)
        pl.plot(self.rot1)
        pl.plot(self.rot2)
        pl.plot(self.rot3)
        pl.ylabel('Rotations in rad')
        pl.legend(('rot1', 'rot2', 'rot3'), loc=0)


    def toarray(self):
        """Returns the data as an array with six columns (same order as in file).
        """
        import numpy as np

        # return as array with time axis first
        return np.array([self[i] for i in McFlirtParams.header_def],
                       dtype='float').T


class FslGLMDesign(object):
    """Load FSL GLM design matrices from file.

    Be aware that such a desig matrix has its regressors in columns and the
    samples in its rows.
    """
    def __init__(self, source):
        """
        Parameters
        ----------
        source : filename
          Compressed files will be read as well, if their filename ends with
          '.gz'.
        """
        # XXX maybe load from array as well
        self._load_file(source)


    ##REF: Name was automagically refactored
    def _load_file(self, fname):
        """Helper function to load GLM definition from a file.
        """
        # header info
        nwaves = 0
        ntimepoints = 0
        matrix_offset = 0

        # open the file compressed or not
        if fname.endswith('.gz'):
            import gzip
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
        self.mat = np.loadtxt(fname, skiprows=matrix_offset)

        # checks
        if not self.mat.shape == (ntimepoints, nwaves):
            raise IOError, "Design matrix file '%s' did not contain expected " \
                           "matrix size (expected %s, got %s)" \
                           % (fname, str((ntimepoints, nwaves)), self.mat.shape)


    def plot(self, style='lines', **kwargs):
        """Visualize the design matrix.

        Parameters
        ----------
        style: 'lines', 'matrix'
        **kwargs:
          Additional arguments will be passed to the corresponding matplotlib
          plotting functions 'plot()' and 'pcolor()' for 'lines' and 'matrix'
          plots respectively.
        """
        # import internally as it takes some time and might not be needed most
        # of the time
        import pylab as pl

        if style == 'lines':
            # common y-axis
            yax = np.arange(0, self.mat.shape[0])
            axcenters = []
            col_offset = max(self.ppheights)

            # for all columns
            for i in xrange(self.mat.shape[1]):
                axcenter = i * col_offset
                pl.plot(self.mat[:, i] + axcenter, yax, **kwargs)
                axcenters.append(axcenter)

            pl.xticks(np.array(axcenters), range(self.mat.shape[1]))
        elif style == 'matrix':
            pl.pcolor(self.mat, **kwargs)
            ticks = np.arange(1, self.mat.shape[1] + 1)
            pl.xticks(ticks - 0.5, ticks)
        else:
            raise ValueError, "Unknown plotting style '%s'" % style

        # labels and turn y-axis upside down
        pl.ylabel('Samples (top to bottom)')
        pl.xlabel('Regressors')
        pl.ylim(self.mat.shape[0], 0)


def read_fsl_design(fsf_file):
    """Reads an FSL FEAT design.fsf file and return the content as a dictionary.

    Parameters
    ----------
    fsf_file : filename, file-like
    """
    # This function was originally contributed by Russell Poldrack

    if isinstance(fsf_file, basestring):
        infile = open(fsf_file, 'r')
    else:
        infile = fsf_file

    # target dict
    fsl = {}

    # loop over all lines
    for line in infile:
        line = line.strip()
        # if there is nothing on the line, do nothing
        if not line or line[0] == '#':
            continue

        # strip leading TCL 'set'
        key, value = line.split(None, 2)[1:]

        # fixup the 'y-' thing
        if value == 'y-':
            value = "y"

        # special case of variable keyword
        if line.count('_files('):
            # e.g. feat_files(1) -> feat_files
            key = key.split('(')[0]

        # decide which type we have for the value
        # int?
        if value.isdigit():
            fsl[key] = int(value)
        else:
            # float?
            try:
                fsl[key] = float(value)
            except ValueError:
                # must be string then, but...
                # sometimes there are quotes, sometimes not, but if the value
                # should be a string we remove them, since the value is already
                # of this type
                fsl[key] = value.strip('"')

    return fsl
