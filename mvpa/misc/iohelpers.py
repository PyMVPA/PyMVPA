#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Some little helper for reading (and writing) common formats from and to
disk."""

__docformat__ = 'restructuredtext'

import numpy as N


class ColumnDataFromFile(dict):
    """Read data that is stored in columns of text files.

    All read data is available via a dictionary-like interface. If
    column headers are available, the column names serve as dictionary keys.
    If no header exists an articfical key is generated: str(number_of_column).

    Two output formats `otype` are supported: ndarray and list. In both cases
    the separator-string `sep` and the output datatype `dtype` can be
    specified. When `otype` is `list` the splitting of strings is performed
    by the standard split() function and each element is converted into the
    desired datatype. If `otype` is `ndarray` NumPy's `fromfile()` is used to
    read the data.

    Because data is read into a dictionary no two columns can have the same
    name in the header!
    """
    def __init__(self, filename, header=True, sep=None, otype=list,
                 dtype=float):
        """Read data from file into a dictionary.

        Parameters
        ----------
         - `filename`: Indeed!
         - `header`: Indicates whether the column names should be read from the
                     first line (`header=True`). If `header=False` unique column
                     names will be generated (see class docs). If `header` is
                     a python list, it's content is used as column header names
                     and its length has to match the number of columns in the
                     file.
         - `sep`: Separator string. The actual meaning depends on the output
                  format (see class docs).
         - `otype`: Output format: `ndarray` or `list`.
         - `dtype`: Desired datatype.
        """
        file = open(filename, 'r')

        # make column names, either take header or generate
        if header == True:
            # read first line and split by 'sep'
            hdr = file.readline().split(sep)
        elif isinstance(header, list):
            hdr = header
        else:
            hdr = [ str(i) for i in xrange(len(file.readline().split(sep))) ]
            # reset file to not miss the first line
            file.seek(0)

        if otype.__name__ == 'ndarray':
            # translate meaning of separator for fromfile()
            if sep == None:
                sep = ' '

            # tbl has columns on first index!
            tbl = N.fromfile(file,
                             sep=sep,
                             dtype=dtype).reshape(-1, len(hdr)).T

        elif otype.__name__ == 'list':
            # string in lists: one per column
            tbl = [ [] for i in xrange(len(hdr)) ]

            # parse line by line and feed into the lists
            for line in file:
                # get rid of leading and trailing whitespace
                line = line.strip()
                # ignore empty lines
                if not line:
                    continue
                l = line.split(sep)

                if not len(l) == len(hdr):
                    raise RuntimeError, \
                          "Number of entries in line does not match number " \
                          "of columns in header."

                for i, v in enumerate(l):
                    tbl[i].append(dtype(v))

        else:
            raise ValueError, "Unknown output type [%s]" % `otype`

        # check
        if not len(tbl) == len(hdr):
            raise RuntimeError, "Number of columns read from file does not " \
                                "match the number of header entries."

        # fill dict
        for i,v in enumerate(hdr):
            self[v] = tbl[i]



class FslEV3(ColumnDataFromFile):
    """IO helper to read FSL's EV3 files.

    This is a three-column textfile format that is used to specify stimulation
    protocols for fMRI data analysis in FSL's FEAT module.

    Data is always read as `float`.
    """
    def __init__(self, filename, otype=list):
        """Read EV3 file from disk.

        Parameter
        ---------

        - `filename`: hmm, what can I say....
        - `otype`: Output format: `ndarray` or `list`.
        """
        # init data from known format
        ColumnDataFromFile.__init__(self, filename,
                                    header=['onset', 'duration', 'intensity'],
                                    sep=None, otype=otype, dtype=float)


    def getNEVs(self):
        """Returns the number of EVs in the file.
        """
        return len(self['onset'])


    def getEV(self, id):
        """Returns a tuple of (onset time, simulus duration, intensity) for a
        certain EV.
        """
        return (self['onset'][id], self['duration'][id], self['intensity'][id])


    onset = property(fget=lambda self: self['onset'])
    duration = property(fget=lambda self: self['duration'])
    intensity = property(fget=lambda self: self['intensity'])
    nevs = property(fget=getNEVs)
