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


class ColumnData(dict):
    """Read data that is stored in columns of text files.

    All read data is available via a dictionary-like interface. If
    column headers are available, the column names serve as dictionary keys.
    If no header exists an articfical key is generated: str(number_of_column).

    Splitting of text file lines is performed by the standard split() function
    (which gets passed the `sep` argument as separator string) and each
    element is converted into the desired datatype.

    Because data is read into a dictionary no two columns can have the same
    name in the header! Each column is stored as a list in the dictionary.
    """
    def __init__(self, source, header=True, sep=None, dtype=float):
        """Read data from file into a dictionary.

        Parameters
        ----------
         - `source`: Can be a filename or a dictionary. In the case of the
                     first all data is read from that file and additonal
                     keyword arguments can be sued to customize the read
                     procedure. If a dictionary is passed a deepcopy is
                     performed.
         - `header`: Indicates whether the column names should be read from the
                     first line (`header=True`). If `header=False` unique
                     column names will be generated (see class docs). If
                     `header` is a python list, it's content is used as column
                     header names and its length has to match the number of
                     columns in the file.
         - `sep`: Separator string. The actual meaning depends on the output
                  format (see class docs).
         - `dtype`: Desired datatype.
        """
        if isinstance(source, str):
            self._fromFile(source, header=header, sep=sep, dtype=dtype)

        elif isinstance(source, dict):
            for k, v in source.iteritems():
                self[k] = v
            # check data integrity
            self._check()

        else:
            raise ValueError, 'Unkown source for ColumnData [%s]' \
                              % `type(source)`


    def _check(self):
        """Performs some checks for data integrity.
        """
        length = None
        for k in self.keys():
            if length == None:
                length = len(self[k])
            else:
                if not len(self[k]) == length:
                    raise ValueError, "Data integrity lost. Columns do not " \
                                      "have equal length."


    def _fromFile(self, filename, header, sep, dtype):
        """Loads column data from file -- clears object first.
        """
        # make a clean table
        self.clear()

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

        # check
        if not len(tbl) == len(hdr):
            raise RuntimeError, "Number of columns read from file does not " \
                                "match the number of header entries."

        # fill dict
        for i,v in enumerate(hdr):
            self[v] = tbl[i]


    def __iadd__(self, other):
        """Merge column data.
        """
        # for all columns in the other object
        for k, v in other.iteritems():
            if not self.has_key(k):
                raise ValueError, 'Unknown key [%s].' % `k`
            if not isinstance(v, list):
                raise ValueError, 'Can only merge list data, but got [%s].' \
                                  % `type(v)`
            # now it seems to be ok
            # XXX check for datatype?
            self[k] += v

        # look for problems, like columns present in self, but not in other
        self._check()

        return self


    def getNColumns(self):
        """Returns the number of columns.
        """
        return len(self.keys())


    def tofile(self, filename, header=True, sep=' '):
        """Write column data to a text file.

        Parameter
        ---------

        - `filename`: Think about it!
        - `header`: If `True` a column header is written, using the column
                    keys. If `False` no header is written. Finally, if it is
                    a list of string, they will be used instead of simply
                    asking for the dictionary keys. However these strings must
                    match the dictionary keys in number and identity. This
                    argument type can be used to determine the order of the
                    columns in the output file.
        - `sep`: String that is written as a separator between to data columns.
        """
        # XXX do the try: except: dance
        file = open(filename, 'w')

        # write header
        if header == True:
            col_hdr = self.keys()
        if isinstance(header, list):
            if not len(header) == self.getNColumns():
                raise ValueError, 'Header list does not matcg number of ' \
                                  'columns.'
            for k in header:
                if not self.has_key(k):
                    raise ValueError, 'Unknown key [%s]' % `k`
            col_hdr = header

        if not header == False:
            file.write(sep.join(col_hdr) + '\n')

        # for all rows
        for r in xrange(self.getNRows()):
            # get attributes for all keys
            l = [str(self[k][r]) for k in col_hdr]
            # write to file with proper separator
            file.write(sep.join(l) + '\n')

        file.close()


    def getNRows(self):
        """Returns the number of rows.
        """
        # no data no rows (after Bob Marley)
        if not len(self.keys()):
            return 0
        # otherwise first key is as good as any other
        else:
            return len(self[self.keys()[0]])



class FslEV3(ColumnData):
    """IO helper to read FSL's EV3 files.

    This is a three-column textfile format that is used to specify stimulation
    protocols for fMRI data analysis in FSL's FEAT module.

    Data is always read as `float`.
    """
    def __init__(self, source):
        """Read and write FSL EV3 files.

        Parameter
        ---------

        - `source`: filename of an EV3 file
        """
        # init data from known format
        ColumnData.__init__(self, source,
                            header=['onsets', 'durations', 'intensities'],
                            sep=None, dtype=float)


    def getNEVs(self):
        """Returns the number of EVs in the file.
        """
        return self.getNRows()


    def getEV(self, id):
        """Returns a tuple of (onset time, simulus duration, intensity) for a
        certain EV.
        """
        return (self['onsets'][id],
                self['durations'][id],
                self['intensities'][id])


    def tofile(self, filename):
        """Write data to a FSL EV3 file.
        """
        ColumnData.tofile(self, filename,
                          header=['onsets', 'durations', 'intensities'],
                          sep=' ')


    onsets = property(fget=lambda self: self['onsets'])
    durations = property(fget=lambda self: self['durations'])
    intensities = property(fget=lambda self: self['intensities'])
    nevs = property(fget=getNEVs)



class SampleAttributes(ColumnData):
    """Read and write PyMVPA sample attribute definitions from and to text
    files.
    """
    def __init__(self, source):
        """Read PyMVPA sample attributes from disk.

        Parameter
        ---------

        - `source`: filename of an atrribute file
        """
        ColumnData.__init__(self, source,
                            header=['labels', 'chunks'],
                            sep=None, dtype=float)


    def tofile(self, filename):
        """Write sample attributes to a text file.
        """
        ColumnData.tofile(self, filename,
                          header=['labels', 'chunks'],
                          sep=' ')


    def getNSamples(self):
        """Returns the number of samples in the file.
        """
        return self.getNRows()


    labels = property(fget=lambda self: self['labels'])
    chunks = property(fget=lambda self: self['chunks'])
    nsamples = property(fget=getNSamples)

