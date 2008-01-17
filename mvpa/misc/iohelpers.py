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

import copy

from mvpa.misc import warning

if __debug__:
    from mvpa.misc import debug

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
        # init base class
        dict.__init__(self)

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

        # generate missing properties for each item in the header
        classdict = self.__class__.__dict__
        for k in self.keys():
            if not classdict.has_key(k):
                getter = "lambda self: self._getAttrib('%s')" % (k)
                if __debug__:
                    debug("IOH", "Registering property %s for ColumnData" % `k`)
                exec "%s.%s = property(fget=%s)"  % \
                     (self.__class__.__name__, k, getter)
                # TODO!!! Check if it is safe actually here to rely on value of
                #         k in lambda. May be it is treated as continuation and
                #         some local space would override it????
                #setattr(self.__class__,
                #        k,
                #        property(fget=lambda x: x._getAttrib("%s" % k)))
                # it seems to be error-prone due to continuation...


    def _getAttrib(self, key):
        """Return corresponding value if given key is known to current instance

        Is used for automatically added properties to the class.

        :raises: ValueError, if `key` is not known to given instance

        :return: value if `key` is known
        """
        if self.has_key(key):
            return self[key]
        else:
            raise ValueError, "Instance %s has no data about %s" \
                % (`self`, `key`)


    def __str__(self):
        s = self.__class__.__name__
        if len(self.keys())>0:
            s += " %d rows, %d columns [" % \
                 (self.getNRows(), self.getNColumns())
            s += reduce(lambda x, y: x+" %s" % y, self.keys())
            s += "]"
        return s

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

        file_ = open(filename, 'r')

        # make column names, either take header or generate
        if header == True:
            # read first line and split by 'sep'
            hdr = file_.readline().split(sep)
        elif isinstance(header, list):
            hdr = header
        else:
            hdr = [ str(i) for i in xrange(len(file_.readline().split(sep))) ]
            # reset file to not miss the first line
            file_.seek(0)

        # string in lists: one per column
        tbl = [ [] for i in xrange(len(hdr)) ]

        # parse line by line and feed into the lists
        for line in file_:
            # get rid of leading and trailing whitespace
            line = line.strip()
            # ignore empty lines and comment lines
            if not line or line.startswith('#'):
                continue
            l = line.split(sep)

            if not len(l) == len(hdr):
                raise RuntimeError, \
                      "Number of entries in line [%i] does not match number " \
                      "of columns in header [%i]." % (len(l), len(hdr))

            for i, v in enumerate(l):
                if not dtype is None:
                    try:
                        v = dtype(v)
                    except ValueError:
                        warning("Can't convert %s to desired datatype %s." %
                                (`v`, `dtype`) + " Leaving original type")
                tbl[i].append(v)

        # check
        if not len(tbl) == len(hdr):
            raise RuntimeError, "Number of columns read from file does not " \
                                "match the number of header entries."

        # fill dict
        for i, v in enumerate(hdr):
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


    def selectSamples(self, selection):
        """Return new ColumnData with selected samples"""

        data = copy.deepcopy(self)
        for k, v in data.iteritems():
            data[k] = [v[x] for x in selection]

        data._check()
        return data


    def getNColumns(self):
        """Returns the number of columns.
        """
        return len(self.keys())


    def tofile(self, filename, header=True, header_order=None, sep=' '):
        """Write column data to a text file.

        Parameter
        ---------

        - `filename`: Think about it!
        - `header`: If `True` a column header is written, using the column
                    keys. If `False` no header is written.
        - `header_order`: If it is a list of strings, they will be used instead
                          of simply asking for the dictionary keys. However
                          these strings must match the dictionary keys in number
                          and identity. This argument type can be used to
                          determine the order of the columns in the output file.
                          The default value is `None`. In this case the columns
                          will be in an arbitrary order.
        - `sep`: String that is written as a separator between to data columns.
        """
        # XXX do the try: except: dance
        file_ = open(filename, 'w')

        # write header
        if header_order == None:
            col_hdr = self.keys()
        else:
            if not len(header_order) == self.getNColumns():
                raise ValueError, 'Header list does not match number of ' \
                                  'columns.'
            for k in header_order:
                if not self.has_key(k):
                    raise ValueError, 'Unknown key [%s]' % `k`
            col_hdr = header_order

        if header == True:
            file_.write(sep.join(col_hdr) + '\n')

        # for all rows
        for r in xrange(self.getNRows()):
            # get attributes for all keys
            l = [str(self[k][r]) for k in col_hdr]
            # write to file with proper separator
            file_.write(sep.join(l) + '\n')

        file_.close()


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
                          header=False,
                          header_order=['labels', 'chunks'],
                          sep=' ')


    def getNSamples(self):
        """Returns the number of samples in the file.
        """
        return self.getNRows()


    nsamples = property(fget=getNSamples)

