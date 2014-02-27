# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Some little helper for reading (and writing) common formats from and to
disk."""

__docformat__ = 'restructuredtext'

import numpy as np
import mvpa2.support.copy as copy
from mvpa2.base.dochelpers import enhanced_doc_string
from re import sub
from mvpa2.base import warning

from mvpa2.misc.support import Event

if __debug__:
    from mvpa2.base import debug

__all__ = ['DataReader', 'ColumnData', 'SampleAttributes',
           'SensorLocations', 'XAVRSensorLocations',
           'TuebingenMEGSensorLocations',
           'labels2chunks', 'design2labels']

class DataReader(object):
    """Base class for data readers.

    Every subclass has to put all information into to variable:

    `self._data`: ndarray
        The data array has to have the samples separating dimension along the
        first axis.
    `self._props`: dict
        All other meaningful information has to be stored in a dictionary.

    This class provides two methods (and associated properties) to retrieve
    this information.
    """
    def __init__(self):
        """Initialize base class `DataReader` -- no parameters.
        """
        self._props = {}
        self._data = None


    @property
    def props(self):
        """Return the dictionary with the data properties.
        """
        return self._props


    @property
    def data(self):
        """Return the data array.
        """
        return self._data



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
    def __init__(self, source, header=True, sep=None, headersep=None,
                 dtype=float, skiplines=0):
        """Read data from file into a dictionary.

        Parameters
        ----------
        source : str or dict
          If values is given as a string all data is read from the
          file and additonal keyword arguments can be used to
          customize the read procedure. If a dictionary is passed
          a deepcopy is performed.
        header : bool or list of str
          Indicates whether the column names should be read from the
          first line (`header=True`). If `header=False` unique
          column names will be generated (see class docs). If
          `header` is a python list, it's content is used as column
          header names and its length has to match the number of
          columns in the file.
        sep : str or None
          Separator string. The actual meaning depends on the output
          format (see class docs).
        headersep : str or None
          Separator string used in the header. The actual meaning
          depends on the output format (see class docs).
        dtype : type or list(types)
          Desired datatype(s). Datatype per column get be specified by
          passing a list of types.
        skiplines : int
          Number of lines to skip at the beginning of the file.
        """
        # init base class
        dict.__init__(self)

        # intialize with default
        self._header_order = None

        if isinstance(source, basestring):
            self._from_file(source, header=header, sep=sep, headersep=headersep,
                           dtype=dtype, skiplines=skiplines)

        elif isinstance(source, dict):
            for k, v in source.iteritems():
                self[k] = v
            # check data integrity
            self._check()

        else:
            raise ValueError, 'Unkown source for ColumnData [%r]' \
                              % type(source)

        # generate missing properties for each item in the header
        classdict = self.__class__.__dict__
        for k in self.keys():
            if not classdict.has_key(k):
                getter = "lambda self: self._get_attrib('%s')" % (k)
                # Sanitarize the key, substitute ' []' with '_'
                k_ = sub('[[\] ]', '_', k)
                # replace multipe _s
                k_ = sub('__+', '_', k_)
                # remove quotes
                k_ = sub('["\']', '', k_)
                if __debug__:
                    debug("IOH", "Registering property %s for ColumnData key %s"
                          % (k_, k))
                # make sure to import class directly into local namespace
                # otherwise following does not work for classes defined
                # elsewhere
                exec 'from %s import %s' % (self.__module__,
                                            self.__class__.__name__)
                exec "%s.%s = property(fget=%s)" % \
                     (self.__class__.__name__, k_, getter)
                # TODO!!! Check if it is safe actually here to rely on value of
                #         k in lambda. May be it is treated as continuation and
                #         some local space would override it????
                #setattr(self.__class__,
                #        k,
                #        property(fget=lambda x: x._get_attrib("%s" % k)))
                # it seems to be error-prone due to continuation...


    __doc__ = enhanced_doc_string('ColumnData', locals())


    ##REF: Name was automagically refactored
    def _get_attrib(self, key):
        """Return corresponding value if given key is known to current instance

        Is used for automatically added properties to the class.

        Raises
        ------
        ValueError:
          If `key` is not known to given instance

        Returns
        -------
        Value if `key` is known
        """
        if self.has_key(key):
            return self[key]
        else:
            raise ValueError, "Instance %r has no data about %r" \
                % (self, key)


    def __str__(self):
        s = self.__class__.__name__
        if len(self.keys()) > 0:
            s += " %d rows, %d columns [" % \
                 (self.nrows, self.ncolumns)
            s += reduce(lambda x, y: x + " %s" % y, self.keys())
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


    def _from_file(self, filename, header, sep, headersep,
                   dtype, skiplines):
        """Loads column data from file -- clears object first.
        """
        # make a clean table
        self.clear()

        with open(filename, 'r') as file_:

            self._header_order = None

            [ file_.readline() for x in range(skiplines) ]
            """Simply skip some lines"""
            # make column names, either take header or generate
            if header == True:
                # read first line and split by 'sep'
                hdr = file_.readline().split(headersep)
                # remove bogus empty header titles
                hdr = [ x for x in hdr if len(x.strip()) ]
                self._header_order = hdr
            elif isinstance(header, list):
                hdr = header
            else:
                hdr = [ str(i) for i in xrange(len(file_.readline().split(sep))) ]
                # reset file to not miss the first line
                file_.seek(0)
                [ file_.readline() for x in range(skiplines) ]


            # string in lists: one per column
            tbl = [ [] for i in xrange(len(hdr)) ]

            # store whether dtype should be determined automagically
            auto_dtype = dtype is None

            # do per column dtypes
            if not isinstance(dtype, list):
                dtype = [dtype] * len(hdr)

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
                    if not dtype[i] is None:
                        try:
                            v = dtype[i](v)
                        except ValueError:
                            warning("Can't convert %r to desired datatype %r." %
                                    (v, dtype) + " Leaving original type")
                    tbl[i].append(v)

            if auto_dtype:
                attempt_convert_dtypes = (int, float)

                for i in xrange(len(tbl)):
                    values = tbl[i]

                    for attempt_convert_dtype in attempt_convert_dtypes:
                        try:
                            values = map(attempt_convert_dtype, values)
                            tbl[i] = values
                            break
                        except:
                            continue

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
                raise ValueError, 'Unknown key [%r].' % (k,)
            if not isinstance(v, list):
                raise ValueError, 'Can only merge list data, but got [%r].' \
                                  % type(v)
            # now it seems to be ok
            # XXX check for datatype?
            self[k] += v

        # look for problems, like columns present in self, but not in other
        self._check()

        return self


    ##REF: Name was automagically refactored
    def select_samples(self, selection):
        """Return new ColumnData with selected samples"""

        data = copy.deepcopy(self)
        for k, v in data.iteritems():
            data[k] = [v[x] for x in selection]

        data._check()
        return data

    @property
    def ncolumns(self):
        """Returns the number of columns.
        """
        return len(self.keys())


    def tofile(self, filename, header=True, header_order=None, sep=' '):
        """Write column data to a text file.

        Parameters
        ----------
        filename : str
          Target filename
        header : bool, optional
          If `True` a column header is written, using the column
          keys. If `False` no header is written.
        header_order : None or list of str
          If it is a list of strings, they will be used instead
          of simply asking for the dictionary keys. However
          these strings must match the dictionary keys in number
          and identity. This argument type can be used to
          determine the order of the columns in the output file.
          The default value is `None`. In this case the columns
          will be in an arbitrary order.
        sep : str, optional
          String that is written as a separator between to data columns.
        """
        # XXX do the try: except: dance
        file_ = open(filename, 'w')

        # write header
        if header_order == None:
            if self._header_order is None:
                col_hdr = self.keys()
            else:
                # use stored order + newly added keys at the last columns
                col_hdr = self._header_order + \
                          list(set(self.keys()).difference(
                                                set(self._header_order)))
        else:
            if not len(header_order) == self.ncolumns:
                raise ValueError, 'Header list does not match number of ' \
                                  'columns.'
            for k in header_order:
                if not self.has_key(k):
                    raise ValueError, 'Unknown key [%r]' % (k,)
            col_hdr = header_order

        if header == True:
            file_.write(sep.join(col_hdr) + '\n')

        # for all rows
        for r in xrange(self.nrows):
            # get attributes for all keys
            l = [str(self[k][r]) for k in col_hdr]
            # write to file with proper separator
            file_.write(sep.join(l) + '\n')

        file_.close()


    @property
    def nrows(self):
        """Returns the number of rows.
        """
        # no data no rows (after Bob Marley)
        if not len(self.keys()):
            return 0
        # otherwise first key is as good as any other
        else:
            return len(self[self.keys()[0]])



class SampleAttributes(ColumnData):
    """Read and write PyMVPA sample attribute definitions from and to text
    files.
    """
    def __init__(self, source, literallabels=True, header=None):
        """Read PyMVPA sample attributes from disk.

        Parameters
        ----------
        source : str
          Filename of an atrribute file
        literallabels : bool, optional
          Either labels are given as literal strings
        header : None or bool or list of str
          If None, ['targets', 'chunks'] is assumed. Otherwise the same
          behavior as of `ColumnData`
        """
        if literallabels:
            if header is None:
                dtypes = [str, float]
            else:
                dtypes = None
        else:
            dtypes = float

        if header is None:
            header = ['targets', 'chunks']
        ColumnData.__init__(self, source,
                            header=header,
                            sep=None, dtype=dtypes)


    def tofile(self, filename):
        """Write sample attributes to a text file.
        """
        ColumnData.tofile(self, filename,
                          header=False,
                          header_order=['targets', 'chunks'],
                          sep=' ')


    @property
    def nsamples(self):
        """Returns the number of samples in the file.
        """
        return self.nrows



class SensorLocations(ColumnData):
    """Base class for sensor location readers.

    Each subclass should provide x, y, z coordinates via the `pos_x`, `pos_y`,
    and `pos_z` attrbibutes.

    Axes should follow the following convention:

      x-axis: left -> right
      y-axis: anterior -> posterior
      z-axis: superior -> inferior
    """
    def __init__(self, *args, **kwargs):
        """Pass arguments to ColumnData.
        """
        ColumnData.__init__(self, *args, **kwargs)


    def locations(self):
        """Get the sensor locations as an array.

        Returns
        -------
        (nchannels x 3) array with coordinates in (x, y, z)
        """
        return np.array((self.pos_x, self.pos_y, self.pos_z)).T



class XAVRSensorLocations(SensorLocations):
    """Read sensor location definitions from a specific text file format.

    File layout is assumed to be 5 columns:

      1. sensor name
      2. some useless integer
      3. position on x-axis
      4. position on y-axis
      5. position on z-axis
    """
    def __init__(self, source):
        """Read sensor locations from file.

        Parameters
        ----------
        source : filename of an attribute file
        """
        SensorLocations.__init__(
            self, source,
            header=['names', 'some_number', 'pos_x', 'pos_y', 'pos_z'],
            sep=None, dtype=[str, int, float, float, float])


class TuebingenMEGSensorLocations(SensorLocations):
    """Read sensor location definitions from a specific text file format.

    File layout is assumed to be 7 columns::

      1:   sensor name
      2:   position on y-axis
      3:   position on x-axis
      4:   position on z-axis
      5-7: same as 2-4, but for some outer surface thingie.

    Note that x and y seem to be swapped, ie. y as defined by SensorLocations
    conventions seems to be first axis and followed by x.

    Only inner surface coordinates are reported by `locations()`.
    """
    def __init__(self, source):
        """Read sensor locations from file.

        Parameters
        ----------
        source : filename of an attribute file
        """
        SensorLocations.__init__(
            self, source,
            header=['names', 'pos_y', 'pos_x', 'pos_z',
                    'pos_y2', 'pos_x2', 'pos_z2'],
            sep=None, dtype=[str, float, float, float, float, float, float])


def design2labels(columndata, baseline_label=0,
                  func=lambda x: x > 0.0):
    """Helper to convert design matrix into a list of labels

    Given a design, assign a single label to any given sample

    TODO: fix description/naming

    Parameters
    ----------
    columndata : ColumnData
      Attributes where each known will be considered as a separate
      explanatory variable (EV) in the design.
    baseline_label
      What label to assign for samples where none of EVs was given a value
    func : functor
      Function which decides either a value should be considered

    Returns
    -------
    list of labels which are taken from column names in ColumnData and
    baseline_label

    """
    # doing it simple naive way but it should be of better control if
    # we decide to process columndata with non-numeric entries etc
    keys = columndata.keys()
    labels = []
    for row in xrange(columndata.nrows):
        entries = [ columndata[key][row] for key in keys ]
        # which entries get selected
        selected = [ x for x in zip(keys, entries) if func(x[1]) ]
        nselected = len(selected)

        if nselected > 1:
            # if there is more than a single one -- we are in problem
            raise ValueError, "Row #%i with items %s has multiple entries " \
                  "meeting the criterion. Cannot decide on the label" % \
                  (row, entries)
        elif nselected == 1:
            label = selected[0][0]
        else:
            label = baseline_label
        labels.append(label)
    return labels


__known_chunking_methods = {
    'alllabels': 'Each chunk must contain instances of all labels'
    }

def labels2chunks(labels, method="alllabels", ignore_labels=None):
    """TOBE ASSIGNED BELOW
    """
    chunks = []
    if ignore_labels is None:
        ignore_labels = []
    alllabels = set(labels).difference(set(ignore_labels))
    if method == 'alllabels':
        seenlabels = set()
        lastlabel = None
        chunk = 0
        for label in labels:
            if label != lastlabel:
                if seenlabels == alllabels:
                    chunk += 1
                    seenlabels = set()
                lastlabel = label
                if not label in ignore_labels:
                    seenlabels.update([label])
            chunks.append(chunk)
        chunks = np.array(chunks)
        # fix up a bit the trailer
        if seenlabels != alllabels:
            chunks[chunks == chunk] = chunk - 1
        chunks = list(chunks)
    else:
        errmsg = "Unknown method to derive chunks is requested. Known are:\n"
        for method, descr in __known_chunking_methods.iteritems():
            errmsg += "  %s : %s\n" % (method, descr)
        raise ValueError, errmsg
    return chunks

labels2chunks.__doc__ = \
 """Automagically decide on chunks based on labels

    Parameters
    ----------
    labels
      labels to base chunking on
    method : str
      codename for method to use. Known are %s
    ignore_labels : list of str
      depends on the method. If method `alllabels`, then don't
      seek for such labels in chunks. E.g. some 'reject' samples

    :rtype: list
    """ % __known_chunking_methods.keys()
