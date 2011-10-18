# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helper for simple storage facility via cPickle and optionally zlib"""

__docformat__ = 'restructuredtext'

import os

from mvpa2.base import externals

_d_geti_ = dict.__getitem__
_d_seti_ = dict.__setitem__

_o_geta_ = dict.__getattribute__
_o_seta_ = dict.__setattr__

if externals.exists('cPickle', raise_=True) and \
   externals.exists('gzip', raise_=True):
    import cPickle, gzip

if __debug__:
    from mvpa2.base import debug

class Hamster(object):
    """Simple container class with basic IO capabilities.

    It is capable of storing itself in a file, or loading from a file using
    cPickle (optionally via zlib from compressed files). Any serializable
    object can be bound to a hamster to be stored.

    To undig burried hamster use Hamster(filename). Here is an example:

    >>> import numpy as np
    >>> import tempfile
    >>> h = Hamster(bla='blai')
    >>> h.boo = np.arange(5)
    >>> tmp = tempfile.NamedTemporaryFile()
    >>> h.dump(tmp.name)
    ...
    >>> h = Hamster(tmp.name)

    Since Hamster introduces methods `dump`, `asdict` and property
    'registered', those names cannot be used to assign an attribute,
    nor provided in among constructor arguments.
    """

    __ro_attr = set(object.__dict__.keys() +
                    ['dump', 'registered', 'asdict'])
    """Attributes which come with being an object"""

    def __new__(cls, *args, **kwargs):
        if len(args) > 0:
            if len(kwargs) > 0:
                raise ValueError, \
                      "Do not mix positional and keyword arguments. " \
                      "Use a single positional argument -- filename, " \
                      "or any number of keyword arguments, without having " \
                      "filename specified"
            if len(args) == 1 and isinstance(args[0], basestring):
                filename = args[0]
                args = args[1:]
                if __debug__:
                    debug('IOH', 'Undigging hamster from %s' % filename)
                # compressed or not -- that is the question
                if filename.endswith('.gz'):
                    f = gzip.open(filename)
                else:
                    f = open(filename)
                result = cPickle.load(f)
                if not isinstance(result, Hamster):
                    warning("Loaded other than Hamster class from %s" % filename)
                return result
            else:
                raise ValueError, "Hamster accepts only a single positional " \
                      "argument and it must be a filename. Got %d " \
                      "arguments" % (len(args),)
        else:
            return object.__new__(cls)


    def __init__(self, *args, **kwargs):
        """Initialize Hamster.

        Providing a single parameter string would treat it as a
        filename from which to undig the data. Otherwise all keyword
        parameters are assigned into the attributes of the object.
        """
        if len(args) > 0:
            if len(args) == 1 and isinstance(args[0], basestring):
                # it was a filename
                args = args[1:]
            else:
                raise RuntimeError, "Should not get here"

        # assign provided attributes
        for k,v in kwargs.iteritems():
            setattr(self, k, v)

        object.__init__(self)


    def dump(self, filename, compresslevel='auto'):
        """Bury the hamster into the file

        Parameters
        ----------
        filename : str
          Name of the target file. When writing to a compressed file the
          filename gets a '.gz' extension if not already specified. This
          is necessary as the constructor uses the extension to decide
          whether it loads from a compressed or uncompressed file.
        compresslevel : 'auto' or int
          Compression level setting passed to gzip. When set to
          'auto', if filename ends with '.gz' `compresslevel` is set
          to 5, 0 otherwise.  However, when `compresslevel` is set to
          0 gzip is bypassed completely and everything is written to
          an uncompressed file.
        """
        if compresslevel == 'auto':
            compresslevel = (0, 5)[int(filename.endswith('.gz'))]
        if compresslevel > 0 and not filename.endswith('.gz'):
            filename += '.gz'
        if __debug__:
            debug('IOH', 'Burying hamster into %s' % filename)
        if compresslevel == 0:
            f = open(filename, 'w')
        else:
            f = gzip.open(filename, 'w', compresslevel)
        cPickle.dump(self, f)
        f.close()


    def __repr__(self):
        reg_attr = ["%s=%s" % (k, repr(getattr(self, k)))
                    for k in self.registered]
        return "%s(%s)" % (self.__class__.__name__,
                           ", ".join(reg_attr))

    # ??? actually seems to be ugly
    #def __str__(self):
    #    registered = self.registered
    #    return "%s with %d elements: %s" \
    #           % (self.__class__.__name__,
    #              len(registered),
    #              ", ".join(self.registered))

    @property
    def registered(self):
        """List registered attributes.
        """
        reg_attr = [k for k in self.__dict__.iterkeys()
                    if not k in self.__ro_attr]
        reg_attr.sort()
        return reg_attr


    def __setattr__(self, k, v):
        """Just to prevent resetting read-only attributes, such as methods
        """
        if k in self.__ro_attr:
            raise ValueError, "'%s' object attribute '%s' is read-only" \
                  % (self.__class__.__name__, k)
        object.__setattr__(self, k, v)


    def asdict(self):
        """Return registered data as dictionary
        """
        return dict([(k, getattr(self, k))
                     for k in self.registered])
