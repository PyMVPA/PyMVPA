#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helper for simple storage facility via cPickle and zlib"""

__docformat__ = 'restructuredtext'

import os

from mvpa.base import externals
externals.exists('cPickle', raiseException=True)
externals.exists('gzip', raiseException=True)

_d_geti_ = dict.__getitem__
_d_seti_ = dict.__setitem__

_o_geta_ = dict.__getattribute__
_o_seta_ = dict.__setattr__

import cPickle, gzip

if __debug__:
    from mvpa.base import debug

class Hamster(object):
    """Simple container class with basic IO capabilities.

    It is capable of storing itself in a file, or loading from a file
    (using cPickle + zlib tandem). Any serializable object can be
    bound to a hamster to be stored.

    To undig burried hamster use Hamster(filename). Here is an example:

      >>> h = Hamster(bla='blai')
      >>> h.boo = N.arange(5)
      >>> h.dump(filename)
      ...
      >>> h = Hamster(filename)

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
                f = gzip.open(filename)
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


    def dump(self, filename):
        """Bury the hamster into the file
        """
        if __debug__:
            debug('IOH', 'Burying hamster into %s' % filename)
        f = gzip.open(filename, 'w')
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
