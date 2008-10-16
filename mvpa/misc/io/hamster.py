#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
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

class Hamster(dict):
    """Simple container class which is derived from the dictionary

    It is capable of storing itself in a file, or loading from a file
    (using cPickle + zlib tandem). Any serializable object can be
    bound to a hamster to be stored.

    To undig burried hamster use Hamster(filename)
    """

    def __new__(cls, *args, **kwargs):
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
            return dict.__new__(cls, *args, **kwargs)


    def __init__(self, *args, **kwargs):
        """Initialize Hamster.

        Providing a single parameter string would treat it as a
        filename from which to undig the data. Otherwise all the
        parameters are equivalent to the ones of dict
        """
        if len(args) == 1 and isinstance(args[0], basestring):
            # it was a filename
            args = args[1:]
        dict.__init__(self, *args, **kwargs)


    def dump(self, filename):
        """Bury the hamster into the file
        """
        if __debug__:
            debug('IOH', 'Buring hamster into %s' % filename)
        f = gzip.open(filename, 'w')
        cPickle.dump(self, f)
        f.close()


    def __setattr__(self, index, value):
        if index[0] == '_':
            _o_seta_(self, index, value)

        _dict_ = _o_geta_(self, '__dict__')

        if index in _dict_ or hasattr(self, index):
            _o_seta_(self, index, value)
        else:
            _d_seti_(self, index, value)


    def __getattribute__(self, index):
        if index[0] == '_' or index == 'has_key':
            return dict.__getattribute__(self, index)
        if self.has_key(index):
            return _d_geti_(self, index)
        else:
            return _o_geta_(self, index)
