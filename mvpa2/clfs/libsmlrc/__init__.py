# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wraper for the stepwise_regression function for SMLR."""

if __debug__:
    from mvpa2.base import debug
    debug('INIT', 'mvpa2.clfs.libsmlrc')

import numpy as np
import os
import sys

from mvpa2.base import externals
if externals.exists('ctypes', raise_=True):
    import ctypes as C

    from mvpa2.clfs.libsmlrc.ctypes_helper import extend_args, c_darray

    # connect to library that's in this directory
    if sys.platform == 'win32':
        # on windows things get tricky as we compile this lib as an extension
        # so it get a .pyd name suffix instead of .dll
        smlrlib = C.cdll[os.path.join(os.path.dirname(__file__), 'smlrc.pyd')]
    elif sys.platform == 'darwin':
        # look for .so extension on Mac (not .dylib this time)
        smlrlib = C.cdll[os.path.join(os.path.dirname(__file__), 'smlrc.so')]
    else:
        smlrlib = np.ctypeslib.load_library('smlrc', os.path.dirname(__file__))

# wrap the stepwise function
def stepwise_regression(*args):
    func = smlrlib.stepwise_regression
    func.argtypes = [C.c_int, C.c_int, c_darray,
                     C.c_int, C.c_int, c_darray,
                     C.c_int, C.c_int, c_darray,
                     C.c_int, C.c_int, c_darray,
                     C.c_int, C.c_int, c_darray,
                     C.c_int, c_darray,
                     C.c_int, c_darray,
                     C.c_int, c_darray,
                     C.c_int,
                     C.c_int,
                     C.c_double,
                     C.c_float,
                     C.c_float,
                     C.c_int64]
    func.restype = C.c_long

    # get the new arglist
    arglist = extend_args(*args)
    return func(*arglist)

if __debug__:
    debug('INIT', 'mvpa2.clfs.libsmlrc end')

