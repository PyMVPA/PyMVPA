#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Wraper for the stepwise_regression function for SMLR."""

if __debug__:
    from mvpa.misc import debug
    debug('INIT', 'mvpa.clfs.libsmlr')

import numpy as N
import ctypes as C
import os

from mvpa.clfs.libsmlr.ctypes_helper import extend_args, c_darray

# connect to library that's in this directory
smlrlib = N.ctypeslib.load_library('smlrc', os.path.dirname(__file__))

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
    debug('INIT', 'mvpa.clfs.libsmlr end')

