#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helpers for wrapping C libraries with ctypes."""

import numpy as N
import ctypes as C

# define an array type to help with wrapping
c_darray = N.ctypeslib.ndpointer(dtype=N.float64)
c_larray = N.ctypeslib.ndpointer(dtype=N.int64)
c_farray = N.ctypeslib.ndpointer(dtype=N.float32)
c_iarray = N.ctypeslib.ndpointer(dtype=N.int32)

def extend_args(*args):
    """Turn ndarray arguments into dims and arrays."""
    arglist = []
    for arg in args:
        if isinstance(arg,N.ndarray):
            # add the dimensions
            arglist.extend(arg.shape)

        # just append the arg
        arglist.append(arg)

    return arglist

#############################################################
# I'm not sure the rest is helpful, but I'll keep it for now.
#############################################################

# incomplete type conversion
typemap = {
    N.float64: C.c_double,
    N.float32: C.c_float,
    N.int64: C.c_long,
    N.int32: C.c_int}

def process_args(*args):
    """Turn ndarray arguments into dims and array pointers for calling
    a ctypes-wrapped function."""
    arglist = []
    for arg in args:
        if isinstance(arg,N.ndarray):
            # add the dimensions
            arglist.extend(arg.shape)

            # add the pointer to the ndarray
            arglist.append(arg.ctypes.data_as(C.POINTER(typemap[arg.dtype.type])))
        else:
            # just append the arg
            arglist.append(arg)

    return arglist

def get_argtypes(*args):
    argtypes = []
    for arg in args:
        if isinstance(arg,N.ndarray):
            # add the dimensions
            arglist.extend([C.c_int]*len(arg.shape))

            # add the pointer to the ndarray
            arglist.append(N.ctypeslib.ndpointer(dtype=arg.dtype))
        else:
            # try and figure out the type
            arglist.append(arg)
    return argtypes


