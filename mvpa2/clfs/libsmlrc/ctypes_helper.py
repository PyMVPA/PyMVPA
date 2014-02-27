# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helpers for wrapping C libraries with ctypes."""

import numpy as np
import ctypes as C

# define an array type to help with wrapping
c_darray = np.ctypeslib.ndpointer(dtype=np.float64, flags='ALIGNED,CONTIGUOUS')
c_larray = np.ctypeslib.ndpointer(dtype=np.int64, flags='ALIGNED,CONTIGUOUS')
c_farray = np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED,CONTIGUOUS')
c_iarray = np.ctypeslib.ndpointer(dtype=np.int32, flags='ALIGNED,CONTIGUOUS')

def extend_args(*args):
    """Turn ndarray arguments into dims and arrays."""
    arglist = []
    for arg in args:
        if isinstance(arg, np.ndarray):
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
    np.float64: C.c_double,
    np.float32: C.c_float,
    np.int64: C.c_int64,
    np.int32: C.c_int32}

def process_args(*args):
    """Turn ndarray arguments into dims and array pointers for calling
    a ctypes-wrapped function."""
    arglist = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            # add the dimensions
            arglist.extend(arg.shape)

            # add the pointer to the ndarray
            arglist.append(arg.ctypes.data_as(
                C.POINTER(typemap[arg.dtype.type])))
        else:
            # just append the arg
            arglist.append(arg)

    return arglist

def get_argtypes(*args):
    argtypes = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            # add the dimensions
            argtypes.extend([C.c_int]*len(arg.shape))

            # add the pointer to the ndarray
            argtypes.append(np.ctypeslib.ndpointer(dtype=arg.dtype))
        else:
            # try and figure out the type
            argtypes.append(arg)
    return argtypes


