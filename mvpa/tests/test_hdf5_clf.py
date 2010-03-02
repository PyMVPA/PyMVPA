# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for storage of classifiers in HDF5'''


from mvpa.testing import *
skip_if_no_external('h5py')

import numpy as np
from mvpa.testing.datasets import datasets
from mvpa.clfs.warehouse import clfswh, regrswh

import os
import tempfile

from mvpa.base.hdf5 import h5save, h5load, obj2hdf


@sweepargs(lrn=clfswh[:] + regrswh[:])
def test_h5py_clfs(lrn):
    f = tempfile.NamedTemporaryFile()
    try:
        h5save(f.name, lrn)
    except Exception, e:
        raise AssertionError, \
              "Failed to store due to %r" % (e,)

    try:
        lrn_ = h5load(f.name)
    except Exception, e:
        raise AssertionError, \
              "Failed to load due to %r" % (e,)
