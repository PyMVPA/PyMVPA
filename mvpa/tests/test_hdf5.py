# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for HDF5 converter'''

import numpy as N
import random
import shutil
import tempfile
import os
import h5py

from numpy.testing import assert_array_equal
from nose.tools import ok_, assert_raises, assert_false, assert_equal, \
        assert_true

from mvpa.base.externals import exists
from mvpa.base.hdf5 import obj2hdf, hdf2obj
from tests_warehouse import *


def test_h5py_datasets():
    tempdir = tempfile.mkdtemp()

    # store the whole datasets warehouse in one hdf5 file
    hdf = h5py.File(os.path.join(tempdir, 'myhdf5.hdf5'), 'w')
    for d in datasets:
        obj2hdf(hdf, datasets[d], d)
    hdf.close()

    hdf = h5py.File(os.path.join(tempdir, 'myhdf5.hdf5'), 'r')
    rc_ds = {}
    for d in hdf:
        rc_ds[d] = hdf2obj(hdf[d])
    hdf.close()

    # global checks
    assert_equal(len(datasets), len(rc_ds))
    assert_equal(sorted(datasets.keys()), sorted(rc_ds.keys()))
    # check each one
    for d in datasets:
        ds = datasets[d]
        ds2 = rc_ds[d]
        assert_array_equal(ds.samples, ds2.samples)
        # we can check all sa and fa attrs
        for attr in ds.sa:
            assert_array_equal(ds.sa[attr].value, ds2.sa[attr].value)
        for attr in ds.fa:
            assert_array_equal(ds.fa[attr].value, ds2.fa[attr].value)
        # with datasets attributes it is more difficult, but we'll do some
        assert_equal(len(ds.a), len(ds2.a))
        assert_equal(sorted(ds.a.keys()), sorted(ds2.a.keys()))
        if 'mapper' in ds.a:
            # since we have no __equal__ do at least some comparison
            assert_equal(repr(ds.a.mapper), repr(ds2.a.mapper))

    #cleanup temp dir
    shutil.rmtree(tempdir, ignore_errors=True)
