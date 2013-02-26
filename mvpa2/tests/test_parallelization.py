# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for parallelization"""

import numpy as np

import os
import tempfile

from mvpa2.testing import *
from mvpa2.misc.parallelization import *
from mvpa2.base import externals

import tempfile

class ParallelizationTests(unittest.TestCase):
    """Test for parallelization"""

    def test_simple(self):
        x = [1, -1, -3]

        def get_proc_func(backend):
            if backend == 'hdf5':
                def f(x):
                    _, fn = tempfile.mkstemp('par', '.h5py')
                    from mvpa2.base.hdf5 import h5save
                    h5save(fn, abs(x))
                    return fn
                return f
            else:
                return abs

        merge_func = sum
        fold_func = lambda x, y:x + y

        parallelizers = [PProcessParallelizer,
                         SingleThreadParallelizer,
                         get_best_parallelizer()]

        for parallelizer in parallelizers:
            not_ok = parallelizer == PProcessParallelizer and \
                        not externals.exists('pprocess')
            if not_ok:
                assert_false(parallelizer.is_available())
                continue

            for backend in ['native'] + \
                    (['hdf5'] if externals.exists('h5py') else []):

                proc_func = get_proc_func(backend)

                for i in xrange(2):
                    if i == 0:
                        f = parallelizer(proc_func, merge_func=merge_func, results_backend=backend)
                    else:
                        f = parallelizer(proc_func, fold_func=fold_func, results_backend=backend)

            assert_true(f.is_available())
            y = f(x)
            assert_equal(y, 5)


def suite():
    """Create the suite"""
    return unittest.makeSuite(ParallelizationTests)

if __name__ == '__main__':
    import runner
