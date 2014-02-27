# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for complementary unittest-ing tools"""

import numpy as np

from mvpa2.base.externals import versions
from mvpa2.testing.tools import *
from mvpa2.testing.sweep import *

import mvpa2.tests as mvtests

def test_assert_objectarray_equal():
    if versions['numpy'] < '1.4':
        raise SkipTest("Skipping because of known segfaults with numpy < 1.4")
    # explicit dtype so we could test with numpy < 1.6
    a = np.array([np.array([0, 1]), np.array(1)], dtype=object)
    b = np.array([np.array([0, 1]), np.array(1)], dtype=object)

    # they should be ok for both types of comparison
    for strict in True, False:
        # good with self
        assert_objectarray_equal(a, a, strict=strict)
        # good with a copy
        assert_objectarray_equal(a, a.copy(), strict=strict)
        # good while operating with an identical one
        # see http://projects.scipy.org/numpy/ticket/2117
        assert_objectarray_equal(a, b, strict=strict)

    # now check if we still fail for a good reason
    for value_equal, b in (
            (False, np.array(1)),
            (False, np.array([1])),
            (False, np.array([np.array([0, 1]), np.array((1, 2))], dtype=object)),
            (False, np.array([np.array([0, 1]), np.array(1.1)], dtype=object)),
            (True, np.array([np.array([0, 1]), np.array(1.0)], dtype=object)),
            (True, np.array([np.array([0, 1]), np.array(1, dtype=object)], dtype=object)),
            ):
        assert_raises(AssertionError, assert_objectarray_equal, a, b)
        if value_equal:
            # but should not raise for non-default strict=False
            assert_objectarray_equal(a, b, strict=False)
        else:
            assert_raises(AssertionError, assert_objectarray_equal, a, b, strict=False)

# Set of basic smoke tests for tests collectors/runners
def test_tests_run():
    ok_(len(mvtests.collect_unit_tests()) > 10)
    ok_(len(mvtests.collect_nose_tests()) > 10)
    ok_(len(mvtests.collect_test_suites()) > 10)
    mvtests.run(limit=[])

@sweepargs(suffix=['', 'customsuffix'])
@sweepargs(prefix=['', 'customprefix'])
#@sweepargs(mkdir=(True, False))
def test_with_tempfile(suffix, prefix): #, mkdir):
    files = []

    @with_tempfile(suffix, prefix) #, mkdir=mkdir)
    def testf(f):
        assert_false(os.path.exists(f)) # not yet
        if suffix:
            assert_true(f.endswith(suffix))
        if prefix:
            assert_true(os.path.basename(f).startswith(prefix))
        #assert_true(os.path.isdir(f) == dir_)
        # make sure it is writable
        with open(f, 'w') as f_:
            f_.write('load')
            files.append(f)
        assert_true(os.path.exists(f)) # should be there
        # and we should be able to create a bunch of those with other suffixes
        with open(f + '1', 'w') as f_:
            f_.write('load')
            files.append(f + '1')

    testf()
    # now we need to figure out what file was actually
    assert_equal(len(files), 2)
    assert_false(os.path.exists(files[0]))
    assert_false(os.path.exists(files[1]))