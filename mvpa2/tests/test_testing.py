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

from os.path import exists

from mvpa2.base.externals import versions
from mvpa2.testing.tools import *
from mvpa2.testing.sweep import *
from mvpa2.datasets.base import Dataset

from mvpa2.testing.tools import assert_datasets_almost_equal, \
    assert_raises, assert_datasets_equal

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
    ok_(len(mvtests.collect_test_suites(instantiate=False)) > 10)
    mvtests.run(limit=[])



@sweepargs(suffix=['', 'customsuffix'])
@sweepargs(prefix=['', 'customprefix'])
# @sweepargs(mkdir=(True, False))
def test_with_tempfile(suffix, prefix):  # , mkdir):
    files = []

    @with_tempfile(suffix, prefix)  # , mkdir=mkdir)
    def testf(f):
        assert_false(os.path.exists(f))  # not yet
        if suffix:
            assert_true(f.endswith(suffix))
        if prefix:
            assert_true(os.path.basename(f).startswith(prefix))
        # assert_true(os.path.isdir(f) == dir_)
        # make sure it is writable
        with open(f, 'w') as f_:
            f_.write('load')
            files.append(f)
        assert_true(os.path.exists(f))  # should be there
        # and we should be able to create a bunch of those with other suffixes
        with open(f + '1', 'w') as f_:
            f_.write('load')
            files.append(f + '1')

    testf()
    # now we need to figure out what file was actually
    assert_equal(len(files), 2)
    assert_false(os.path.exists(files[0]))
    assert_false(os.path.exists(files[1]))



@nodebug(['ID_IN_REPR', 'DS_ID'])
@with_tempfile('.hdf5')
def test_generate_testing_fmri_dataset(tempfile):
    skip_if_no_external('nibabel')
    skip_if_no_external('h5py')

    from mvpa2.base.hdf5 import h5load
    from mvpa2.testing.regress import generate_testing_fmri_dataset

    ds, filename = generate_testing_fmri_dataset(tempfile)
    try:
        import IPython
        assert_true(externals.exists('ipython'))
    except:
        assert_false(externals.exists('ipython'))
        assert('ipython' not in ds.a.versions)
    assert_equal(tempfile, filename)
    assert_true(exists(tempfile))
    ds_reloaded = h5load(tempfile)
    assert_datasets_equal(ds, ds_reloaded, ignore_a={'wtf'})



@sweepargs(attribute=['samples', 'sa', 'fa', 'a'])
@sweepargs(digits=[None, 1, 2, 3])
def test_assert_datasets_almost_equal(digits, attribute):
    samples = np.random.standard_normal((2, 5))
    args = dict(sa=dict(targets=np.asarray([1., 2])),
                fa=dict(ids=np.asarray([0., 1, 2, 3, 4])),
                a=dict(a_value=[66]))

    ds = Dataset(samples=samples, **args)

    def negate_assert(f):
        def raiser(*args, **kwargs):
            assert_raises(AssertionError, f, *args, **kwargs)

        return raiser

    assert_datasets_not_almost_equal = negate_assert(assert_datasets_almost_equal)
    assert_datasets_not_equal = negate_assert(assert_datasets_equal)

    def change_attribute(name, how_much):
        # change a single attribute in samples, a, fa, or sa.
        ds2 = ds.copy(deep=True)
        attr = ds2.__dict__[name]
        if name == 'samples':
            value = attr
        else:
            for key in attr:
                break

            value = attr[key].value

        value[0] += how_much

        return ds2

    def remove_attribute(name):
        ds2 = ds.copy(deep=True)
        attr = ds2.__dict__[name]
        for key in attr.keys():
            attr.pop(key)
        return ds2

    if digits is None:
        ds2 = change_attribute(attribute, 0)
        assert_datasets_equal(ds, ds2)
    else:
        ds2 = change_attribute(attribute, .5 * 10 ** -digits)
        assert_datasets_not_equal(ds, ds2)
        assert_datasets_not_almost_equal(ds, ds2, decimal=digits + 1)

        if attribute == 'samples':
            assert_datasets_almost_equal(ds, ds2, decimal=digits)
        else:
            assert_datasets_not_almost_equal(ds, ds2, decimal=digits - 1)

            # test ignore_ options
            args = {('ignore_' + attribute): args[attribute].keys()}
            assert_datasets_equal(ds, ds2, **args)
            assert_datasets_almost_equal(ds, ds2, **args)

            ds3 = remove_attribute(attribute)
            assert_datasets_not_equal(ds, ds3)
            assert_datasets_not_almost_equal(ds, ds3)
