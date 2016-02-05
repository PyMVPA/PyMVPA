# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""A Collection of tools found useful in unittests.

Primarily the ones from nose.tools
"""
__docformat__ = 'restructuredtext'

import glob, os, sys, shutil
from os.path import join as pathjoin
import tempfile
import unittest
from contextlib import contextmanager

import numpy as np

import mvpa2
from mvpa2.base import externals, warning
from mvpa2.base.dochelpers import strip_strid

if __debug__:
    from mvpa2.base import debug

if externals.exists('nose'):
    # We use nose now
    from nose import SkipTest
    from nose.tools import (
        ok_, eq_,
        # Asserting (pep8-ed from unittest)
        assert_true, assert_false, assert_raises,
        assert_equal, assert_equals, assert_not_equal, assert_not_equals,
        assert_in, assert_not_in,
        # Decorators
        timed, with_setup, raises, istest, nottest, make_decorator)
else:
    # Lets make it possible to import testing.tools even if nose is
    # NA, and run unittests which do not require nose yet
    def _need_nose(*args, **kwargs):
        """Catcher for unittests requiring nose functionality
        """
        raise unittest.TestCase.failureException(
            "Unittest requires nose testing framework")


    ok_ = eq_ = assert_true = assert_false = assert_raises = \
        assert_equal = assert_equals = assert_not_equal = asserte_not_equals = \
        timed = with_setup = raises = istest = nottest = make_decorator = _need_nose

    class SkipTest(Exception):
        """Raise this exception to mark a test as skipped.
        """
        pass

# Some pieces are useful from numpy.testing
from numpy.testing import (
    assert_almost_equal, assert_approx_equal,
    assert_array_almost_equal, assert_array_equal, assert_array_less,
    assert_string_equal)

from numpy.testing.decorators import skipif

def assert_array_lequal(x, y):
    assert_array_less(-y, -x)



def assert_dict_keys_equal(x, y):
    assert_equal(set(x.keys()), set(y.keys()))



def assert_reprstr_equal(x, y):
    """Whenever comparison fails otherwise, we might revert to compare those"""
    if __debug__ and ("ID_IN_REPR" in debug.active or "DS_ID" in debug.active):
        repr_ = lambda x: strip_strid(repr(x))
        str_ = lambda x: strip_strid(str(x))
    else:
        repr_, str_ = repr, str
    assert_equal(repr_(x), repr_(y))
    assert_equal(str_(x), str_(y))



def assert_collections_equal(x, y, ignore=None):
    # Seems to cause a circular import leading to problems
    if ignore is None:
        ignore = {}
    from mvpa2.base.node import Node

    assert_dict_keys_equal(x, y)
    for k in x.keys():
        v1, v2 = x[k].value, y[k].value
        assert_equal(type(v1), type(v2),
                     msg="Values for key %s have different types: %s and %s"
                         % (k, type(v1), type(v2)))
        if k in ignore:
            continue
        if isinstance(v1, np.ndarray):
            assert_array_equal(v1, v2)
        elif isinstance(v1, Node) and not (
                hasattr(v1, '__cmp__')
                or (hasattr(v1, '__eq__') and v1.__class__.__eq__ is not object.__eq__)):
            # we don't have comparators inplace for all of them yet, so test
            # based on repr and str
            assert_reprstr_equal(v1, v2)
        else:
            try:
                assert_equal(v1, v2, msg="Values for key %s have different values: %s and %s"
                                         % (k, v1, v2))
            except ValueError as e:
                ## we must be hitting some comparison issue inside (e.g. "Use a.any ..."
                ## but we do not to dive into providing comparators all around for now
                assert_reprstr_equal(v1, v2)



def assert_datasets_almost_equal(x, y, ignore_a=None, ignore_sa=None,
                                 ignore_fa=None, decimal=6):
    """
    Parameters
    ----------
    x, y: Dataset
      Two datasets that are asserted to be almost equal.
    ignore_a, ignore_sa, ignore_fa: iterable
      Differences in values of which attributes to ignore
    decimal: int or None (default: 6)
      Number of decimal up to which equality of samples is considered.
      If None, it is required that x.samples and y.samples have the same
      dtype
    """
    if ignore_a is None:
        ignore_a = {}
    if ignore_sa is None:
        ignore_sa = {}
    if ignore_fa is None:
        ignore_fa = {}
    assert_equal(type(x), type(y))
    assert_collections_equal(x.a, y.a, ignore=ignore_a)
    assert_collections_equal(x.sa, y.sa, ignore=ignore_sa)
    assert_collections_equal(x.fa, y.fa, ignore=ignore_fa)

    if decimal is None:
        assert_array_equal(x.samples, y.samples)
        assert_equal(x.samples.dtype, y.samples.dtype)
    else:
        assert_array_almost_equal(x.samples, y.samples, decimal=decimal)



def assert_datasets_equal(x, y, ignore_a=None, ignore_sa=None, ignore_fa=None):
    """
    Parameters
    ----------
    ignore_a, ignore_sa, ignore_fa: iterable
      Differences in values of which attributes to ignore
    """
    if ignore_a is None:
        ignore_a = {}
    if ignore_sa is None:
        ignore_sa = {}
    if ignore_fa is None:
        ignore_fa = {}
    assert_datasets_almost_equal(x, y, ignore_a=ignore_a, ignore_sa=ignore_sa,
                                 ignore_fa=ignore_fa, decimal=None)



if sys.version_info < (2, 7):
    # compatibility helpers for testing functions introduced in more recent versions
    # of unittest/nose

    def assert_is_instance(obj, cls, msg=None):
        assert_true(isinstance(obj, cls), msg=msg)

if externals.exists('mock'):
    import mock

    @contextmanager
    def assert_warnings(messages):
        with mock.patch("warnings.warn") as mock_warnings:
            yield

            if externals.versions['mock'] >= '0.8.0':
                mock_calls = mock_warnings.mock_calls
            else:
                # lacks leading element
                mock_calls = [(None,) + mock_warnings.call_args]

            warning_list = [(call[2]['category'], call[1][0])
                            for call in mock_calls]
            assert_equal(
                messages,
                warning_list
            )
else:
    @contextmanager
    def assert_warnings(messages):
        yield
        raise SkipTest, "install python-mock for testing either warnings were issued"



def skip_if_no_external(dep, ver_dep=None, min_version=None, max_version=None):
    """Raise SkipTest if external is missing

    Parameters
    ----------
    dep : string
      Name of the external
    ver_dep : string, optional
      If for version checking use some different key, e.g. shogun:rev.
      If not specified, `dep` will be used.
    min_version : None or string or tuple
      Minimal required version
    max_version : None or string or tuple
      Maximal required version
    """

    if not externals.exists(dep):
        raise SkipTest, \
            "External %s is not present thus tests battery skipped" % dep

    if ver_dep is None:
        ver_dep = dep

    if min_version is not None and externals.versions[ver_dep] < min_version:
        raise SkipTest, \
            "Minimal version %s of %s is required. Present version is %s" \
            ". Test was skipped." \
            % (min_version, ver_dep, externals.versions[ver_dep])

    if max_version is not None and externals.versions[ver_dep] > max_version:
        raise SkipTest, \
            "Maximal version %s of %s is required. Present version is %s" \
            ". Test was skipped." \
            % (min_version, ver_dep, externals.versions[ver_dep])



def with_tempfile(*targs, **tkwargs):
    """Decorator function to provide a temporary file name and remove it at the end.

    All arguments are passed into the call to tempfile.mktemp(), and
    resultant temporary filename is passed as the first argument into
    the test.  If no 'prefix' argument is provided, it will be
    constructed using module and function names ('.' replaced with
    '_').

    Example use::

        @with_tempfile()
        def test_write(tfile):
            open(tfile, 'w').write('silly test')
    """


    def decorate(func):
        def newfunc(*arg, **kw):
            if len(targs) < 2 and not 'prefix' in tkwargs:
                try:
                    tkwargs['prefix'] = 'tempfile_%s.%s' \
                                        % (func.__module__, func.func_name)
                except:
                    # well -- if something wrong just proceed with defaults
                    pass

            filename = tempfile.mktemp(*targs, **tkwargs)
            if __debug__:
                debug('TEST', 'Running %s with temporary filename %s'
                      % (func.__name__, filename))
            try:
                func(*(arg + (filename,)), **kw)
            finally:
                # glob here for all files with the same name (-suffix)
                # would be useful whenever we requested .img filename,
                # and function creates .hdr as well
                lsuffix = len(tkwargs.get('suffix', ''))
                filename_ = lsuffix and filename[:-lsuffix] or filename
                filenames = glob.glob(filename_ + '*')
                if len(filename_) < 3 or len(filenames) > 5:
                    # For paranoid yoh who stepped into this already ones ;-)
                    warning("It is unlikely that it was intended to remove all"
                            " files matching %r. Skipping" % filename_)
                    return
                for f in filenames:
                    try:
                        # Can also be a directory
                        if os.path.isdir(f):
                            shutil.rmtree(f)
                        else:
                            os.unlink(f)
                    except OSError:
                        pass


        newfunc = make_decorator(func)(newfunc)
        return newfunc


    return decorate



def reseed_rng():
    """Decorator to assure the use of MVPA_SEED while running the test

    It resets random number generators (both python and numpy) to the
    initial value of the seed value which was set while importing
    :mod:`mvpa`, which could be controlled through
    configuration/environment.

    Examples
    --------
    >>> @reseed_rng()
    ... def test_random():
    ...     import numpy.random as rnd
    ...     print rnd.randint(100)

    """


    def decorate(func):
        def newfunc(*arg, **kwargs):
            mvpa2.seed(mvpa2._random_seed)
            return func(*arg, **kwargs)


        newfunc = make_decorator(func)(newfunc)
        return newfunc


    return decorate



def nodebug(entries=None):
    """Decorator to temporarily turn off some debug targets

    Parameters
    ----------
    entries : None or list of string, optional
      If None, all debug entries get turned off.  Otherwise only provided
      ones
    """


    def decorate(func):
        def newfunc(*arg, **kwargs):
            if __debug__:
                from mvpa2.base import debug
                # store a copy
                old_active = debug.active[:]
                if entries is None:
                    # turn them all off
                    debug.active = []
                else:
                    for e in entries:
                        if e in debug.active:
                            debug.active.remove(e)
            try:
                res = func(*arg, **kwargs)
                return res
            finally:
                # we should return the debug states to the original
                # state regardless either test passes or not!
                if __debug__:
                    # turn debug targets back on
                    debug.active = old_active


        newfunc = make_decorator(func)(newfunc)
        return newfunc


    return decorate



def labile(niter=3, nfailures=1):
    """Decorator for labile tests -- runs multiple times

    Let's reduce probability of random failures but re-running the
    test multiple times allowing to fail few in a row.  Makes sense
    only for tests which run on random data, so usually decorated with
    reseed_rng.  Otherwise it is unlikely that result would change if
    algorithms are deterministic and operate on the same data

    Similar in idea to  https://pypi.python.org/pypi/flaky .

    Parameters
    ----------
    niter: int, optional
      How many iterations to run maximum
    nfailures: int, optional
      How many failures to allow

    """


    def decorate(func):
        def newfunc(*arg, **kwargs):
            nfailed, i = 0, 0  # define i just in case
            for i in xrange(niter):
                try:
                    ret = func(*arg, **kwargs)
                    if i + 1 - nfailed >= niter - nfailures:
                        # so we know already that we wouldn't go over
                        # nfailures
                        break
                except AssertionError, e:
                    nfailed += 1
                    if __debug__:
                        debug('TEST', "Upon %i-th run, test %s failed with %s",
                              (i, func.__name__, e))

                    if nfailed > nfailures:
                        if __debug__:
                            debug('TEST', "Ran %s %i times. Got %d failures, "
                                          "while was allowed %d "
                                          "-- re-throwing the last failure %s",
                                  (func.__name__, i + 1, nfailed, nfailures, e))
                        exc_info = sys.exc_info()
                        raise exc_info[1], None, exc_info[2]
            if __debug__:
                debug('TEST', "Ran %s %i times. Got %d failures.",
                      (func.__name__, i + 1, nfailed))
            return ret


        newfunc = make_decorator(func)(newfunc)
        return newfunc


    assert (niter > nfailures)
    return decorate



def assert_objectarray_equal(x, y, xorig=None, yorig=None, strict=True):
    """Wrapper around assert_array_equal to compare object arrays

    See http://projects.scipy.org/numpy/ticket/2117
    for the original report on oddity of dtype object arrays comparisons

    Parameters
    ----------

    strict: bool
        Assure also that dtypes are the same.  Otherwise it is pretty much
        value comparison
    """
    try:
        assert_array_equal(x, y)
    except AssertionError, e:
        if not ((x.dtype == object) and (y.dtype == object)):
            raise
        # pass inside original arrays for a meaningful assertion
        # failure msg
        if xorig is None:
            xorig = x
        if yorig is None:
            yorig = y
        try:
            # we will try harder comparing each element the same way
            # and also enforcing equal dtype
            for x_, y_ in zip(x, y):
                assert (type(x_) == type(y_))
                if strict and isinstance(x_, np.ndarray) and not (x_.dtype == y_.dtype):
                    raise AssertionError("dtypes %r and %r do not match" %
                                         (x_.dtype, y_.dtype))
                assert_objectarray_equal(x_, y_, xorig, yorig)
        except Exception, e:
            if not isinstance(e, AssertionError):
                raise AssertionError("%r != %r, thus %s != %s" %
                                     (x, y, xorig, yorig))
            raise
