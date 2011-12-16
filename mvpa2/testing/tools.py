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

import os, sys
import tempfile
import unittest

import mvpa2
from mvpa2.base import externals

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
        # Decorators
        timed, with_setup, raises, istest, nottest, make_decorator )
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

def assert_array_lequal(x, y):
    assert_array_less(-y, -x)

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
            if len(targs)<2 and not 'prefix' in tkwargs:
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
                try:
                    os.unlink(filename)
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

        @reseed_rng()
        def test_random():
            import numpy.random as rnd
            print rnd.randint(100)

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

    Parameters
    ----------
    niter: int, optional
      How many iterations to run maximum
    nfailures: int, optional
      How many failures to allow

    """
    def decorate(func):
        def newfunc(*arg, **kwargs):
            nfailed, i = 0, 0           # define i just in case
            for i in xrange(niter):
                try:
                    ret = func(*arg, **kwargs)
                    #if not nfailed:
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
                                  (func.__name__, i+1, nfailed, nfailures, e))
                        exc_info = sys.exc_info()
                        raise exc_info[1], None, exc_info[2]
            if __debug__:
                debug('TEST', "Ran %s %i times. Got %d failures.",
                      (func.__name__, i+1, nfailed))
            return ret
        newfunc = make_decorator(func)(newfunc)
        return newfunc
    assert(niter > nfailures)
    return decorate
            
