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

import os
import tempfile
import unittest

from mvpa.base import externals

if __debug__:
    from mvpa.base import debug

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
                func(filename, *arg, **kw)
            finally:
                try:
                    os.unlink(filename)
                except OSError:
                    pass
        newfunc = make_decorator(func)(newfunc)
        return newfunc

    return decorate
