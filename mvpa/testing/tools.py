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

import unittest
from mvpa.base import externals

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


def skip_if_no_external(dep, min_version=None, max_version=None):
    """Raise SkipTest if external is missing

    Parameters
    ----------
    dep : string
      Name of the external
    min_version : None or string or tuple
      Minimal required version
    max_version : None or string or tuple
      Maximal required version
    """

    if not externals.exists(dep):
        raise SkipTest, \
              "External %s is not present thus tests battery skipped" % dep

    if min_version is not None and externals.versions[dep] < min_version:
        raise SkipTest, \
              "Minimal version %s of %s is required. Present version is %s" \
              ". Test was skipped." \
              % (min_version, dep, externals.versions[dep])

    if max_version is not None and externals.versions[dep] > max_version:
        raise SkipTest, \
              "Maximal version %s of %s is required. Present version is %s" \
              ". Test was skipped." \
              % (min_version, dep, externals.versions[dep])
