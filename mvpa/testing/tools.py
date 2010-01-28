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

from mvpa.base import externals

externals.exists('nose', raiseException=True)

# We use nose now
from nose.tools import (
    ok_, eq_,
    # Asserting (pep8-ed from unittest)
    assert_true, assert_false, assert_raises,
    assert_equal, assert_equals, assert_not_equal, assert_not_equals,
    # Decorators
    timed, with_setup, raises, istest, nottest, make_decorator )

# But some pieces are useful from numpy.testing
from numpy.testing import (
    assert_almost_equal, assert_approx_equal,
    assert_array_almost_equal, assert_array_equal, assert_array_less,
    assert_string_equal)
