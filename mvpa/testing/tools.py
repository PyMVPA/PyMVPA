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

# We use nose now, and we can't simply rely on numpy.testing since it
# isn't present in 1.1 available on lenny.  May be later
from nose.tools import (
    ok_, eq_,
    # Asserting (pep8-ed from unittest)
    assert_true, assert_false, assert_raises,
    assert_equal, assert_equals, assert_not_equal, assert_not_equals,
    # Decorators
    timed, with_setup, raises, istest, nottest, make_decorator )
