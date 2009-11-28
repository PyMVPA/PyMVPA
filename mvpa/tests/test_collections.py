# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for attribute collections and their collectables'''

import numpy as N
from copy import copy

from nose.tools import ok_, assert_raises, assert_false, assert_equal, assert_true

from mvpa.base.collections import Collectable

def test_basic_collectable():
    c = Collectable()

    # empty by default
    assert_equal(c.name, None)
    assert_equal(c.value, None)
    assert_equal(c.__doc__, None)

    # late assignment
    c.name = 'somename'
    c.value = 12345
    assert_equal(c.name, 'somename')
    assert_equal(c.value, 12345)

    # immediate content
    c = Collectable('value', 'myname', "This is a test")
    assert_equal(c.name, 'myname')
    assert_equal(c.value, 'value')
    assert_equal(c.__doc__, "This is a test")
    assert_equal(str(c), 'myname')

    # repr
    e = eval(repr(c))
    assert_equal(e.name, 'myname')
    assert_equal(e.value, 'value')
    assert_equal(e.__doc__, "This is a test")

    # shallow copy does not create a view of value array
    c.value = N.arange(5)
    d = copy(c)
    assert_false(d.value.base is c)

    # names starting with _ are not allowed
    assert_raises(ValueError, c._setName, "_underscore")
