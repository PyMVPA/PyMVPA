# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for attribute collections and their collectables'''

import numpy as np
import copy

from mvpa2.testing.tools import assert_raises, assert_false, assert_equal, \
    assert_true,  assert_array_equal, assert_array_almost_equal, reseed_rng

from mvpa2.base.collections import Collectable, ArrayCollectable, \
        SampleAttributesCollection


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
    c.value = np.arange(5)
    d = copy.copy(c)
    assert_false(d.value.base is c.value)

    # names starting with _ are not allowed
    assert_raises(ValueError, c._set_name, "_underscore")


@reseed_rng()
def test_array_collectable():
    c = ArrayCollectable()

    # empty by default
    assert_equal(c.name, None)
    assert_equal(c.value, None)

    # late assignment
    c.name = 'somename'
    assert_raises(ValueError, c._set, 12345)
    assert_equal(c.value, None)
    c.value = np.arange(5)
    assert_equal(c.name, 'somename')
    assert_array_equal(c.value, np.arange(5))

    # immediate content
    data = np.random.random(size=(3,10))
    c = ArrayCollectable(data.copy(), 'myname',
                         "This is a test", length=3)
    assert_equal(c.name, 'myname')
    assert_array_equal(c.value, data)
    assert_equal(c.__doc__, "This is a test")
    assert_equal(str(c), 'myname')

    # repr
    from numpy import array
    e = eval(repr(c))
    assert_equal(e.name, 'myname')
    assert_array_almost_equal(e.value, data)
    assert_equal(e.__doc__, "This is a test")

    # cannot assign array of wrong length
    assert_raises(ValueError, c._set, np.arange(5))
    assert_equal(len(c), 3)

    # shallow copy DOES create a view of value array
    c.value = np.arange(3)
    d = copy.copy(c)
    assert_true(d.value.base is c.value)

    # names starting with _ are not allowed
    assert_raises(ValueError, c._set_name, "_underscore")


def test_collections():
    sa = SampleAttributesCollection()
    assert_equal(len(sa), 0)

    assert_raises(ValueError, sa.__setitem__, 'test', 0)
    l = range(5)
    sa['test'] = l
    # auto-wrapped
    assert_true(isinstance(sa['test'], ArrayCollectable))
    assert_equal(len(sa), 1)

    # names which are already present in dict interface
    assert_raises(ValueError, sa.__setitem__, 'values', range(5))

    sa_c = copy.deepcopy(sa)
    assert_equal(len(sa), len(sa_c))
    assert_array_equal(sa.test, sa_c.test)
