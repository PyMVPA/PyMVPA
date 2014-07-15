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
import sys

from mvpa2.testing.tools import assert_raises, assert_false, assert_equal, \
    assert_true,  assert_array_equal, assert_array_almost_equal, reseed_rng
from mvpa2.testing import sweepargs

from mvpa2.base.collections import Collectable, ArrayCollectable, \
        SampleAttributesCollection

from mvpa2.base.attributes import ConditionalAttribute
from mvpa2.base.node import Node
from mvpa2.measures.base import Measure, RepeatedMeasure
from mvpa2.clfs.transerror import ConfusionMatrix

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


@sweepargs(a=(
    np.arange(4),
    # note: numpy casts int(1) it into dtype=float due to presence of
    #       nan since there is no int('nan'), so float right away
    [1., np.nan],
    np.array((1, np.nan)),
    [1, None],
    [np.nan, None],
    [1, 2.0, np.nan, None, "string"],
    np.arange(6).reshape((2, -1)),      # 2d's unique
    np.array([(1, 'mom'), (2,)], dtype=object),       # elaborate object ndarray
    ))
def test_array_collectable_unique(a):
    c = ArrayCollectable(a)
    a_flat = np.asanyarray(a).ravel()
    # Since nan != nan, we better compare based on string
    # representation here
    # And sort since order of those is not guaranteed (failed test
    # on squeeze)
    def repr_(x):
        x_ = set(x)
        if sys.version_info[0] < 3:
            x_ = list(x_)
            return repr(sorted(x_))
        else:
            return repr(np.sort(x_))

    assert_equal(repr_(a_flat), repr_(c.unique))
    # even if we request it 2nd time ;)
    assert_equal(repr_(a_flat), repr_(c.unique))
    assert_equal(len(a_flat), len(c.unique))

    c2 = ArrayCollectable(list(a_flat) + [float('nan')])
    # and since nan != nan, we should get new element
    assert_equal(len(c2.unique), len(c.unique) + 1)


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


class TestNodeOffDefault(Node):
   test = ConditionalAttribute(enabled=False, doc="OffTest")
   stats = ConditionalAttribute(enabled=False, doc="OffStats")

class TestNodeOnDefault(Node):
   test = ConditionalAttribute(enabled=True, doc="OnTest")
   stats = ConditionalAttribute(enabled=True, doc="OnStats")


def test_conditional_attr():
    import copy
    import cPickle
    for node in (TestNodeOnDefault(enable_ca=['test', 'stats']),
                 TestNodeOffDefault(enable_ca=['test', 'stats'])):
        node.ca.test = range(5)
        node.ca.stats = ConfusionMatrix(labels=['one', 'two'])
        node.ca.stats.add(('one', 'two', 'one', 'two'),
                    ('one', 'two', 'two', 'one'))
        node.ca.stats.compute()

        dc_node = copy.deepcopy(node)
        assert_equal(set(node.ca.enabled), set(dc_node.ca.enabled))
        assert(node.ca['test'].enabled)
        assert(node.ca['stats'].enabled)
        assert_array_equal(node.ca['test'].value, dc_node.ca['test'].value)
        assert_array_equal(node.ca['stats'].value.matrix, dc_node.ca['stats'].value.matrix)

        # check whether values survive pickling
        pickled = cPickle.dumps(node)
        up_node = cPickle.loads(pickled)
        assert_array_equal(up_node.ca['test'].value, range(5))
        assert_array_equal(up_node.ca['stats'].value.matrix, node.ca['stats'].value.matrix)

