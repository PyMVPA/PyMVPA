# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as N

from numpy.testing import assert_array_equal
from nose.tools import assert_raises, ok_, assert_false, assert_equal

from mvpa.misc.attrmap import AttributeMap


def test_attrmap():
    map_default = {'eins': 0, 'zwei': 2, 'sieben': 1}
    map_custom = {'eins': 11, 'zwei': 22, 'sieben': 33}
    literal = ['eins', 'zwei', 'sieben', 'eins', 'sieben', 'eins']
    literal_nonmatching = ['uno', 'dos', 'tres']
    num_default = [0, 2, 1, 0, 1, 0]
    num_custom = [11, 22, 33, 11, 33, 11]

    # no custom mapping given
    am = AttributeMap()
    assert_false(am)
    ok_(len(am) == 0)
    assert_array_equal(am.to_numeric(literal), num_default)
    assert_array_equal(am.to_literal(num_default), literal)
    ok_(am)
    ok_(len(am) == 3)

    #
    # Tests for recursive mapping + preserving datatype
    class myarray(N.ndarray):
        pass

    assert_raises(KeyError, am.to_literal, [(1, 2), 2, 0])
    literal_fancy = [(1, 2), 2, [0], N.array([0, 1]).view(myarray)]
    literal_fancy_tuple = tuple(literal_fancy)
    literal_fancy_array = N.array(literal_fancy, dtype=object)

    for l in (literal_fancy, literal_fancy_tuple,
              literal_fancy_array):
        res = am.to_literal(l, recurse=True)
        assert_equal(res[0], ('sieben', 'zwei'))
        assert_equal(res[1], 'zwei')
        assert_equal(res[2], ['eins'])
        assert_array_equal(res[3], ['eins', 'sieben'])

        # types of result and subsequences should be preserved
        ok_(isinstance(res, l.__class__))
        ok_(isinstance(res[0], tuple))
        ok_(isinstance(res[1], str))
        ok_(isinstance(res[2], list))
        ok_(isinstance(res[3], myarray))

    # yet another example
    a = N.empty(1, dtype=object)
    a[0] = (0, 1)
    res = am.to_literal(a, recurse=True)
    ok_(isinstance(res[0], tuple))

    #
    # with custom mapping
    am = AttributeMap(map=map_custom)
    assert_array_equal(am.to_numeric(literal), num_custom)
    assert_array_equal(am.to_literal(num_custom), literal)

    # if not numeric nothing is mapped
    assert_array_equal(am.to_numeric(num_custom), num_custom)
    # even if the map doesn't fit
    assert_array_equal(am.to_numeric(num_default), num_default)

    # need to_numeric first
    am = AttributeMap()
    assert_raises(RuntimeError, am.to_literal, [1,2,3])
    # stupid args
    assert_raises(ValueError, AttributeMap, map=num_custom)

    # map mismatch
    am = AttributeMap(map=map_custom)
    assert_raises(KeyError, am.to_numeric, literal_nonmatching)
    # needs reset and should work afterwards
    am.clear()
    assert_array_equal(am.to_numeric(literal_nonmatching), [2, 0, 1])
    # and now reverse
    am = AttributeMap(map=map_custom)
    assert_raises(KeyError, am.to_literal, num_default)

    # dict-like interface
    am = AttributeMap()

    ok_([(k, v) for k, v in am.iteritems()] == [])


def test_attrmap_repr():
    assert_equal(repr(AttributeMap()), "AttributeMap()")
    assert_equal(repr(AttributeMap(dict(a=2, b=1))),
                 "AttributeMap({'a': 2, 'b': 1})")
    assert_equal(repr(AttributeMap(dict(a=2, b=1), mapnumeric=True)),
                 "AttributeMap({'a': 2, 'b': 1}, mapnumeric=True)")
