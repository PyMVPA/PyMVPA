# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from numpy.testing import assert_array_equal
from nose.tools import assert_raises

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
    assert_array_equal(am.to_numeric(literal), num_default)
    assert_array_equal(am.to_literal(num_default), literal)

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
